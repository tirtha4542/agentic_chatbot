"""
Merged backend: combines the two versions you had.

From main.py:      SqliteSaver persistence, per-thread config, streaming
                    loop, and the `chatBot` export used by the Streamlit UI.
From test_tool.py: the tool-calling graph (search / calculator / stock /
                    weather), with the fixes already made in that file:
                      - Calculator's eval() actually sandboxes builtins now
                        (was "__builtines__", a typo that let builtins
                        through -- also the except clause was invalid,
                        `except expression as e`, since expression is a str).
                      - search_tool is a minimal-schema wrapper instead of
                        the full TavilySearch class, to avoid its large
                        bound-tool schema eating into Groq's tokens/minute
                        limit on every request.
                      - Tool outputs (search / stock / weather) are
                        truncated so a single big result can't push the
                        trimmed message window empty.
                      - trim_messages uses a lightweight tiktoken counter
                        instead of the model's fallback tokenizer (which
                        needs the heavy `transformers` package).
                      - chatbot() falls back to the untrimmed history if
                        trimming ever produces an empty list (Groq rejects
                        empty message lists outright), and retries once
                        without tool-binding if Groq reports a malformed
                        tool_call (`tool_use_failed`), which happens
                        occasionally with Qwen on Groq.

Caching added:
  - LLM responses are cached to llm_cache.db via langchain's SQLiteCache,
    so an identical (prompt -> response) pair costs zero tokens on a repeat.
  - search_tool / get_stock_price / get_current_weather results are cached
    to a local `tool_cache/` directory via diskcache, with short TTLs
    (5-10 min) since these values go stale -- avoids hammering Tavily /
    Alpha Vantage / OpenWeatherMap (each with their own free-tier limits)
    for a query that was just answered.

Human-in-the-loop approval added:
  - A new `buy_stock` tool represents placing a real order (mocked here --
    swap the body for a real brokerage API call). Because this is an
    action with real-world consequences (unlike the read-only tools),
    the graph routes any `buy_stock` tool call through an `approval` node
    first, which calls `interrupt()` and pauses the graph until the human
    responds with approve/reject via `Command(resume=...)`. Only after
    approval does execution continue to the normal ToolNode. If rejected,
    the pending tool_call is answered with a ToolMessage saying so (so the
    next LLM call doesn't choke on an unresolved tool_call), and the flow
    returns straight to the chatbot node instead of executing the trade.
"""

import json
import math
import os
import re
import sqlite3
import time

from dotenv import load_dotenv
from groq import BadRequestError, RateLimitError
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from tavily import TavilyClient
from typing import Annotated, TypedDict
from diskcache import Cache
import requests
import tiktoken

load_dotenv()

# --------------------------------------------------------------------------
# Caching
# --------------------------------------------------------------------------
# 1) LLM response cache: identical (prompt -> response) pairs are served
#    from a local SQLite file instead of calling Groq again -- free, and
#    directly cuts token usage against the rate limits we've been hitting.
set_llm_cache(SQLiteCache(database_path="llm_cache.db"))

# 2) Tool output cache: search / stock / weather all hit external APIs
#    that have their own rate limits, and the same query is often repeated
#    (testing, or a user asking the same thing twice). diskcache persists
#    to disk (survives restarts) and supports a per-entry TTL.
tool_cache = Cache("tool_cache")

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.5, max_tokens=900)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# --------------------------------------------------------------------------
# RAG index (PDF -> chunks -> embeddings -> FAISS)
# --------------------------------------------------------------------------
# Configurable so this isn't hardcoded to one machine's absolute path; set
# RAG_PDF_PATH in your .env if you want a different source document.
_RAG_PDF_PATH = os.getenv(
    "RAG_PDF_PATH",
    "F:/newAgent/Data/The Prevalence of Sleep Disorders in College Students  Impact on Academic Performance.pdf",
)
_FAISS_INDEX_DIR = "faiss_index"

_embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

if os.path.isdir(_FAISS_INDEX_DIR):
    # Reuse the index saved on a previous run instead of re-loading the PDF
    # and re-embedding every chunk on every startup.
    vector_store = FAISS.load_local(
        _FAISS_INDEX_DIR,
        _embedding,
        allow_dangerous_deserialization=True,
    )
else:
    loader = PyPDFLoader(file_path=_RAG_PDF_PATH)
    data = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(data)
    vector_store = FAISS.from_documents(chunks, _embedding)
    vector_store.save_local(_FAISS_INDEX_DIR)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def add_pdf_to_index(file_bytes: bytes, filename: str = "uploaded.pdf") -> int:
    """
    Ingest a PDF (as raw bytes, e.g. from a Streamlit file upload) into the
    existing FAISS index so rag_tool can retrieve from it immediately, and
    persist the updated index to disk. Returns the number of chunks added.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(file_path=tmp_path)
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(data)
        for chunk in chunks:
            # Replace the temp file's path with the original filename so
            # citations in rag_tool's output are meaningful to the user.
            chunk.metadata["source"] = filename

        vector_store.add_documents(chunks)
        vector_store.save_local(_FAISS_INDEX_DIR)
        return len(chunks)
    finally:
        os.remove(tmp_path)


# --------------------------------------------------------------------------
# Tool output size guard
# --------------------------------------------------------------------------
_MAX_TOOL_OUTPUT_CHARS = 2000


def _truncate(text: str, limit: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} more chars]"


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
@tool
def search_tool(query: str) -> str:
    """Search the web for current information. Input should be a plain search query string."""
    cache_key = f"search:{query.strip().lower()}"
    if cache_key in tool_cache:
        return tool_cache[cache_key]
    try:
        result = tavily_client.search(
            query=query,
            max_results=5,
            topic="general",
            search_depth="advanced",
        )
        output = _truncate(json.dumps(result))
        tool_cache.set(cache_key, output, expire=600)  # 10 min -- news/search results go stale
        return output
    except Exception as e:
        return f"Search failed: {e}"


@tool
def Calculator(expression: str) -> str:
    """Useful for simple math calculations. Input should be a valid math expression.
    Example: 2+2, math.sqrt(16), 10*5"""
    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given ticker symbol using Alpha Vantage."""
    cache_key = f"stock:{symbol.strip().upper()}"
    if cache_key in tool_cache:
        return tool_cache[cache_key]
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        result = json.loads(_truncate(json.dumps(data)))
        tool_cache.set(cache_key, result, expire=300)  # 5 min -- prices move
        return result
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


@tool
def get_current_weather(city: str, units: str = "metric"):
    """
    Fetch current weather for a given city using OpenWeatherMap API.

    Args:
        city (str): City name, e.g. "Dhaka" or "Dhaka,BD"
        units (str): "metric" (Celsius), "imperial" (Fahrenheit), or "standard" (Kelvin)

    Returns:
        dict: Parsed weather data, or an error dict if the request failed
    """
    cache_key = f"weather:{city.strip().lower()}:{units}"
    if cache_key in tool_cache:
        return tool_cache[cache_key]

    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": units}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        result = {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "weather": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
        }
        tool_cache.set(cache_key, result, expire=600)  # 10 min -- weather doesn't change that fast
        return result
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except KeyError as e:
        return {"error": f"Unexpected response format, missing key: {e}"}


@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document that has been
    indexed for this app. Use this tool when the user asks a factual or
    conceptual question that may be answered by the stored PDF content.

    Args:
        query: The question or search query used to retrieve PDF content.
    """
    documents = retriever.invoke(query)
    if not documents:
        return "No relevant information was found in the PDF."

    formatted = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page", "unknown")
        formatted.append(
            f"Document: {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )
    return _truncate("\n\n".join(formatted))


@tool
def buy_stock(symbol: str, quantity: int) -> dict:
    """
    Place a buy order for a stock ticker. Use this ONLY when the user
    explicitly asks to buy/purchase shares, e.g. "buy 10 shares of TSLA"
    or "please buy 20 BMW stock". Do not use this for price lookups --
    use get_stock_price for that.

    Args:
        symbol: Ticker symbol to buy, e.g. "TSLA", "BMW".
        quantity: Number of shares to buy.
    """
    # NOTE: this is a mocked/simulated execution. Swap the body of this
    # function for a real brokerage API call (Alpaca, Interactive Brokers,
    # etc.) when you're ready to place real orders. The approval gate in
    # the graph (see `approval_node` below) runs regardless of what this
    # function actually does, so wiring in a real API later is a drop-in
    # change -- no graph changes needed.
    return {
        "status": "executed",
        "symbol": symbol.strip().upper(),
        "quantity": quantity,
        "note": "Simulated order -- no real trade was placed.",
    }


# Tool names that must be approved by a human before they run. buy_stock is
# the only one right now, but any future action-with-consequences tool
# (sell_stock, send_email, place_order, ...) can just be added here.
_APPROVAL_REQUIRED_TOOLS = {"buy_stock"}

tools = [search_tool, Calculator, get_stock_price, get_current_weather, rag_tool, buy_stock]
llm_with_tools = llm.bind_tools(tools=tools)

# Lightweight token counter for trim_messages -- avoids depending on the
# heavy `transformers` package that the model's fallback tokenizer needs.
_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(_encoding.encode(content))
    return total


# --------------------------------------------------------------------------
# Graph state + persistence
# --------------------------------------------------------------------------
class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)


_WAIT_TIME_PATTERN = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")

# Per-minute limits give short waits (seconds) worth auto-retrying through.
# Daily (TPD) limits give waits of several minutes -- blocking a Streamlit
# request that long is a bad idea, so above this threshold we surface a
# friendly message instead of sleeping.
_MAX_AUTO_RETRY_WAIT_SECONDS = 60.0


def _parse_wait_seconds(error_message: str) -> float | None:
    """
    Parse Groq's "try again in ...s" wait time, which comes in two formats:
    "try again in 21.06s" (seconds only) or "try again in 16m13.296s"
    (minutes + seconds, seen on daily/TPD limits). Returns None if the
    message doesn't match either format.
    """
    match = _WAIT_TIME_PATTERN.search(error_message)
    if not match:
        return None
    minutes = float(match.group(1)) if match.group(1) else 0.0
    seconds = float(match.group(2))
    return minutes * 60 + seconds


def _invoke_with_rate_limit_retry(model, messages, max_retries: int = 3):
    """
    Call model.invoke(messages), retrying on Groq's 429 rate-limit errors --
    but only for short (per-minute) waits. Longer waits (typically a daily
    token cap) are re-raised so the caller can show a friendly message
    instead of blocking the request for several minutes.
    """
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except RateLimitError as e:
            wait_seconds = _parse_wait_seconds(str(e))
            if wait_seconds is None:
                wait_seconds = 5.0
            if wait_seconds > _MAX_AUTO_RETRY_WAIT_SECONDS:
                raise
            if attempt == max_retries - 1:
                raise
            time.sleep(wait_seconds + 1)
    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("Exceeded max retries for rate-limited request.")


def _rate_limit_message(e: RateLimitError) -> AIMessage:
    """Friendly, non-crashing response for a rate limit we won't auto-retry."""
    wait_seconds = _parse_wait_seconds(str(e))
    if wait_seconds is not None:
        minutes, seconds = divmod(int(wait_seconds), 60)
        wait_str = f"{minutes}m{seconds}s" if minutes else f"{seconds}s"
        wait_note = f"Please try again in about {wait_str}."
    else:
        wait_note = "Please try again shortly."
    return AIMessage(
        content=(
            "⚠️ I've hit Groq's rate limit for this model/tier and can't "
            f"respond right now. {wait_note} "
            "(You can also upgrade your Groq tier for more headroom.)"
        )
    )


def chatbot(state: chatState):
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,
        strategy="last",
        token_counter=count_tokens,
        include_system=True,
        start_on="human",
    )

    # Safety net: never send an empty message list to Groq (it rejects it
    # outright) if trimming ever strips everything.
    if not trimmed:
        trimmed = state["messages"]

    try:
        response = _invoke_with_rate_limit_retry(llm_with_tools, trimmed)
    except BadRequestError as e:
        # Qwen occasionally emits a malformed tool_call on Groq; retry once
        # without tool binding instead of crashing the whole run.
        if "tool_use_failed" in str(e):
            try:
                fallback = _invoke_with_rate_limit_retry(llm, trimmed)
                response = AIMessage(
                    content=(
                        "(A tool call failed to generate correctly, so here's a "
                        "plain answer instead.)\n\n" + fallback.content
                    )
                )
            except RateLimitError as rate_err:
                response = _rate_limit_message(rate_err)
        else:
            raise
    except RateLimitError as e:
        # A long wait (e.g. daily token cap) wasn't auto-retried -- show a
        # friendly message in the chat instead of crashing the app.
        response = _rate_limit_message(e)

    return {"messages": [response]}


def approval_node(state: chatState):
    """
    Runs only when the last AIMessage contains a tool_call for one of the
    _APPROVAL_REQUIRED_TOOLS (currently just buy_stock). Pauses the graph
    with interrupt() and waits for a human decision delivered via
    Command(resume={"approved": "yes"/"no"}).

    - If approved: returns no new messages, and routing sends the state on
      to the real ToolNode, which executes buy_stock normally.
    - If rejected: answers the pending tool_call directly with a
      ToolMessage so the model isn't left with a dangling, unanswered
      tool_call (which would break the next LLM turn), and routing sends
      the state straight back to the chatbot node -- no trade is executed.
    """
    last = state["messages"][-1]
    call = next(
        c for c in last.tool_calls if c["name"] in _APPROVAL_REQUIRED_TOOLS
    )

    decision = interrupt(
        {
            "type": "approval",
            "reason": "Confirm this stock purchase before it is executed.",
            "action": call["args"],
            "tool": call["name"],
            "instruction": "Approve this purchase? yes/no",
        }
    )

    approved = str(decision.get("approved", "")).strip().lower() in ("yes", "y", "true")

    if not approved:
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"Purchase request {call['args']} was NOT approved "
                        "by the user. No order was placed."
                    ),
                    tool_call_id=call["id"],
                )
            ]
        }

    # Approved -- no message to add here; the real ToolNode will execute
    # buy_stock and append its ToolMessage next.
    return {"messages": []}


def route_after_chatbot(state: chatState):
    """
    After the chatbot node runs: if it made no tool calls, end the turn.
    If any tool call is on the approval list, go through approval first.
    Otherwise go straight to the normal tool node (search/calc/etc. never
    need approval).
    """
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None)
    if not tool_calls:
        return END
    if any(c["name"] in _APPROVAL_REQUIRED_TOOLS for c in tool_calls):
        return "approval"
    return "tools"


def route_after_approval(state: chatState):
    """
    If approval_node already answered the tool_call itself (the rejection
    path, which appends a ToolMessage), skip the real tool node entirely
    and let the chatbot respond to the rejection. Otherwise (approved),
    proceed to the real tool node to actually execute buy_stock.
    """
    if isinstance(state["messages"][-1], ToolMessage):
        return "chatbot"
    return "tools"


tool_node = ToolNode(tools)

graph = StateGraph(chatState)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)
graph.add_node("approval", approval_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges(
    "chatbot",
    route_after_chatbot,
    {"approval": "approval", "tools": "tools", END: END},
)
graph.add_conditional_edges(
    "approval",
    route_after_approval,
    {"tools": "tools", "chatbot": "chatbot"},
)
graph.add_edge("tools", "chatbot")
graph.add_edge("chatbot", END)

chatBot = graph.compile(checkpointer=checkpoint)


# --------------------------------------------------------------------------
# Everything below only runs when you do `python main.py` directly,
# NOT when Streamlit (or anything else) imports chatBot from this file.
# --------------------------------------------------------------------------
if __name__ == "__main__":
    thread_id = "default_thread"
    config = {"configurable": {"thread_id": thread_id}}

    print("Enter Exit to quit")
    while True:
        user_input = input("Enter Your query: ")
        print("User:", user_input)

        if user_input.strip().lower() == "exit":
            break

        # Note: interrupt() doesn't play nicely with .stream() -- when the
        # graph hits an approval interrupt, streaming just stops without
        # producing a final answer. So for a turn that might need approval,
        # we invoke() (which returns cleanly with __interrupt__ set) instead
        # of streaming. If you want token-by-token streaming back for the
        # non-approval-needed case, you could try .stream() first and only
        # fall back to this loop when __interrupt__ shows up -- kept simple
        # here for clarity.
        result = chatBot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        while "__interrupt__" in result:
            info = result["__interrupt__"][0].value
            print(f"\n[Approval needed] {info['reason']}")
            print(f"Tool: {info.get('tool')}  Action: {info.get('action')}")
            answer = input(f"{info['instruction']} ")
            result = chatBot.invoke(
                Command(resume={"approved": answer}),
                config=config,
            )

        print("Bot:", result["messages"][-1].content)