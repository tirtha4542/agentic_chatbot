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
"""

import json
import math
import os
import re
import sqlite3
import time

from dotenv import load_dotenv
from groq import BadRequestError, RateLimitError
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
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
from tavily import TavilyClient
from typing import Annotated, TypedDict
import requests
import tiktoken

load_dotenv()

llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.5, max_tokens=900)
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
    try:
        result = tavily_client.search(
            query=query,
            max_results=5,
            topic="general",
            search_depth="advanced",
        )
        return _truncate(json.dumps(result))
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
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return json.loads(_truncate(json.dumps(data)))
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
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": units}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return {
            "city": data["name"],
            "country": data["sys"]["country"],
            "temperature": data["main"]["temp"],
            "feels_like": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "weather": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"],
        }
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


tools = [search_tool, Calculator, get_stock_price, get_current_weather, rag_tool]
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


tool_node = ToolNode(tools)

graph = StateGraph(chatState)
graph.add_node("chatbot", chatbot)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chatbot")
graph.add_conditional_edges("chatbot", tools_condition)
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

        print("Bot: ", end="", flush=True)
        bot_response = ""

        for chunk, metadata in chatBot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages",
        ):
            # Only print tokens from the chatbot node -- otherwise raw tool
            # output (search JSON, weather dicts, etc.) from the "tools"
            # node gets streamed and printed too, which is noisy.
            if metadata.get("langgraph_node") != "chatbot":
                continue
            if chunk.content:
                print(chunk.content, end="", flush=True)
                bot_response += chunk.content

        print()