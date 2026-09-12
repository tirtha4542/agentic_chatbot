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


tools = [search_tool, Calculator, get_stock_price, get_current_weather]
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


def _invoke_with_rate_limit_retry(model, messages, max_retries: int = 3):
    """
    Call model.invoke(messages), retrying on Groq's 429 rate-limit errors.

    Groq's error message includes how long to wait (e.g. "try again in
    21.06s") -- we parse that and sleep for it (plus a small buffer)
    instead of failing the whole request outright. This is different from
    the tool_use_failed retry below: that one is a malformed-generation
    issue, this one is "you're over budget, wait and resubmit the exact
    same request."
    """
    wait_pattern = re.compile(r"try again in ([\d.]+)s")
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            match = wait_pattern.search(str(e))
            wait_seconds = float(match.group(1)) + 1 if match else 5.0
            time.sleep(wait_seconds)
    # Unreachable, but keeps type checkers happy.
    raise RuntimeError("Exceeded max retries for rate-limited request.")


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
            fallback = _invoke_with_rate_limit_retry(llm, trimmed)
            response = AIMessage(
                content=(
                    "(A tool call failed to generate correctly, so here's a "
                    "plain answer instead.)\n\n" + fallback.content
                )
            )
        else:
            raise

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