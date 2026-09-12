from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, trim_messages
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from groq import BadRequestError
import requests
import math
import os
import json
import tiktoken

from tavily import TavilyClient

load_dotenv()

llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.5, max_tokens=1024)
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# Tool outputs (raw search JSON, stock quotes, weather payloads) can be big
# enough on their own to blow the trim_messages token budget and push the
# original human message out of the window -- cap them so that can't happen.
_MAX_TOOL_OUTPUT_CHARS = 2000


def _truncate(text: str, limit: int = _MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated, {len(text) - limit} more chars]"


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

# Custom, minimal-schema search tool instead of the full TavilySearch class.
# TavilySearch's built-in schema (search_depth, topic, time_range,
# include/exclude domains, images, etc.) gets serialized into every request
# once bound to the LLM, and was likely the main contributor to blowing past
# Groq's 7,000 input-tokens/minute limit on a single, short conversation.
@tool
def web_search(query: str) -> str:
    """Search the web for current information. Input should be a plain search query string."""
    try:
        result = tavily_client.search(query=query, max_results=3)
        return _truncate(str(result))
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
        # Fixed: was `{"__builtines__": {}}` (typo'd key, so builtins were
        # never actually blocked) -- correct key is "__builtins__".
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        # Fixed: original caught `except expression as e`, which is invalid
        # (expression is a str, not an exception type) and would raise a
        # TypeError instead of handling bad input gracefully.
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
        # Alpha Vantage's "Global Quote" payload is already small, but be
        # defensive in case of an unexpected large error/info payload.
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
        dict: Parsed weather data, or None if the request failed
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

tools = [web_search, Calculator, get_stock_price, get_current_weather]
llm_with_tools = llm.bind_tools(tools=tools)

# Lightweight token counter for trim_messages, so we don't depend on the
# heavy `transformers` package (which trim_messages otherwise falls back to
# when a chat model has no native token counter -- and Qwen via Groq doesn't).
# cl100k_base is an approximation for non-OpenAI models, but it's close
# enough for the purpose of staying under a token/minute budget.
_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = m.content if isinstance(m.content, str) else str(m.content)
        total += len(_encoding.encode(content))
    return total


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chatbot(state):
    # Keeps later, longer conversations under the token-per-minute limit too
    # (the slimmed web_search tool fixes the *first-call* 413, this fixes
    # it for turn 20+ once history grows).
    trimmed = trim_messages(
        state["messages"],
        max_tokens=4000,
        strategy="last",
        token_counter=count_tokens,
        include_system=True,
        start_on="human",
    )

    # Safety net: if a bulky tool result pushed the human message itself out
    # of the trim window, start_on="human" can strip everything and leave an
    # empty list, which Groq's API rejects outright ("minimum number of
    # items is 1"). Fall back to the untrimmed history rather than send
    # nothing -- with tool outputs now capped above, this should be rare.
    if not trimmed:
        trimmed = state["messages"]

    # Qwen's tool-calling on Groq occasionally emits a malformed / truncated
    # tool_call and Groq rejects the whole request with a 400
    # ("Failed to call a function"). Retry once without tool binding so the
    # conversation doesn't just crash -- the user gets a plain-text answer
    # instead of a tool result for that turn.
    try:
        response = llm_with_tools.invoke(trimmed)
    except BadRequestError as e:
        if "tool_use_failed" in str(e):
            fallback = llm.invoke(trimmed)
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

wf = graph.compile()


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
if __name__ == "__main__":
    init = {"messages": [HumanMessage(content="what is the stock price of BMW today?")]}
    result = wf.invoke(init)
    print(result["messages"][-1].content)

    while True:
        print("Press EXIT to return.")
        user_input = input("Enter the query: ")
        if user_input.lower() == "exit":
            print("Good bye")
            break

        # Fixed: was `{"messages": user_input}` -- a raw string instead of a
        # wrapped HumanMessage, which breaks the add_messages reducer.
        init = {"messages": [HumanMessage(content=user_input)]}
        result = wf.invoke(init)
        print("AI:", result["messages"][-1].content)