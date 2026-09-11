from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from dotenv import load_dotenv

load_dotenv()


class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.5, max_tokens=500)
conn = sqlite3.connect(database="chatbot.db",check_same_thread=False)
checkpoint = SqliteSaver(conn)


def chatbot(state: chatState):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


graph = StateGraph(chatState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
chatBot = graph.compile(checkpointer=checkpoint)


# Everything below only runs when you do `python main.py` directly,
# NOT when Streamlit (or anything else) imports chatBot from this file.
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
            if chunk.content:
                print(chunk.content, end="", flush=True)
                bot_response += chunk.content

        print()