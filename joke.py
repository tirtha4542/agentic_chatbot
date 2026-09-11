from langchain.chat_models import init_chat_model
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = init_chat_model(
    "gemini-3.6-flash",
    model_provider="google_genai",
    temperature=0.7,
    timeout=30,
    max_tokens=1000,   # was cutting off mid-sentence at 500
    max_retries=6,
)

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

def extract_text(content) -> str:
    """Newer langchain-google-genai returns content as a list of blocks
    (e.g. [{'type': 'text', 'text': '...', 'extras': {...}}]) instead of
    a plain string. Normalize either shape to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()

def GEN_JOKE(state: JokeState):
    prompt = HumanMessage(
        content=f"You are a funny assistant to generate a joke based on the topic provided: "
                f"Topic {state['topic']}. Provide ONLY the joke text."
    )
    res = llm.invoke([prompt])
    return {"joke": extract_text(res.content)}

def GEN_EXP(state: JokeState):
    prompt = HumanMessage(
        content=f"You are an expert to explain a joke perfectly. Do not add additional conversational "
                f"text, only the explanation about this joke that is realistic and funny:\n\njoke: {state['joke']}"
    )
    res = llm.invoke([prompt])
    return {"explanation": extract_text(res.content)}

checkpointer = MemorySaver()

graph = StateGraph(JokeState)
graph.add_node("GEN_JOKE", GEN_JOKE)
graph.add_node("GEN_EXP", GEN_EXP)
graph.add_edge(START, "GEN_JOKE")
graph.add_edge("GEN_JOKE", "GEN_EXP")
graph.add_edge("GEN_EXP", END)

wf = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
init = {"topic": "football"}

result = wf.invoke(init, config=config)

print(f"""
{'='*40}
THEME: {init['topic'].upper()}
{'='*40}

[ JOKE ]
{result['joke']}

----------------------------------------

[ EXPLANATION ]
{result['explanation']}

{'='*40}
""")