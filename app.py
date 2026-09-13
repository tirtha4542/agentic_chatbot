"""
Streamlit UI for the LangGraph + Groq chatbot (main.py), updated for the
SqliteSaver checkpointer and the tool-calling graph (search / calculator /
stock / weather).

Changes vs. the MemorySaver version:
- Thread list is no longer only kept in st.session_state (which is lost
  whenever the Streamlit process restarts). Since main.py now persists
  checkpoints to chatbot.db via SqliteSaver, we rebuild the sidebar's
  thread list FROM the database on startup, so old conversations are
  still there after a restart -- not just after a page refresh.
- Thread titles are derived from each thread's first human message
  (read back from the checkpointer) instead of a title dict that only
  lived in memory for the current session.
- The assistant's turn now shows its work: when the model calls a tool,
  a "🔧 Calling tool: `name`" line appears as soon as that decision is
  made (parsed from the streamed tool_call_chunks), and that tool's
  output appears in an expander as soon as it's ready -- both before the
  final streamed answer, which still updates token-by-token like before.
- The chat input now has a `+` attach icon (via accept_file=True) for
  uploading a PDF directly in the input field. An uploaded PDF is
  ingested into the RAG index right away (main.add_pdf_to_index) and
  confirmed in the chat, so rag_tool can retrieve from it on the very
  next question -- no restart needed.
- Everything else (per-thread config, switching threads) works the same,
  since chatBot.stream / chatBot.get_state have the same interface
  regardless of which checkpointer backend or graph nodes are used.
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from main import chatBot, conn, add_pdf_to_index

# Friendly labels for tool-call announcements, keyed by the tool's actual
# name (as bound to the LLM / seen in ToolMessage.name). Falls back to the
# raw tool name for anything not listed here, so adding a new tool in
# main.py never breaks this UI -- it just won't have a custom label yet.
TOOL_DISPLAY_NAMES = {
    "search_tool": ("🔎", "Searching the web"),
    "Calculator": ("🧮", "Calculating"),
    "get_stock_price": ("📈", "Checking stock price"),
    "get_current_weather": ("🌤️", "Checking the weather"),
    "rag_tool": ("📄", "Searching the PDF"),
}


def tool_display(tool_name: str) -> str:
    icon, label = TOOL_DISPLAY_NAMES.get(tool_name, ("🔧", f"Calling `{tool_name}`"))
    return f"{icon} *{label}...*"

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        [data-testid="stSidebar"] button { text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def generate_thread_id() -> str:
    return str(uuid.uuid4())


def get_saved_thread_ids() -> list[str]:
    """
    Read every thread_id that already has checkpoints saved in chatbot.db,
    most-recently-active first.

    langgraph's SqliteSaver stores rows in a `checkpoints` table with a
    `checkpoint_id` that's time-sortable, so grouping + ordering by the max
    checkpoint_id per thread gives us recency order "for free".
    """
    try:
        cur = conn.execute(
            """
            SELECT thread_id, MAX(checkpoint_id) AS latest
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY latest DESC
            """
        )
        return [row[0] for row in cur.fetchall()]
    except Exception:
        # Table doesn't exist yet (brand new / empty database).
        return []


def load_conversation(thread_id: str):
    """Pull message history for a thread out of the LangGraph checkpointer."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chatBot.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []

    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})
    return history


def title_for_thread(thread_id: str) -> str:
    """First user message becomes the thread's display title."""
    history = load_conversation(thread_id)
    first_user_msg = next((m["content"] for m in history if m["role"] == "user"), None)
    return first_user_msg if first_user_msg else "New Chat"


def add_thread(thread_id: str, title: str = "New Chat"):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)
    st.session_state["thread_titles"].setdefault(thread_id, title)


def reset_chat():
    """Start a brand new conversation thread."""
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    add_thread(thread_id)


def switch_thread(thread_id: str):
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = load_conversation(thread_id)


def thread_label(thread_id: str) -> str:
    title = st.session_state["thread_titles"].get(thread_id, "New Chat")
    return title if len(title) <= 28 else title[:28] + "…"


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
if "chat_threads" not in st.session_state:
    # Rebuild from the sqlite db -- this is what makes old chats survive
    # a full app / server restart, not just a page refresh.
    st.session_state["chat_threads"] = get_saved_thread_ids()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {
        tid: title_for_thread(tid) for tid in st.session_state["chat_threads"]
    }

if "thread_id" not in st.session_state:
    if st.session_state["chat_threads"]:
        # Resume the most recently active thread.
        st.session_state["thread_id"] = st.session_state["chat_threads"][0]
        st.session_state["message_history"] = load_conversation(
            st.session_state["thread_id"]
        )
    else:
        reset_chat()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = load_conversation(st.session_state["thread_id"])

CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🤖 Agentic Chatbot")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.subheader("Conversations")

    if not st.session_state["chat_threads"]:
        st.caption("No conversations yet.")

    for tid in st.session_state["chat_threads"]:
        is_active = tid == st.session_state["thread_id"]
        label = ("🟢 " if is_active else "💬 ") + thread_label(tid)
        if st.button(label, key=f"thread_{tid}", use_container_width=True):
            if not is_active:
                switch_thread(tid)
                st.rerun()


# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.title("Agentic Chatbot with LangGraph")
st.caption(f"Thread: `{st.session_state['thread_id']}`")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Ask a question, or click + to attach a PDF for the knowledge base",
    accept_file=True,
    file_type=["pdf"],
)

if user_input:
    # With accept_file=True, chat_input returns an object with .text and
    # .files instead of a plain string -- unpack both.
    user_text = (user_input.text or "").strip()
    uploaded_files = user_input.files or []

    # Ingest any attached PDFs into the RAG index first, showing each as
    # its own exchange in the chat so it's clear what got added and when.
    for f in uploaded_files:
        upload_note = f"📎 Uploaded **{f.name}**"
        with st.chat_message("user"):
            st.markdown(upload_note)
        st.session_state["message_history"].append({"role": "user", "content": upload_note})

        with st.chat_message("assistant"):
            with st.spinner(f"Indexing {f.name}..."):
                num_chunks = add_pdf_to_index(f.read(), filename=f.name)
            confirmation = (
                f"✅ Added **{f.name}** to the knowledge base ({num_chunks} chunks). "
                "You can ask questions about it now."
            )
            st.markdown(confirmation)
        st.session_state["message_history"].append(
            {"role": "assistant", "content": confirmation}
        )

    # Only run the chat graph if the user actually typed something --
    # a PDF-only submission (no question) just indexes the file above.
    if user_text:
        # Give the thread a real title the first time it's used, and make
        # sure it's tracked in the sidebar (covers the case where it was
        # just created via reset_chat and has no checkpoints yet).
        if st.session_state["thread_titles"].get(st.session_state["thread_id"]) == "New Chat":
            st.session_state["thread_titles"][st.session_state["thread_id"]] = user_text
        add_thread(st.session_state["thread_id"], user_text)

        st.session_state["message_history"].append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        with st.chat_message("assistant"):
            text_placeholder = st.empty()
            seen_tool_call_ids = set()
            final_text = ""

            # Walk the raw stream ourselves instead of handing it straight to
            # st.write_stream, so we can react differently depending on which
            # node produced each chunk:
            #   - "chatbot" node, tool_call_chunks present -> the model just
            #     decided to call a tool; announce it once per tool_call id.
            #   - "chatbot" node, plain content -> the model's actual answer;
            #     stream it token-by-token like before.
            #   - "tools" node -> a completed ToolMessage with that tool's
            #     output; show it immediately (not streamed token-by-token,
            #     since tool output isn't generated by the LLM -- it arrives
            #     as one finished value from the Python function -- but it's
            #     still shown live, as soon as it's ready, rather than only
            #     after the whole run finishes).
            try:
                for message_chunk, metadata in chatBot.stream(
                    {"messages": [HumanMessage(content=user_text)]},
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    node = metadata.get("langgraph_node")

                    if node == "chatbot":
                        for tc in getattr(message_chunk, "tool_call_chunks", None) or []:
                            tc_id = tc.get("id")
                            tc_name = tc.get("name")
                            if tc_name and tc_id and tc_id not in seen_tool_call_ids:
                                seen_tool_call_ids.add(tc_id)
                                st.markdown(tool_display(tc_name))

                        if message_chunk.content:
                            final_text += message_chunk.content
                            text_placeholder.markdown(final_text + "▌")

                    elif node == "tools":
                        tool_name = getattr(message_chunk, "name", None) or "tool"
                        icon, label = TOOL_DISPLAY_NAMES.get(tool_name, ("📄", f"Output from `{tool_name}`"))
                        with st.expander(f"{icon} {label} result", expanded=False):
                            st.code(str(message_chunk.content), language="text")
            except RuntimeError as e:
                # Streamlit tore down this script run mid-stream -- e.g. a
                # source file was saved and autoreload restarted the app
                # while this response was still generating -- which can
                # race with LangGraph's internal thread pool shutting down.
                # Not a bug in the graph itself; show a clean message
                # instead of a traceback if this slips through.
                if "cannot schedule new futures after shutdown" in str(e):
                    final_text = final_text or (
                        "⚠️ Response interrupted by an app restart. Please resend your message."
                    )
                else:
                    raise

            text_placeholder.markdown(final_text)
            ai_message = final_text

        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})