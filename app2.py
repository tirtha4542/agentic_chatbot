import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from main import chatBot, conn, add_pdf_to_index

# Friendly labels for tool-call announcements, keyed by the tool's actual
# name.
TOOL_DISPLAY_NAMES = {
    "search_tool": ("🔎", "Searching the web"),
    "Calculator": ("🧮", "Calculating"),
    "get_stock_price": ("📈", "Checking stock price"),
    "get_current_weather": ("🌤️", "Checking the weather"),
    "rag_tool": ("📄", "Searching the PDF"),
    "buy_stock": ("🛒", "Preparing stock purchase order"),
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
    """Read every thread_id that already has checkpoints saved in chatbot.db."""
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
            # Skip empty AI messages used for tool calls/interrupts
            if msg.content:
                history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, ToolMessage):
            # Optional: handle if you want tool outputs in history transcript
            pass
    return history


def title_for_thread(thread_id: str) -> str:
    """First user message becomes the thread's display title."""
    config = {"configurable": {"thread_id": thread_id}}
    state = chatBot.get_state(config)
    messages = state.values.get("messages", []) if state and state.values else []
    first_user_msg = next((m.content for m in messages if isinstance(m, HumanMessage)), None)
    return first_user_msg if first_user_msg else "New Chat"


def add_thread(thread_id: str, title: str = "New Chat"):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)
    st.session_state["thread_titles"].setdefault(thread_id, title)


def reset_chat():
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
    st.session_state["chat_threads"] = get_saved_thread_ids()

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {
        tid: title_for_thread(tid) for tid in st.session_state["chat_threads"]
    }

if "thread_id" not in st.session_state:
    if st.session_state["chat_threads"]:
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

# Check graph state for any active human-in-the-loop interrupts
current_state = chatBot.get_state(CONFIG)
has_interrupt = bool(current_state and current_state.tasks and any(task.interrupts for task in current_state.tasks))

if has_interrupt:
    # Extract interrupt payload
    interrupt_info = None
    for task in current_state.tasks:
        if task.interrupts:
            interrupt_info = task.interrupts[0].value
            break

    if interrupt_info:
        st.warning("🛡️ **Action Requires Your Approval**")
        st.info(
            f"**Reason:** {interrupt_info.get('reason')}\n\n"
            f"**Tool:** `{interrupt_info.get('tool')}`\n\n"
            f"**Action Details:** `{interrupt_info.get('action')}`"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Approve Order", use_container_width=True, type="primary"):
                with st.spinner("Executing approved transaction..."):
                    chatBot.invoke(Command(resume={"approved": "yes"}), config=CONFIG)
                    st.session_state["message_history"] = load_conversation(st.session_state["thread_id"])
                    st.rerun()
        with col2:
            if st.button("❌ Reject Order", use_container_width=True):
                with st.spinner("Cancelling transaction..."):
                    chatBot.invoke(Command(resume={"approved": "no"}), config=CONFIG)
                    st.session_state["message_history"] = load_conversation(st.session_state["thread_id"])
                    st.rerun()

# Disable input when waiting for approval
user_input = st.chat_input(
    "Ask a question, or click + to attach a PDF for the knowledge base",
    accept_file=True,
    file_type=["pdf"],
    disabled=has_interrupt,
)

if user_input and not has_interrupt:
    user_text = (user_input.text or "").strip()
    uploaded_files = user_input.files or []

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

    if user_text:
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
                if "cannot schedule new futures after shutdown" in str(e):
                    final_text = final_text or (
                        "⚠️ Response interrupted by an app restart. Please resend your message."
                    )
                else:
                    raise

            text_placeholder.markdown(final_text)
            ai_message = final_text

        # Check if the execution stopped because of an approval interrupt
        post_state = chatBot.get_state(CONFIG)
        if post_state and post_state.tasks and any(task.interrupts for task in post_state.tasks):
            st.rerun()

        if ai_message:
            st.session_state["message_history"].append({"role": "assistant", "content": ai_message})