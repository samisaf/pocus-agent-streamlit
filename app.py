"""
POCUS Tutor (Streamlit)
===========================================
An application that delivers an AI-powered tutoring experience for physicians and
health care professionals learning point-of-care ultrasound (POCUS). 
The app presents curated content and offers an inapp chat assistant that uses 
OpenAI GPT via LangChain to answer learners' questions in context.

The app processes `*.md` files in *content/* 
"""

import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# ----------------------------- Constants ----------------------------------- #

CONTENT_DIR = "content"  # folder that holds markdown lessons

# ----------------------------- Helper functions ----------------------------- #

def list_md_files(directory: str) -> list[str]:
    """Return a sorted list of all *.md files inside *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".md"))


def pretty_name(filename: str) -> str:
    """Convert a file name like `chapter_1.md` → `Chapter 1`."""
    stem = os.path.splitext(filename)[0]
    return stem.replace("_", " ").title()


def load_markdown(path: str) -> str:
    """Read the file at *path* and return its text (or empty string if missing)."""
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return fp.read()
    except FileNotFoundError:
        return ""

# ----------------------------- Page layout --------------------------------- #
st.set_page_config(page_title="POCUS AI Tutor", layout="wide")
st.title("POCUS AI Tutor")
st.caption("© 2025, Sami Safadi, MD") # Copyright footnote

st.write(
    "Welcome! This interactive tutor pairs textbook-style chapters with an AI assistant. "
    "Paste your **OpenAI API key** below, choose a lesson on the left, and ask a question like:\n "
    "  - Create a MCQ to test my understanding of LV EF\n"
    "  - Explain to me TAPSE"
)

# ----------------------------- API key input ------------------------------- #

api_key = st.text_input(
    "OpenAI API key", value=st.session_state.get("api_key", ""), type="password", key="api"
)
if api_key:
    st.session_state.api_key = api_key

# ----------------------------- Sidebar (lessons) --------------------------- #

st.sidebar.title("📚 Lessons")
md_files = list_md_files(CONTENT_DIR)
if not md_files:
    st.sidebar.warning("No markdown files found in the *content/* directory.")
    st.stop()

chapters = {pretty_name(f): f for f in md_files}
selected_label = st.sidebar.radio("Navigate", list(chapters.keys()), label_visibility="collapsed")
filename = chapters[selected_label]
filepath = os.path.join(CONTENT_DIR, filename)

# Show chosen lesson name on main page
st.subheader(selected_label)

# ----------------------------- Chat state ---------------------------------- #

if st.session_state.get("active_file") != filename:
    st.session_state["messages"] = []
    st.session_state["active_file"] = filename

# ----------------------------- Render chat history ------------------------- #

for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------- Chat input ---------------------------------- #

user_prompt = st.chat_input(f"Ask me anything about: {pretty_name(filename)}")
if user_prompt:
    # Echo user message to UI & state
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Guard clause: API key required
    if "api_key" not in st.session_state or not st.session_state.api_key:
        assistant_response = "⚠️ Please enter a valid OpenAI API key to continue."
    else:
        # Build prompt with lesson context
        lesson_md = load_markdown(filepath)
        system_message = SystemMessage(
            content=(
                "You are an expert ultrasound educator guiding physicians through a point-of-care ultrasound course. "
                "Do not answer any questions that are not related to point of care ultrasound"
                "Answer clearly, concisely, and cite evidence when appropriate.\n\n"
                f"### {selected_label} Content\n{lesson_md}"
            )
        )

        chat_history = [system_message] + [
            HumanMessage(content=m["content"]) for m in st.session_state.messages if m["role"] == "user"
        ]

        llm = ChatOpenAI(
            model_name="gpt-4o-mini",  # adjust as needed
            temperature=0.3,
            openai_api_key=st.session_state.api_key,
            streaming=True
        )
        # ------------------------ Stream back the response ------------------ #

        with st.chat_message("assistant"):
            placeholder = st.empty()
            partial_response = ""
            for chunk in llm.stream(chat_history):
                partial_response += chunk.content or ""
                placeholder.markdown(partial_response + "▌")  # live typing effect
            placeholder.markdown(partial_response)  # final clean‑up

        # Store full assistant response
        st.session_state.messages.append({"role": "assistant", "content": partial_response})