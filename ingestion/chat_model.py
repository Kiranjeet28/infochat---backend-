"""Groq chat model configuration for the notice-answering chain."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_TIMEOUT_SECONDS = 30


def create_chat_model() -> ChatGroq:
    """Create the configured Groq model; set ``GROQ_API_KEY`` in ``.env``."""
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        timeout=GROQ_TIMEOUT_SECONDS,
    )
