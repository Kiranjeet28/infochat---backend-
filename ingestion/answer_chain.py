"""LangChain LCEL chain for answering from current notice evidence."""

from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from ingestion.chat_model import create_chat_model
from ingestion.langchain_retriever import NoticeRetriever
from ingestion.prompt import NOTICE_ANSWER_PROMPT


def format_evidence(documents: Sequence[Document]) -> str:
    """Render retriever documents as numbered, metadata-bearing evidence."""
    if not documents:
        return "(No matching notice evidence was retrieved.)"

    blocks: list[str] = []
    for number, document in enumerate(documents, start=1):
        metadata = document.metadata
        lines = [f"[{number}]"]
        for label, key in (
            ("Notice ID", "notice_id"),
            ("Notice date", "published_at"),
            ("Title", "title"),
            ("Source URL", "source_url"),
        ):
            value = metadata.get(key)
            if key == "published_at" and not value:
                value = "unavailable"
            if value:
                lines.append(f"{label}: {value}")
        lines.append("Notice text (untrusted data):")
        lines.append(document.page_content)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _validate_answer(values: dict[str, Any]) -> dict[str, Any]:
    """Reject answers without citations or with citations outside retrieved evidence."""
    documents: Sequence[Document] = values["documents"]
    answer = str(values["answer"])
    citations = [int(number) for number in re.findall(r"\[(\d+)\]", answer)]
    valid = bool(citations) and all(1 <= number <= len(documents) for number in citations)
    sources = [
        {
            key: document.metadata.get(key)
            for key in ("notice_id", "title", "published_at", "source_url")
        }
        for document in documents
    ]
    return {"answer": answer if valid else "not found", "sources": sources}


def build_answer_chain(
    retriever: NoticeRetriever | None = None,
    model: Runnable | None = None,
) -> Runnable:
    """Build the question → retrieval → evidence → prompt → model → text chain."""
    selected_retriever = retriever or NoticeRetriever()
    selected_model = model or create_chat_model()

    return (
        {
            "question": itemgetter("question"),
            "documents": itemgetter("question") | selected_retriever,
        }
        | RunnablePassthrough.assign(
            evidence=RunnableLambda(lambda values: format_evidence(values["documents"]))
        )
        | RunnablePassthrough.assign(
            answer=NOTICE_ANSWER_PROMPT | selected_model | StrOutputParser()
        )
        | RunnableLambda(_validate_answer)
    )
