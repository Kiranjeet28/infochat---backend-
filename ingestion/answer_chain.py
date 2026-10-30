"""LangChain LCEL chain for answering from current notice evidence."""

from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from ingestion.chat_model import create_chat_model
from ingestion.langchain_retriever import NoticeRetriever
from ingestion.prompt import NOTICE_ANSWER_PROMPT


def format_evidence(documents: Sequence[Document]) -> str:
    """Render retriever documents as numbered, metadata-bearing evidence."""
    if not documents:
        return "(No matching notice evidence was retrieved.)"

    blocks: list[str] = []
    for number, document in enumerate(documents, start=1):
        blocks.append(f"[{number}]\n{document.page_content}")
    return "\n\n".join(blocks)


def _normalize_answer(answer: str) -> str:
    """Normalize common text artifacts before returning an answer to the UI."""
    answer = answer.translate(str.maketrans({
        "\u00a0": " ",  # no-break space
        "\u2007": " ",  # figure space
        "\u202f": " ",  # narrow no-break space
        "\u2011": "-",  # non-breaking hyphen
    }))
    for joined, separated in (
        (r"\bThenotice\b", "The notice"),
        (r"\bThepublished\b", "The published"),
        (r"\bMakeupExam\b", "Makeup Exam"),
    ):
        answer = re.sub(joined, separated, answer, flags=re.IGNORECASE)
    answer = re.sub(r"[ \t]+", " ", answer)
    answer = re.sub(r"\s+([,.;:!?])", r"\1", answer)
    answer = re.sub(r" *\n *", "\n", answer)
    return answer.strip()


def _validate_answer(values: dict[str, Any]) -> dict[str, Any]:
    """Reject answers without citations or with citations outside retrieved evidence."""
    documents: Sequence[Document] = values["documents"]
    answer = _normalize_answer(str(values["answer"]))
    citations = [int(number) for number in re.findall(r"\[(\d+)\]", answer)]
    valid = bool(citations) and all(1 <= number <= len(documents) for number in citations)
    sources = [
        {
            key: document.metadata.get(key)
            for key in ("notice_id", "title", "published_at", "source_url")
        }
        for document in documents
    ]
    return {
        "answer": answer if valid else "not found",
        "sources": sources,
        "status": "answered" if valid else "invalid_citation",
        "citations_valid": valid,
    }


def _no_evidence(values: dict[str, Any]) -> dict[str, Any]:
    return {"answer": "not found", "sources": [], "status": "not_found", "citations_valid": False}


def build_answer_chain(
    retriever: NoticeRetriever | None = None,
    model: Runnable | None = None,
) -> Runnable:
    """Build the question → retrieval → evidence → prompt → model → text chain."""
    selected_retriever = retriever or NoticeRetriever()
    selected_model = model or create_chat_model()

    prepared = (
        {
            "question": itemgetter("question"),
            "documents": itemgetter("question") | selected_retriever,
        }
        | RunnablePassthrough.assign(
            evidence=RunnableLambda(lambda values: format_evidence(values["documents"]))
        )
    )
    answer = (
        RunnablePassthrough.assign(
            answer=NOTICE_ANSWER_PROMPT | selected_model | StrOutputParser()
        )
        | RunnableLambda(_validate_answer)
    )
    return prepared | RunnableBranch(
        (lambda values: not values["documents"], RunnableLambda(_no_evidence)),
        answer,
    )
