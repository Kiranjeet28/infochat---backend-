"""LangChain LCEL chain for answering from current notice evidence."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from operator import itemgetter
import re
import unicodedata
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough

from ingestion.chat_model import (
    AI_ERROR_MESSAGE,
    AI_UNAVAILABLE_MESSAGE,
    generate_answer,
)
from ingestion.answer_config import (
    FRIENDLY_NOT_FOUND_MESSAGE,
    MODEL_INPUT_EXCLUDED_CATEGORIES,
    MODEL_INPUT_EXCLUDED_NOTICE_IDS,
    MODEL_INPUT_EXCLUDED_TITLE_MARKERS,
)
from ingestion.langchain_retriever import NoticeRetriever
from ingestion.prompt import NOTICE_ANSWER_PROMPT


def format_evidence(documents: Sequence[Document]) -> str:
    """Render retriever documents as numbered, metadata-bearing evidence."""
    if not documents:
        return "(No matching notice evidence was retrieved.)"

    blocks: list[str] = []
    for number, document in enumerate(documents, start=1):
        metadata = document.metadata
        title = metadata.get("title") or "Untitled notice"
        notice_date = metadata.get("published_at") or "not provided"
        source_url = metadata.get("source_url") or "not provided"
        if _exclude_full_text(document):
            content = (
                "Summary: This notice concerns students struck off the rolls; "
                "individual names and roll numbers are withheld."
            )
        else:
            content = document.page_content
        blocks.append(
            f"[{number}]\nNotice: {title}\nNotice date: {notice_date}\n"
            f"Source: {source_url}\n{content}"
        )
    return "\n\n".join(blocks)


def _exclude_full_text(document: Document) -> bool:
    """Identify notice types whose contents must not reach the model."""
    metadata = document.metadata
    notice_id = str(metadata.get("notice_id") or "").casefold()
    category = str(metadata.get("category") or "").casefold().replace("-", "_")
    title = str(metadata.get("title") or "").casefold().replace("-", "_")
    excluded_ids = {value.casefold() for value in MODEL_INPUT_EXCLUDED_NOTICE_IDS}
    excluded_categories = {value.casefold().replace("-", "_") for value in MODEL_INPUT_EXCLUDED_CATEGORIES}
    excluded_markers = {value.casefold().replace("-", "_") for value in MODEL_INPUT_EXCLUDED_TITLE_MARKERS}
    return (
        notice_id in excluded_ids
        or category in excluded_categories
        or any(marker in title for marker in excluded_markers)
    )


def normalize_answer(text: str) -> str:
    """Normalize Unicode whitespace and dash artifacts before use or display."""
    normalized: list[str] = []
    invisible = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    forced_spaces = {"\u2009", "\u200a", "\u2060", "\u202f", "\u00a0"}
    for character in text:
        if character in invisible:
            continue
        category = unicodedata.category(character)
        if character in forced_spaces or category in {"Zs", "Zl", "Zp"} or character in "\t\f\v":
            normalized.append(" ")
        elif category == "Pd" or character in {"\u2212", "\u00ad"}:
            normalized.append("-")
        else:
            normalized.append(character)
    text = "".join(normalized).replace("\r\n", "\n").replace("\r", "\n")
    for joined, separated in (
        (r"\bThenotice\b", "The notice"),
        (r"\bThepublished\b", "The published"),
        (r"\bMakeupExam\b", "Makeup Exam"),
    ):
        text = re.sub(joined, separated, text, flags=re.IGNORECASE)
    lines = [re.sub(r" {2,}", " ", line).rstrip(" ") for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_citations(text: str) -> tuple[list[int], bool]:
    """Return citations, expanding ranges such as [1-3], and syntax validity."""
    citations: list[int] = []
    syntax_valid = True
    for group in re.findall(r"\[([^\[\]]*)\]", text):
        parts = [value.strip() for value in group.split(",")]
        if not group.strip() or any(not part for part in parts):
            if re.search(r"\d", group):
                syntax_valid = False
            continue
        for part in parts:
            match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
            if match:
                start = int(match.group(1))
                end = int(match.group(2) or start)
                if end < start:
                    syntax_valid = False
                else:
                    citations.extend(range(start, end + 1))
            elif re.search(r"\d", part) or part.lower() == "n":
                syntax_valid = False
    return citations, syntax_valid


def _validate_answer(values: dict[str, Any]) -> dict[str, Any]:
    """Reject answers without citations or with citations outside retrieved evidence."""
    documents: Sequence[Document] = values["documents"]
    answer = normalize_answer(str(values["answer"]))
    citations, syntax_valid = extract_citations(answer)
    sources = [
        {
            key: document.metadata.get(key)
            for key in ("notice_id", "title", "published_at", "source_url")
        }
        for document in documents
    ]
    if answer == "NOT_FOUND" and not citations:
        return {
            "answer": FRIENDLY_NOT_FOUND_MESSAGE,
            "sources": sources,
            "status": "not_found_model",
            "citations_valid": None,
        }
    valid = bool(citations) and syntax_valid and all(
        1 <= number <= len(documents) for number in citations
    )
    return {
        "answer": answer if valid else "not found",
        "sources": sources,
        "status": "answered" if valid else "invalid_citation",
        "citations_valid": valid,
    }


def _validate_generation(values: dict[str, Any]) -> dict[str, Any]:
    """Validate provider text and attach generation diagnostics to the result."""
    generation: dict[str, Any] = values["generation"]
    documents: Sequence[Document] = values["documents"]
    status = generation.get("status")
    if status == "generated":
        validated = _validate_answer({"documents": documents, "answer": generation.get("text", "")})
    else:
        sources = [
            {
                key: document.metadata.get(key)
                for key in ("notice_id", "title", "published_at", "source_url")
            }
            for document in documents
        ]
        validated = {
            "answer": AI_UNAVAILABLE_MESSAGE if status == "ai_unavailable" else AI_ERROR_MESSAGE,
            "sources": sources,
            "status": status or "error",
            "citations_valid": None,
        }
    return {
        **validated,
        **{
            key: generation.get(key)
            for key in (
                "provider", "model", "latency_seconds", "token_usage", "attempts",
                "retry_wait_seconds", "error_type",
            )
        },
    }


def _no_evidence(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "answer": FRIENDLY_NOT_FOUND_MESSAGE,
        "sources": [],
        "status": "not_found_retrieval",
        "citations_valid": None,
        "provider": None,
        "model": None,
        "latency_seconds": 0.0,
        "token_usage": None,
        "attempts": 0,
        "retry_wait_seconds": 0.0,
        "error_type": None,
    }


def build_answer_chain(
    retriever: NoticeRetriever | None = None,
    *,
    answer_generator: Callable[[Any], dict[str, Any]] | None = None,
    raw_output_callback: Callable[[Any], None] | None = None,
) -> Runnable:
    """Build the question → retrieval → evidence → prompt → model → text chain."""
    selected_retriever = retriever or NoticeRetriever()
    selected_generator = answer_generator or generate_answer

    prepared = (
        {
            "question": itemgetter("question"),
            "current_date": RunnableLambda(lambda _values: date.today().isoformat()),
            "documents": itemgetter("question") | selected_retriever,
        }
        | RunnablePassthrough.assign(
            evidence=RunnableLambda(lambda values: format_evidence(values["documents"]))
        )
    )
    generation = NOTICE_ANSWER_PROMPT | RunnableLambda(selected_generator)
    if raw_output_callback is not None:
        generation = generation | RunnableLambda(
            lambda result: (raw_output_callback(result.get("text")), result)[1]
        )
    answer = (
        RunnablePassthrough.assign(generation=generation)
        | RunnableLambda(_validate_generation)
    )
    return prepared | RunnableBranch(
        (lambda values: not values["documents"], RunnableLambda(_no_evidence)),
        answer,
    )
