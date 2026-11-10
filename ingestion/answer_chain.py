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
    FRIENDLY_INVALID_CITATION_MESSAGE,
    MODEL_INPUT_EXCLUDED_CATEGORIES,
    MODEL_INPUT_EXCLUDED_NOTICE_IDS,
    MODEL_INPUT_EXCLUDED_TITLE_MARKERS,
    normalize_marker,
    restricted_notice_patterns,
)
from ingestion.langchain_retriever import NoticeRetriever
from ingestion.prompt import NOTICE_ANSWER_PROMPT
from ingestion.provider_config import ProviderRuntimeConfig
from ingestion.redirects import REDIRECT_CALCULATOR, REDIRECT_TIMETABLE


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
            content = _withheld_summary(document)
        else:
            content = document.page_content
        blocks.append(
            f"[{number}]\nNotice: {title}\nNotice date: {notice_date}\n"
            f"Source: {source_url}\n{content}"
        )
    return "\n\n".join(blocks)


def _withheld_summary(document: Document) -> str:
    title = normalize_marker(str(document.metadata.get("title") or ""))
    if "struck off" in title or "struck off" in normalize_marker(str(document.metadata.get("category") or "")):
        return (
            "Summary: This notice concerns students struck off the rolls; "
            "individual names and roll numbers are withheld."
        )
    return (
        "Summary: This notice concerns individual students (a student list or personal decisions); "
        "names, roll numbers and other personal details are withheld. Refer the user to the official notice at the Source link."
    )


def _exclude_full_text(document: Document) -> bool:
    """Identify notice types whose contents must not reach the model."""
    metadata = document.metadata
    notice_id = str(metadata.get("notice_id") or "").casefold()
    category = str(metadata.get("category") or "").casefold().replace("-", "_")
    title = str(metadata.get("title") or "").casefold().replace("-", "_")
    excluded_ids = {value.casefold() for value in MODEL_INPUT_EXCLUDED_NOTICE_IDS}
    excluded_categories = {value.casefold().replace("-", "_") for value in MODEL_INPUT_EXCLUDED_CATEGORIES}
    excluded_markers = {value.casefold().replace("-", "_") for value in MODEL_INPUT_EXCLUDED_TITLE_MARKERS}
    restricted = restricted_notice_patterns()
    searchable = (normalize_marker(str(metadata.get("title") or "")), normalize_marker(str(metadata.get("category") or "")))
    return (
        notice_id in excluded_ids
        or category in excluded_categories
        or any(marker in title for marker in excluded_markers)
        or any(pattern in text for pattern in restricted for text in searchable)
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


_CITATION_TOKEN_PATTERN = re.compile(
    r"\[(?:[0-9]+(?:\s*,\s*[0-9]+)?|[0-9]+\s*-\s*[0-9]+)\]"
)


def citation_tokens(text: str) -> list[str]:
    """Return only bracket tokens in the supported numeric citation forms."""
    return [match.group(0) for match in _CITATION_TOKEN_PATTERN.finditer(text)]


def extract_citations(text: str) -> tuple[list[int], bool]:
    """Return citation indexes; unrelated bracketed text is ignored."""
    citations: list[int] = []
    syntax_valid = True
    for token in citation_tokens(text):
        group = token[1:-1]
        if "-" in group:
            start_text, end_text = (part.strip() for part in group.split("-", 1))
            start, end = int(start_text), int(end_text)
            if end < start:
                syntax_valid = False
            else:
                citations.extend(range(start, end + 1))
        else:
            citations.extend(int(part.strip()) for part in group.split(","))
    return citations, syntax_valid


def renumber_citations(text: str, mapping: dict[int, int]) -> str:
    """Rewrite citation tokens so old source numbers become their new positions."""

    def replace(match: re.Match[str]) -> str:
        group = match.group(0)[1:-1]
        is_range = "-" in group
        if is_range:
            start, end = (int(part) for part in group.split("-", 1))
            old = list(range(start, end + 1))
        else:
            old = [int(part) for part in group.split(",")]
        new = [mapping.get(number, number) for number in old]
        if is_range and len(new) > 1 and new == list(range(new[0], new[0] + len(new))):
            return f"[{new[0]}-{new[-1]}]"
        if not is_range and len(new) <= 2:
            return "[" + ", ".join(str(number) for number in new) + "]"
        return "".join(f"[{number}]" for number in new)

    return _CITATION_TOKEN_PATTERN.sub(replace, text)


def _validate_answer(values: dict[str, Any]) -> dict[str, Any]:
    """Reject answers without citations or with citations outside retrieved evidence."""
    documents: Sequence[Document] = values["documents"]
    answer = normalize_answer(str(values["answer"]))
    citations, syntax_valid = extract_citations(answer)
    sources = [
        {
            key: document.metadata.get(key)
            for key in ("notice_id", "title", "published_at", "source_url", "content_hash")
        }
        for document in documents
    ]
    if answer in {REDIRECT_CALCULATOR, REDIRECT_TIMETABLE}:
        return {
            "answer": answer,
            "sources": [],
            "status": "redirect",
            "citations_valid": True,
        }
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
        "answer": answer if valid else FRIENDLY_INVALID_CITATION_MESSAGE,
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
        if validated["status"] == "redirect":
            return {
                **validated,
                **{key: generation.get(key) for key in (
                    "provider", "model", "latency_seconds", "token_usage", "attempts",
                    "retry_wait_seconds", "error_type",
                )},
            }
    else:
        sources = [
            {
                key: document.metadata.get(key)
                for key in ("notice_id", "title", "published_at", "source_url", "content_hash")
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
    provider_config: ProviderRuntimeConfig | None = None,
    raw_output_callback: Callable[[Any], None] | None = None,
) -> Runnable:
    """Build the question → retrieval → evidence → prompt → model → text chain."""
    selected_retriever = retriever or NoticeRetriever()
    selected_generator = answer_generator or (
        lambda prompt: generate_answer(prompt, provider_config=provider_config)
    )

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
