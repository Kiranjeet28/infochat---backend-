"""LangChain LCEL chain for answering from current notice evidence."""

from __future__ import annotations

from collections.abc import Sequence
from operator import itemgetter
import re
import os
import unicodedata
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


def normalize_answer(text: str) -> str:
    """Normalize Unicode whitespace and dash artifacts before use or display."""
    normalized: list[str] = []
    invisible = {"\u200b", "\u200c", "\u200d", "\ufeff"}
    for character in text:
        if character in invisible:
            continue
        category = unicodedata.category(character)
        if category in {"Zs", "Zl", "Zp"} or character in "\t\f\v":
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
    """Return numeric citation references and whether their syntax is valid."""
    citations: list[int] = []
    syntax_valid = True
    for group in re.findall(r"\[([^\[\]]*)\]", text):
        if re.fullmatch(r"\s*\d+(?:\s*,\s*\d+)*\s*", group):
            citations.extend(int(value.strip()) for value in group.split(","))
        elif re.search(r"\d", group) or group.strip().lower() == "n":
            syntax_valid = False
    return citations, syntax_valid


def _validate_answer(values: dict[str, Any]) -> dict[str, Any]:
    """Reject answers without citations or with citations outside retrieved evidence."""
    documents: Sequence[Document] = values["documents"]
    answer = normalize_answer(str(values["answer"]))
    citations, syntax_valid = extract_citations(answer)
    valid = bool(citations) and syntax_valid and all(
        1 <= number <= len(documents) for number in citations
    )
    sources = [
        {
            key: document.metadata.get(key)
            for key in ("notice_id", "title", "published_at", "source_url")
        }
        for document in documents
    ]
    is_not_found = bool(
        re.search(r"\bnot found\b|available notices do not establish", answer, re.IGNORECASE)
    )
    return {
        "answer": answer if valid else "not found",
        "sources": sources,
        "status": (
            "invalid_citation" if not valid else "not_found" if is_not_found else "answered"
        ),
        "citations_valid": valid,
    }


def _no_evidence(values: dict[str, Any]) -> dict[str, Any]:
    return {"answer": "not found", "sources": [], "status": "not_found", "citations_valid": False}


def _print_normalized_answer_prefix(answer: str) -> str:
    """Print a normalized, redacted answer prefix for evaluator diagnostics."""
    prefix = answer[:80]
    for secret_name in ("GROQ_API_KEY", "SUPABASE_SERVICE_ROLE_KEY"):
        secret = os.environ.get(secret_name)
        if secret:
            prefix = prefix.replace(secret, "[REDACTED]")
    prefix = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", prefix)
    print(repr(prefix))
    return answer


def build_answer_chain(
    retriever: NoticeRetriever | None = None,
    model: Runnable | None = None,
    *,
    debug_raw_output: bool = False,
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
    parsed_answer = (
        NOTICE_ANSWER_PROMPT
        | selected_model
        | StrOutputParser()
        | RunnableLambda(normalize_answer)
    )
    if debug_raw_output:
        parsed_answer = parsed_answer | RunnableLambda(_print_normalized_answer_prefix)

    answer = (
        RunnablePassthrough.assign(
            answer=parsed_answer
        )
        | RunnableLambda(_validate_answer)
    )
    return prepared | RunnableBranch(
        (lambda values: not values["documents"], RunnableLambda(_no_evidence)),
        answer,
    )
