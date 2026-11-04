"""FastAPI application exposing notice question answering."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import re
import traceback
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingestion.answer_chain import build_answer_chain
from ingestion.answer_chain import extract_citations
from ingestion.answer_config import plain_text_answers_enabled
from ingestion.cache_config import cache_enabled, cache_ttl_seconds
from ingestion.cache_store import (
    SupabaseCacheStore, cache_key, cache_answer_if_valid, make_cache_entry,
    privacy_guard, question_hash, read_kb_version, verify_cached_sources,
    validated_cache_hit,
)
from ingestion.langchain_retriever import NoticeRetriever
from ingestion.prompt import PROMPT_VERSION
from ingestion.provider_config import ProviderRuntimeConfig, require_enabled_provider_config
from ingestion.retrieve import initialize_embedding_model


logger = logging.getLogger("cascadeiq.api")
MAX_QUESTION_LENGTH = 1000
_answer_chain = None
_provider_config: ProviderRuntimeConfig | None = None
_cache_store: SupabaseCacheStore | None = None
_cache_counters = {"hit": 0, "miss": 0}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _provider_config
    _provider_config = require_enabled_provider_config()
    initialize_embedding_model()
    yield


app = FastAPI(title="CascadeIQ API", lifespan=lifespan)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


def _log_safe_traceback(request: Request, error: Exception) -> None:
    """Log stack frames and exception type while withholding exception text."""
    frames = traceback.extract_tb(error.__traceback__)
    stack = "".join(traceback.format_list(frames))
    request_id = getattr(request.state, "request_id", "unavailable")
    logger.error(
        "request_id=%s unexpected API error (%s):\n%s%s: [details redacted]",
        request_id,
        request.url.path,
        stack,
        type(error).__name__,
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    _log_safe_traceback(request, error)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Something went wrong. Please try again."},
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(
    _request: Request, error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "message": f"question must contain between 1 and {MAX_QUESTION_LENGTH} characters.",
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Lightweight process health check; provider and local model are checked at startup."""
    return {"status": "ok"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)
    department: str | None = None
    semester: str | None = None


class SourceCard(BaseModel):
    notice_id: str | None = None
    title: str | None = None
    published_at: str | None = None
    source_url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCard]
    status: Literal["answered", "not_found", "invalid_citation", "ai_unavailable"]
    sources_kind: Literal["cited", "related"]
    cached: bool = False


def strip_markdown(text: str) -> str:
    """Remove common Markdown formatting while retaining link text and URLs."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"(?<!\w)(\*\*|__|\*|_)(?=\S)(.*?\S)\1(?!\w)", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"~~(.*?)~~", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"`{1,3}", "", text)
    text = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+", "", text)
    text = re.sub(r"(?m)^[ \t]{0,3}>[ \t]?", "", text)
    text = re.sub(r"(?m)^[ \t]*[-+*][ \t]+", "", text)
    text = re.sub(r"(?m)^[ \t]*\d+[.)][ \t]+", "", text)
    return text


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse | JSONResponse:
    """Answer a question from current notices and return its source cards."""
    question = body.question.strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "question must not be empty."},
        )

    global _answer_chain, _cache_store
    try:
        cache_allowed = cache_enabled() and not privacy_guard(question)
        question_digest = question_hash(question)
        if not cache_allowed:
            logger.info("cache bypass question_hash=%s cached=false cache_age_seconds=null hits=%s misses=%s",
                        question_digest, _cache_counters["hit"], _cache_counters["miss"])
        current_key = None
        kb_version = None
        cache_age_seconds = None
        if cache_allowed:
            try:
                if _cache_store is None:
                    import os
                    _cache_store = SupabaseCacheStore(
                        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
                    )
                kb_version = read_kb_version(_cache_store.project_url, _cache_store.service_role_key)
                current_key = cache_key(
                    question, body.department, body.semester, kb_version, PROMPT_VERSION
                )
                now = datetime.now(timezone.utc)
                entry, cache_age_seconds = validated_cache_hit(
                    _cache_store, current_key, now=now, kb_version=kb_version,
                    prompt_version=PROMPT_VERSION,
                    source_verifier=lambda candidate: verify_cached_sources(
                        _cache_store.project_url, _cache_store.service_role_key, candidate
                    ),
                )
                if entry:
                    entry["hit_count"] = int(entry.get("hit_count", 0)) + 1
                    _cache_store.set(entry)
                    _cache_counters["hit"] += 1
                    logger.info("cache hit question_hash=%s cached=true cache_age_seconds=%s hits=%s misses=%s",
                                question_digest, cache_age_seconds, _cache_counters["hit"], _cache_counters["miss"])
                    return AskResponse(
                        answer=entry["answer"], sources=entry["sources"], status="answered",
                        sources_kind="cited", cached=True,
                    )
                _cache_counters["miss"] += 1
                logger.info("cache miss question_hash=%s cached=false cache_age_seconds=null hits=%s misses=%s",
                            question_digest, _cache_counters["hit"], _cache_counters["miss"])
            except Exception as cache_error:
                logger.warning("cache unavailable (%s); proceeding without cache", type(cache_error).__name__)
                cache_allowed = False

        chain = _answer_chain
        if body.department or body.semester:
            chain = build_answer_chain(
                retriever=NoticeRetriever(department=body.department, semester=body.semester),
                provider_config=_provider_config,
            )
        elif chain is None:
            chain = build_answer_chain(provider_config=_provider_config)
            _answer_chain = chain
        result = await chain.ainvoke({"question": question})

        raw_status = result.get("status", "answered")
        if raw_status in {"not_found", "not_found_model", "not_found_retrieval"}:
            status = "not_found"
        elif raw_status in {"answered", "invalid_citation", "ai_unavailable"}:
            status = raw_status
        else:
            raise RuntimeError("answer chain returned an unsupported status")

        answer = result.get("answer", "not found")
        retrieved_sources = result.get("sources", [])
        citations, syntax_valid = extract_citations(answer)
        citations_valid = (
            status == "answered"
            and result.get("citations_valid") is True
            and syntax_valid
            and bool(citations)
            and all(1 <= number <= len(retrieved_sources) for number in citations)
        )
        if citations_valid:
            cited_indexes = list(dict.fromkeys(citations))
            sources = [retrieved_sources[number - 1] for number in cited_indexes]
            sources_kind = "cited"
        else:
            sources = retrieved_sources
            sources_kind = "related"

        if plain_text_answers_enabled():
            answer = strip_markdown(answer)

        if cache_allowed and status == "answered" and current_key and kb_version:
            source_ids = list(dict.fromkeys(
                str(source.get("notice_id")) for source in sources if source.get("notice_id")
            ))
            source_hashes = {
                str(source["notice_id"]): source["content_hash"]
                for source in sources
                if source.get("notice_id") and source.get("content_hash")
            }
            if source_ids and len(source_hashes) == len(source_ids):
                now = datetime.now(timezone.utc)
                entry = make_cache_entry(
                    cache_key_value=current_key,
                    question=question,
                    answer=answer,
                    sources=sources,
                    source_content_hashes=source_hashes,
                    kb_version=kb_version,
                    prompt_version=PROMPT_VERSION,
                    provider=result.get("provider"),
                    model=result.get("model"),
                    now=now,
                    ttl_seconds=cache_ttl_seconds(),
                )
                try:
                    cache_answer_if_valid(_cache_store, result, citations_valid, entry)
                except Exception as cache_error:
                    logger.warning("cache write unavailable (%s)", type(cache_error).__name__)

        return AskResponse(
            answer=answer,
            sources=sources,
            status=status,
            sources_kind=sources_kind,
            cached=False,
        )
    except Exception as error:
        _log_safe_traceback(request, error)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Something went wrong. Please try again."},
        )
