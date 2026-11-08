"""FastAPI application exposing notice question answering."""

from __future__ import annotations

# Workaround for the Windows Application Control block on uuid_utils; must run
# before any langchain import.
from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()

from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import time
import traceback
import json
from threading import Lock
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4
from pathlib import Path
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ingestion.answer_chain import build_answer_chain
from ingestion.answer_chain import extract_citations, renumber_citations
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
from ingestion.redirects import redirect_response, marker_response


logger = logging.getLogger("cascadeiq.api")
_request_id: ContextVar[str] = ContextVar("cascadeiq_request_id", default="startup")
MAX_QUESTION_LENGTH = 1000
_answer_chain = None
_provider_config: ProviderRuntimeConfig | None = None
_cache_store: SupabaseCacheStore | None = None
_cache_counters = {"hit": 0, "miss": 0}
_AUDIT_LOG_PATH = Path(__file__).resolve().parent / "logs" / "audit.log"
_AUDIT_LOCK = Lock()
_DEFAULT_CORS_ALLOWED_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
)


def _cors_allowed_origins() -> list[str]:
    configured_origins = os.getenv("CORS_ALLOWED_ORIGINS")
    if configured_origins is None:
        return list(_DEFAULT_CORS_ALLOWED_ORIGINS)
    origins = []
    for value in configured_origins.split(","):
        origin = value.strip().rstrip("/")
        if origin and origin != "*":
            origins.append(origin)
    return origins or list(_DEFAULT_CORS_ALLOWED_ORIGINS)


_APP_CORS_ALLOWED_ORIGINS = _cors_allowed_origins()


def _with_cors(application, origins: list[str] | None = None):
    return CORSMiddleware(
        app=application,
        allow_origins=list(_APP_CORS_ALLOWED_ORIGINS if origins is None else origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["content-type", "accept", "x-request-id"],
        allow_credentials=False,
        expose_headers=["X-Request-ID"],
        max_age=600,
    )


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def _configure_logging() -> None:
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "api.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    handler.addFilter(_RequestIdFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s request_id=%(request_id)s %(message)s"))
    logger.setLevel(logging.INFO)
    if not any(isinstance(item, RotatingFileHandler) and item.baseFilename == handler.baseFilename for item in logger.handlers):
        logger.addHandler(handler)


_configure_logging()


def _append_audit_event(event: dict[str, object]) -> None:
    """Write metadata-only audit records without question or answer text."""
    try:
        with _AUDIT_LOCK:
            with _AUDIT_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("status=audit_write_failed error_type=OSError")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _provider_config, _answer_chain
    logger.info("cors_allowed_origins=%s", _APP_CORS_ALLOWED_ORIGINS)
    _provider_config = require_enabled_provider_config()
    initialize_embedding_model()
    if _answer_chain is None:
        _answer_chain = build_answer_chain(provider_config=_provider_config)
    yield


app = FastAPI(title="CascadeIQ API", lifespan=lifespan)
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = uuid4().hex
    request.state.request_id = request_id
    token = _request_id.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        _request_id.reset(token)


def _log_safe_traceback(request: Request, error: Exception) -> None:
    """Log stack frames and exception type while withholding exception text."""
    frames = traceback.extract_tb(error.__traceback__)
    stack = "".join(traceback.format_list(frames))
    logger.error("status=error error_type=%s traceback=Traceback (most recent call last):\n%s%s",
                 type(error).__name__, stack, type(error).__name__)


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception) -> JSONResponse:
    _log_safe_traceback(request, error)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Something went wrong. Please try again.",
                 "request_id": getattr(request.state, "request_id", "unavailable")},
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
    status: Literal["answered", "not_found", "invalid_citation", "ai_unavailable", "redirect"]
    sources_kind: Literal["cited", "related"]
    cached: bool = False
    links: list[dict[str, str]] = Field(default_factory=list)


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
def ask(request: Request, body: AskRequest) -> AskResponse | JSONResponse:
    """Answer a question from current notices and return its source cards."""
    question = body.question.strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "question must not be empty."},
        )

    global _answer_chain, _cache_store
    request_id = getattr(request.state, "request_id", "unavailable")
    started = time.monotonic()
    provider = model = None
    attempts = 0
    status = "error"
    cached = False
    try:
        routed = redirect_response(question)
        if routed:
            status = "redirect"
            logger.info("status=redirect route=%s question_length=%d", routed["route"], len(question))
            _append_audit_event({
                "request_id": request_id, "status": "redirect", "route": routed["route"],
                "question_length": len(question),
            })
            return AskResponse(
                answer=routed["answer"], sources=[], status="redirect", sources_kind="related",
                cached=False, links=routed["links"],
            )

        cache_allowed = cache_enabled() and not privacy_guard(question)
        if not cache_allowed:
            logger.info("status=cache_bypass cached=false question_length=%d", len(question))
        current_key = None
        kb_version = None
        cache_age_seconds = None
        if cache_allowed:
            try:
                if _cache_store is None:
                    import os
                    _cache_store = SupabaseCacheStore(
                        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                        timeout_seconds=3,
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
                    try:
                        _cache_store.set(entry)
                    except Exception as cache_error:
                        logger.warning("status=answered cached=true cache_op=hit_count error_type=%s",
                                       type(cache_error).__name__)
                    _cache_counters["hit"] += 1
                    logger.info("status=answered cached=true provider=%s model=%s attempts=0 latency=%.3f question_length=%d",
                                entry.get("provider"), entry.get("model"), time.monotonic() - started, len(question))
                    return AskResponse(
                        answer=entry["answer"], sources=entry["sources"], status="answered",
                        sources_kind="cited", cached=True,
                    )
                _cache_counters["miss"] += 1
                logger.info("status=cache_miss cached=false question_length=%d", len(question))
            except Exception as cache_error:
                logger.warning("status=cache_miss cached=false cache_op=get_or_invalidate error_type=%s",
                               type(cache_error).__name__)
                cache_allowed = False

        chain = _answer_chain or build_answer_chain(provider_config=_provider_config)
        if body.department or body.semester:
            chain = build_answer_chain(
                retriever=NoticeRetriever(department=body.department, semester=body.semester),
                provider_config=_provider_config,
            )
        invocation = {"question": question}
        if hasattr(chain, "invoke"):
            result = chain.invoke(invocation)
        else:  # Compatibility for async fakes and custom chains in tests.
            import asyncio
            result = asyncio.run(chain.ainvoke(invocation))

        raw_status = result.get("status", "answered")
        model_redirect = marker_response(str(result.get("answer", "")))
        if model_redirect is not None:
            logger.info("status=redirect route=%s question_length=%d", model_redirect["route"], len(question))
            _append_audit_event({
                "request_id": request_id, "status": "redirect", "route": model_redirect["route"],
                "question_length": len(question),
            })
            return AskResponse(
                answer=model_redirect["answer"], sources=[], status="redirect", sources_kind="related",
                cached=False, links=model_redirect["links"],
            )
        if raw_status in {"not_found", "not_found_model", "not_found_retrieval"}:
            status = "not_found"
        elif raw_status in {"answered", "invalid_citation", "ai_unavailable"}:
            status = raw_status
        else:
            raise RuntimeError("answer chain returned an unsupported status")

        answer = result.get("answer", "not found")
        provider, model, attempts = result.get("provider"), result.get("model"), result.get("attempts", 0)
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
            # Sources are trimmed to the cited ones, so the answer's [n] must follow.
            answer = renumber_citations(
                answer, {old: new for new, old in enumerate(cited_indexes, 1)}
            )
            sources_kind = "cited"
        else:
            sources = retrieved_sources
            sources_kind = "related"

        if plain_text_answers_enabled():
            answer = strip_markdown(answer)

        if cache_allowed and status == "answered" and citations_valid and current_key and kb_version:
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
                    logger.warning("status=%s cached=false cache_op=set error_type=%s", status,
                                   type(cache_error).__name__)

        logger.info("status=%s cached=false provider=%s model=%s attempts=%s latency=%.3f question_length=%d",
                    status, provider, model, attempts, time.monotonic() - started, len(question))
        return AskResponse(
            answer=answer,
            sources=sources,
            status=status,
            sources_kind=sources_kind,
            cached=False,
        )
    except Exception as error:
        _log_safe_traceback(request, error)
        logger.error("status=error cached=false provider=%s model=%s attempts=%s latency=%.3f error_type=%s question_length=%d",
                     provider, model, attempts, time.monotonic() - started, type(error).__name__, len(question))
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Something went wrong. Please try again.",
                     "request_id": request_id},
        )


# Wrap the complete FastAPI stack so CORS headers also appear on framework errors.
fastapi_app = app
app = _with_cors(fastapi_app)
