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
import sys
import time
import traceback
import json
import requests
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
from pydantic import BaseModel, ConfigDict, Field

from ingestion.answer_chain import build_answer_chain
from ingestion.answer_chain import extract_citations, renumber_citations
from ingestion.feedback_store import SupabaseFeedbackStore
from ingestion.limits import GlobalModelCap, SlidingWindowLimiter, client_id, env_int
from ingestion.answer_config import plain_text_answers_enabled
from ingestion.answer_config import OFFICIAL_NOTICEBOARD_URL
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
from ingestion.redirects import latest_notices_topic, marker_response, redirect_response


logger = logging.getLogger("cascadeiq.api")
_request_id: ContextVar[str] = ContextVar("cascadeiq_request_id", default="startup")
MAX_QUESTION_LENGTH = 1000
_answer_chain = None
_provider_config: ProviderRuntimeConfig | None = None
_cache_store: SupabaseCacheStore | None = None
_cache_counters = {"hit": 0, "miss": 0}
_data_as_of_cache: dict[str, object] | None = None
_data_as_of_cached_at = 0.0
_DATA_AS_OF_TTL_SECONDS = 300
_DATA_AS_OF_LOCK = Lock()
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


def _ask_windows() -> list[tuple[int, int]]:
    return [(60, env_int("RATE_LIMIT_PER_MINUTE", 10)), (86400, env_int("RATE_LIMIT_PER_DAY", 80))]


# In-memory and per instance: both reset when the process restarts (see ingestion/limits.py).
_ask_limiter = SlidingWindowLimiter(_ask_windows)
_model_cap = GlobalModelCap(lambda: env_int("GLOBAL_DAILY_MODEL_CALLS", 400))

BUSY_TODAY_MESSAGE = (
    "CascadeIQ has reached its limit for AI answers today, so it cannot answer new questions right now. "
    "Please open the notices below or check the official notice board, or try again tomorrow."
)


def _client_id(request: Request) -> str:
    return client_id(request.client.host if request.client else None, request.headers.get("x-forwarded-for"))


def _busy_generator(_prompt) -> dict:
    """Stand-in for the model when the daily cap is reached: retrieval runs, no provider is called."""
    return {"status": "ai_unavailable", "attempts": 0, "provider": None, "model": None}


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


class _OneLineFormatter(logging.Formatter):
    """Keep every log record on one line (stack traces are joined with ' | ')."""

    def format(self, record: logging.LogRecord) -> str:
        return super().format(record).replace("\r", " ").replace("\n", " | ")


def _configure_logging(target: logging.Logger | None = None) -> None:
    """Log to stdout; also to LOG_FILE only when that variable is set."""
    target = target or logger
    formatter = _OneLineFormatter("%(asctime)s %(levelname)s request_id=%(request_id)s %(message)s")

    def attach(handler: logging.Handler, marker: str) -> None:
        if any(getattr(item, "_cascadeiq_marker", None) == marker for item in target.handlers):
            return
        handler.addFilter(_RequestIdFilter())
        handler.setFormatter(formatter)
        handler._cascadeiq_marker = marker  # type: ignore[attr-defined]
        target.addHandler(handler)

    target.setLevel(logging.INFO)
    target.propagate = False
    attach(logging.StreamHandler(sys.stdout), "stdout")
    log_file = os.getenv("LOG_FILE")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        attach(RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"), f"file:{path}")


_configure_logging()


def _append_audit_event(event: dict[str, object]) -> None:
    """Write metadata-only audit records without question or answer text."""
    try:
        with _AUDIT_LOCK:
            _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
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
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
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
    request: Request, error: RequestValidationError
) -> JSONResponse:
    if request.url.path == "/api/feedback":
        message = "Invalid feedback."
    else:
        message = f"question must contain between 1 and {MAX_QUESTION_LENGTH} characters."
    return JSONResponse(status_code=422, content={"status": "error", "message": message})


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


class LatestNotice(BaseModel):
    title: str
    date: str | None = None
    link: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCard]
    status: Literal[
        "answered", "not_found", "invalid_citation", "ai_unavailable", "redirect", "latest_notices"
    ]
    sources_kind: Literal["cited", "related"]
    cached: bool = False
    links: list[dict[str, str]] = Field(default_factory=list)
    data_as_of: dict[str, str] | None = None
    notices: list[LatestNotice] | None = None


def _supabase_read_headers() -> dict[str, str]:
    key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
    return {"apikey": key, "Authorization": f"Bearer {key}"}


def _notice_select(params: dict[str, str]) -> list[dict]:
    response = requests.get(
        f"{os.environ['SUPABASE_URL'].rstrip('/')}/rest/v1/notices",
        headers=_supabase_read_headers(),
        params=params,
        timeout=3,
    )
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError("notices response was not a list")
    return rows


def _data_as_of() -> dict[str, str] | None:
    """Read freshness metadata with a short in-memory TTL; failures are fail-open."""
    global _data_as_of_cache, _data_as_of_cached_at
    now = time.monotonic()
    with _DATA_AS_OF_LOCK:
        if _data_as_of_cache is not None and now - _data_as_of_cached_at < _DATA_AS_OF_TTL_SECONDS:
            return dict(_data_as_of_cache)
    try:
        published = _notice_select({
            "select": "published_at", "review_status": "eq.ok", "superseded_by": "is.null",
            "order": "published_at.desc.nullslast", "limit": "1",
        })
        fetched = _notice_select({
            "select": "fetched_at", "order": "fetched_at.desc.nullslast", "limit": "1",
        })
        result: dict[str, str] = {}
        if published and published[0].get("published_at"):
            result["latest_notice_date"] = str(published[0]["published_at"])
        if fetched and fetched[0].get("fetched_at"):
            result["last_ingested_at"] = str(fetched[0]["fetched_at"])
        if not result:
            return None
    except (KeyError, ValueError, TypeError, requests.RequestException) as error:
        logger.warning("status=data_as_of_unavailable error_type=%s", type(error).__name__)
        return None
    with _DATA_AS_OF_LOCK:
        _data_as_of_cache = result
        _data_as_of_cached_at = time.monotonic()
    return dict(result)


def _latest_notice_rows(topic: str | None) -> list[dict]:
    params = {
        "select": "title,published_at,source_url",
        "review_status": "eq.ok",
        "superseded_by": "is.null",
        "order": "published_at.desc.nullslast",
        "limit": "5",
    }
    if topic:
        params["title"] = f"ilike.*{topic}*"
    rows = _notice_select(params)
    if topic and not rows:
        params.pop("title")
        rows = _notice_select(params)
    return [
        {"title": str(row.get("title") or "Untitled notice"),
         "date": str(row["published_at"]) if row.get("published_at") else None,
         "link": str(row["source_url"])}
        for row in rows
        if isinstance(row.get("source_url"), str) and row["source_url"].startswith("https://")
    ][:5]


def _noticeboard_guidance(answer: str, data_as_of: dict[str, str] | None) -> str:
    latest_date = data_as_of.get("latest_notice_date") if data_as_of else None
    if not latest_date:
        return answer
    noticeboard_url = os.getenv("OFFICIAL_NOTICEBOARD_URL", OFFICIAL_NOTICEBOARD_URL)
    return (
        f"{answer.rstrip()} My latest notice is dated {latest_date}. "
        f"Please check the official noticeboard: {noticeboard_url}"
    )


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


_feedback_limiter = SlidingWindowLimiter(lambda: [(86400, env_int("FEEDBACK_PER_DAY", 20))])
_feedback_store: SupabaseFeedbackStore | None = None
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^[0-9a-f]{1,64}$")
    rating: Literal["up", "down"]
    reason: Literal["wrong_answer", "outdated", "not_relevant", "other"] | None = None
    comment: str | None = Field(default=None, max_length=200)


@app.post("/api/feedback")
def feedback(request: Request, body: FeedbackRequest) -> JSONResponse:
    """Store a thumbs up/down for an earlier answer; the question and answer are never stored or logged."""
    global _feedback_store
    retry_after = _feedback_limiter.hit(_client_id(request))
    if retry_after is not None:
        return JSONResponse(
            status_code=429, headers={"Retry-After": str(retry_after)},
            content={"status": "rate_limited", "retry_after_seconds": retry_after,
                     "message": "Thanks, you have sent enough feedback for today."},
        )
    comment = _CONTROL_CHARS.sub(" ", body.comment).strip() if body.comment else None
    row = {"request_id": body.request_id, "rating": body.rating, "reason": body.reason, "comment": comment or None}
    try:
        if _feedback_store is None:
            _feedback_store = SupabaseFeedbackStore(
                os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"]
            )
        _feedback_store.add(row)
    except Exception as error:
        logger.warning("status=feedback_error error_type=%s", type(error).__name__)
        return JSONResponse(status_code=503, content={"status": "error", "message": "Feedback is not available right now."})
    logger.info("status=feedback rating=%s reason=%s", body.rating, body.reason)
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/api/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(request: Request, body: AskRequest) -> AskResponse | JSONResponse:
    """Answer a question from current notices and return its source cards."""
    retry_after = _ask_limiter.hit(_client_id(request))
    if retry_after is not None:
        logger.info("status=rate_limited retry_after=%d", retry_after)
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after)},
            content={"status": "rate_limited", "retry_after_seconds": retry_after,
                     "message": "Too many questions in a short time. Please wait a moment and try again."},
        )
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
        freshness = _data_as_of()
        latest_topic = latest_notices_topic(question)
        if latest_topic is not None:
            notices = _latest_notice_rows(latest_topic or None)
            topic_text = f" matching {latest_topic}" if latest_topic else ""
            answer = (
                f"Here are the newest notices{topic_text}."
                if notices else
                f"There are no searchable notices{topic_text}; please check the official noticeboard."
            )
            return AskResponse(
                answer=answer,
                sources=[],
                status="latest_notices",
                sources_kind="related",
                cached=False,
                data_as_of=freshness,
                notices=notices,
            )
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
                cached=False, links=routed["links"], data_as_of=freshness,
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
                        sources_kind="cited", cached=True, data_as_of=freshness,
                    )
                _cache_counters["miss"] += 1
                logger.info("status=cache_miss cached=false question_length=%d", len(question))
            except Exception as cache_error:
                logger.warning("status=cache_miss cached=false cache_op=get_or_invalidate error_type=%s",
                               type(cache_error).__name__)
                cache_allowed = False

        model_slot = _model_cap.reserve()
        busy = not model_slot
        if busy:  # daily cap reached: retrieve the related notices but call no provider
            chain = build_answer_chain(
                retriever=NoticeRetriever(department=body.department, semester=body.semester),
                answer_generator=_busy_generator,
            )
        else:
            chain = _answer_chain or build_answer_chain(provider_config=_provider_config)
            if body.department or body.semester:
                chain = build_answer_chain(
                    retriever=NoticeRetriever(department=body.department, semester=body.semester),
                    provider_config=_provider_config,
                )
        invocation = {"question": question}
        try:
            if hasattr(chain, "invoke"):
                result = chain.invoke(invocation)
            else:  # Compatibility for async fakes and custom chains in tests.
                import asyncio
                result = asyncio.run(chain.ainvoke(invocation))
        except Exception:
            if model_slot:
                _model_cap.adjust(-1)  # nothing reached a provider
            raise

        raw_status = result.get("status", "answered")
        if model_slot:  # only real provider calls count towards the daily cap
            calls = int(result.get("attempts") or 0)
            if raw_status == "not_found_retrieval" or calls == 0:
                _model_cap.adjust(-1)
            elif calls > 1:
                _model_cap.adjust(calls - 1)
        if busy and raw_status == "ai_unavailable":
            result = {**result, "answer": BUSY_TODAY_MESSAGE}
            logger.info("status=ai_unavailable reason=daily_cap")
        model_redirect = marker_response(str(result.get("answer", "")))
        if model_redirect is not None:
            logger.info("status=redirect route=%s question_length=%d", model_redirect["route"], len(question))
            _append_audit_event({
                "request_id": request_id, "status": "redirect", "route": model_redirect["route"],
                "question_length": len(question),
            })
            return AskResponse(
                answer=model_redirect["answer"], sources=[], status="redirect", sources_kind="related",
                cached=False, links=model_redirect["links"], data_as_of=freshness,
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
        if status in {"not_found", "ai_unavailable"}:
            answer = _noticeboard_guidance(answer, freshness)

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
            data_as_of=freshness,
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
