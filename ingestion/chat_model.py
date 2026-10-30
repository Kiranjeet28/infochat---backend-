"""Provider-neutral answer generation with bounded fallback and audit records."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from dotenv import load_dotenv

from ingestion.provider_config import ProviderRuntimeConfig, get_provider_runtime_config


load_dotenv(Path(__file__).resolve().parent.parent / ".env")
AI_UNAVAILABLE_MESSAGE = "The AI service is temporarily unavailable. Please try again later."
AI_ERROR_MESSAGE = "The AI service could not process this request."
AUDIT_PATH = Path(__file__).resolve().parent.parent / "results" / "provider_audit.jsonl"
RETRYABLE_HTTP_STATUSES = {429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_STATUSES = {400, 401, 403, 404}


class ProviderConfigurationError(RuntimeError):
    """A local provider setting is invalid; do not retry or switch providers."""


def _groq_adapter(prompt: Any, *, model: str, timeout_seconds: float) -> Any:
    from langchain_groq import ChatGroq

    chat = ChatGroq(
        model=model,
        temperature=0,
        timeout=timeout_seconds,
        max_retries=0,
    )
    return chat.invoke(prompt)


def _gemini_adapter(prompt: Any, *, model: str, timeout_seconds: float) -> Any:
    # Before enabling, verify the current model name, free quota, and data-use terms in Google's official docs.
    if not os.environ.get("GEMINI_API_KEY"):
        raise ProviderConfigurationError("GEMINI_API_KEY is not configured")
    from langchain_google_genai import ChatGoogleGenerativeAI

    chat = ChatGoogleGenerativeAI(
        model=model,
        temperature=0,
        timeout=timeout_seconds,
        max_retries=0,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )
    return chat.invoke(prompt)


DEFAULT_ADAPTERS: dict[str, Callable[..., Any]] = {
    "groq": _groq_adapter,
    "gemini": _gemini_adapter,
}


def _http_status(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status


def is_retryable_provider_error(error: Exception) -> bool:
    """Classify only explicitly transient provider failures as retryable."""
    status = _http_status(error)
    if status in NON_RETRYABLE_HTTP_STATUSES:
        return False
    if status in RETRYABLE_HTTP_STATUSES:
        return True
    if isinstance(error, ProviderConfigurationError):
        return False
    names = {cls.__name__.casefold() for cls in type(error).__mro__}
    return any(
        "timeout" in name or "timedout" in name or "connection" in name or "connecterror" in name
        for name in names
    )


def _response_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in content
        )
    raise TypeError("Provider returned no text content")


def _token_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        metadata = getattr(response, "response_metadata", None)
        usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    return dict(usage) if isinstance(usage, dict) else None


def _append_audit_event(event: dict[str, Any]) -> None:
    """Append metadata-only audit data; never write prompts, notice text, or credentials."""
    try:
        AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with AUDIT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError:
        return


def _result(
    *, text: str | None, provider: str | None, model: str | None,
    latency_seconds: float, token_usage: dict[str, Any] | None,
    attempts: int, status: str, error_type: str | None,
    retry_wait_seconds: float,
) -> dict[str, Any]:
    return {
        "text": text,
        "provider": provider,
        "model": model,
        "latency_seconds": latency_seconds,
        "token_usage": token_usage,
        "attempts": attempts,
        "status": status,
        "error_type": error_type,
        "retry_wait_seconds": retry_wait_seconds,
    }


def generate_answer(
    prompt: Any,
    *,
    config: ProviderRuntimeConfig | None = None,
    adapters: Mapping[str, Callable[..., Any]] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    audit_sink: Callable[[dict[str, Any]], None] = _append_audit_event,
) -> dict[str, Any]:
    """Generate one answer through the configured, bounded provider sequence."""
    settings = config or get_provider_runtime_config()
    selected_adapters = adapters if adapters is not None else DEFAULT_ADAPTERS
    providers = [
        name for name in settings.provider_order
        if name in settings.providers
        and settings.providers[name].enabled
        and name in selected_adapters
    ]
    started = clock_fn()
    retry_wait = 0.0
    last_error: Exception | None = None
    last_provider: str | None = None
    last_model: str | None = None
    attempts = 0

    if not providers:
        audit_sink({
            "provider": None,
            "model": None,
            "latency_seconds": 0.0,
            "status": "error",
            "attempts": 0,
            "retry_wait_seconds": 0.0,
            "error_type": "NoProviderEnabledError",
            "token_usage": None,
        })
        return _result(
            text=None, provider=None, model=None, latency_seconds=0.0,
            token_usage=None, attempts=0, status="error",
            error_type="NoProviderEnabledError", retry_wait_seconds=0.0,
        )

    attempt_limit = min(settings.max_attempts, len(providers))
    for provider in providers[:attempt_limit]:
        remaining = settings.max_total_time_seconds - (clock_fn() - started)
        if remaining <= 0:
            break
        provider_settings = settings.providers[provider]
        timeout_seconds = min(provider_settings.timeout_seconds, remaining)
        attempts += 1
        last_provider = provider
        last_model = provider_settings.model
        attempt_started = clock_fn()
        try:
            response = selected_adapters[provider](
                prompt,
                model=provider_settings.model,
                timeout_seconds=timeout_seconds,
            )
            text = _response_text(response)
            usage = _token_usage(response)
            elapsed_total = clock_fn() - started
            if elapsed_total > settings.max_total_time_seconds:
                raise TimeoutError("provider exceeded the configured total answer time")
            audit_sink({
                "provider": provider,
                "model": provider_settings.model,
                "latency_seconds": max(0.0, clock_fn() - attempt_started),
                "status": "generated",
                "attempts": attempts,
                "retry_wait_seconds": retry_wait,
                "error_type": None,
                "token_usage": usage,
            })
            return _result(
                text=text, provider=provider, model=provider_settings.model,
                latency_seconds=max(0.0, elapsed_total), token_usage=usage,
                attempts=attempts, status="generated", error_type=None,
                retry_wait_seconds=retry_wait,
            )
        except Exception as error:
            last_error = error
            retryable = is_retryable_provider_error(error)
            remaining = settings.max_total_time_seconds - (clock_fn() - started)
            can_switch = retryable and attempts < attempt_limit and remaining > 0
            wait = min(settings.retry_wait_seconds, remaining) if can_switch else 0.0
            audit_sink({
                "provider": provider,
                "model": provider_settings.model,
                "latency_seconds": max(0.0, clock_fn() - attempt_started),
                "status": (
                    "retryable_error" if can_switch
                    else "ai_unavailable" if retryable
                    else "error"
                ),
                "attempts": attempts,
                "retry_wait_seconds": wait,
                "error_type": type(error).__name__,
                "token_usage": None,
            })
            if not retryable:
                return _result(
                    text=None, provider=provider, model=provider_settings.model,
                    latency_seconds=max(0.0, clock_fn() - started), token_usage=None,
                    attempts=attempts, status="error", error_type=type(error).__name__,
                    retry_wait_seconds=retry_wait,
                )
            if can_switch:
                if wait:
                    sleep_fn(wait)
                    retry_wait += wait
                continue
            break

    status = "ai_unavailable" if last_error is None or is_retryable_provider_error(last_error) else "error"
    error_type = type(last_error).__name__ if last_error is not None else "TotalTimeLimitExceeded"
    return _result(
        text=None, provider=last_provider, model=last_model,
        latency_seconds=max(0.0, clock_fn() - started), token_usage=None,
        attempts=attempts, status=status, error_type=error_type,
        retry_wait_seconds=retry_wait,
    )
