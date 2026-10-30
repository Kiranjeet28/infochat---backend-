"""Environment-backed provider settings; credentials remain in .env."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(ENV_FILE)

# Override through .env: LLM_PROVIDER_ORDER, LLM_MAX_ATTEMPTS,
# LLM_MAX_TOTAL_TIME_SECONDS, LLM_RETRY_WAIT_SECONDS, and each provider's
# *_ENABLED, *_MODEL, and *_TIMEOUT_SECONDS settings.
DEFAULT_PROVIDER_ORDER = ("groq", "gemini")
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_TOTAL_TIME_SECONDS = 45.0
DEFAULT_RETRY_WAIT_SECONDS = 1.0


@dataclass(frozen=True)
class ProviderSettings:
    enabled: bool
    model: str
    timeout_seconds: float


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    provider_order: tuple[str, ...]
    providers: dict[str, ProviderSettings]
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    max_total_time_seconds: float = DEFAULT_MAX_TOTAL_TIME_SECONDS
    retry_wait_seconds: float = DEFAULT_RETRY_WAIT_SECONDS


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _positive_float(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def get_provider_runtime_config() -> ProviderRuntimeConfig:
    order_text = os.environ.get("LLM_PROVIDER_ORDER", "").strip()
    configured_order = tuple(dict.fromkeys(
        name.strip().casefold() for name in order_text.split(",") if name.strip()
    ))
    provider_order = configured_order or DEFAULT_PROVIDER_ORDER
    providers = {
        "groq": ProviderSettings(
            enabled=_env_bool("GROQ_ENABLED", True),
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            timeout_seconds=_positive_float("GROQ_TIMEOUT_SECONDS", 30.0),
        ),
        "gemini": ProviderSettings(
            enabled=_env_bool("GEMINI_ENABLED", False),
            model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
            timeout_seconds=_positive_float("GEMINI_TIMEOUT_SECONDS", 30.0),
        ),
    }
    return ProviderRuntimeConfig(
        provider_order=provider_order,
        providers=providers,
        max_attempts=_positive_int("LLM_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS),
        max_total_time_seconds=_positive_float(
            "LLM_MAX_TOTAL_TIME_SECONDS", DEFAULT_MAX_TOTAL_TIME_SECONDS
        ),
        retry_wait_seconds=_positive_float("LLM_RETRY_WAIT_SECONDS", DEFAULT_RETRY_WAIT_SECONDS),
    )


def enabled_provider_names(config: ProviderRuntimeConfig) -> tuple[str, ...]:
    """Return enabled providers in configured order, excluding unknown names."""
    return tuple(
        name for name in config.provider_order
        if name in config.providers and config.providers[name].enabled
    )


def require_enabled_provider_config(
    config: ProviderRuntimeConfig | None = None,
) -> ProviderRuntimeConfig:
    """Load and validate server model configuration before serving requests."""
    selected = config or get_provider_runtime_config()
    if not enabled_provider_names(selected):
        raise RuntimeError("no model provider configured")
    return selected
