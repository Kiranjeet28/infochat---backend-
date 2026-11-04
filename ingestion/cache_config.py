"""Answer cache configuration; disabled unless explicitly enabled."""

from __future__ import annotations

import os


def cache_enabled() -> bool:
    return os.getenv("CACHE_ENABLED", "false").strip().casefold() in {"1", "true", "yes", "on"}


def cache_ttl_seconds() -> int:
    try:
        return max(1, int(os.getenv("CACHE_TTL_SECONDS", "21600")))
    except ValueError:
        return 21600
