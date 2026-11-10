"""In-memory abuse and cost limits for the API.

Limits live in process memory: they reset on restart and are per instance, so with
several workers or instances each one counts on its own. Clients are identified by a
salted hash of the IP address; raw addresses are never stored.
"""

from __future__ import annotations

import hashlib
import math
import os
import secrets
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Callable

IST = timezone(timedelta(hours=5, minutes=30))
DAY_SECONDS = 24 * 60 * 60
_PROCESS_SALT = secrets.token_hex(16)


def env_int(name: str, default: int) -> int:
    """Non-negative integer setting; 0 turns the limit off; garbage falls back to the default."""
    try:
        return max(0, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


def client_id(client_host: str | None, forwarded_for: str | None = None) -> str:
    """Salted hash of the client address; the first X-Forwarded-For hop only if TRUST_PROXY_HEADERS=1."""
    address = client_host or "unknown"
    if os.getenv("TRUST_PROXY_HEADERS") == "1" and forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            address = first
    salt = os.getenv("RATE_LIMIT_SALT") or _PROCESS_SALT
    return hashlib.sha256(f"{salt}:{address}".encode()).hexdigest()[:24]


class SlidingWindowLimiter:
    """Per-client sliding windows, e.g. 10 per minute and 80 per day."""

    def __init__(self, windows: Callable[[], list[tuple[int, int]]], clock: Callable[[], float] = time.monotonic,
                 max_clients: int = 10000) -> None:
        self._windows, self._clock, self._max_clients = windows, clock, max_clients
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def hit(self, client: str) -> int | None:
        """Record a request; return seconds to wait if the client is over a limit (and record nothing)."""
        windows = [(seconds, limit) for seconds, limit in self._windows() if limit > 0]
        if not windows:
            return None
        now = self._clock()
        longest = max(seconds for seconds, _ in windows)
        with self._lock:
            hits = self._hits.setdefault(client, deque())
            while hits and now - hits[0] >= longest:
                hits.popleft()
            wait = 0
            for seconds, limit in windows:
                inside = [t for t in hits if now - t < seconds]
                if len(inside) >= limit:
                    wait = max(wait, math.ceil(inside[len(inside) - limit] + seconds - now))
            if wait:
                return max(1, wait)
            hits.append(now)
            if len(self._hits) > self._max_clients:
                self._evict(now, longest)
        return None

    def _evict(self, now: float, longest: int) -> None:
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= longest]:
            del self._hits[key]
        while len(self._hits) > self._max_clients:  # still full of active clients: drop the least recent
            del self._hits[min(self._hits, key=lambda k: self._hits[k][-1])]


class GlobalModelCap:
    """Daily cap on real model calls across all clients (IST day, like the answer cache)."""

    def __init__(self, limit: Callable[[], int], today: Callable[[], str] | None = None) -> None:
        self._limit = limit
        self._today = today or (lambda: datetime.now(IST).date().isoformat())
        self._day, self._used = self._today(), 0
        self._lock = threading.Lock()

    def _roll(self) -> None:
        today = self._today()
        if today != self._day:
            self._day, self._used = today, 0

    def reserve(self) -> bool:
        """Take one call; False when the day's cap is reached (0 disables the cap)."""
        limit = self._limit()
        with self._lock:
            self._roll()
            if limit and self._used >= limit:
                return False
            self._used += 1
            return True

    def adjust(self, delta: int) -> None:
        """Correct a reservation: -1 refunds a call that never reached a provider; +n adds retries."""
        with self._lock:
            self._roll()
            self._used = max(0, self._used + delta)

    @property
    def used(self) -> int:
        with self._lock:
            self._roll()
            return self._used
