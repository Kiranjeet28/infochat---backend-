"""Pure-Python fallback for the ``uuid_utils`` native module.

WORKAROUND for Windows Application Control, which can block the compiled
``uuid_utils\\_uuid_utils.pyd`` ("DLL load failed ... Application Control
policy has blocked this file"). langchain_core only needs ``uuid7`` from
``uuid_utils.compat`` to create run ids, so the fallback affects UUID
generation for run ids only; it has no security role. The clean fix is a
package rollback or running the backend in WSL.

Call ``ensure_uuid_utils()`` BEFORE any langchain import.
"""

from __future__ import annotations

import logging
import secrets
import sys
import threading
import time
import types
import uuid

_LOGGER = logging.getLogger(__name__)
_WARNING = "uuid_utils native module blocked; using pure-Python fallback"

_lock = threading.Lock()
_last_ms = -1
_last_counter = 0


def uuid7(timestamp: int | None = None, nanos: int | None = None) -> uuid.UUID:
    """Return an RFC 9562 UUIDv7 (same call shape as ``uuid_utils.compat.uuid7``).

    ``timestamp`` is Unix seconds and ``nanos`` the extra nanoseconds, as
    langchain_core passes them. Layout: 48-bit unix ms | version 7 | 12 random
    bits | variant 10 | 62 random bits. Values generated from the clock are
    monotonic within a process (the 74 random bits act as a counter).
    """
    global _last_ms, _last_counter

    if timestamp is None:
        ms = time.time_ns() // 1_000_000
        with _lock:
            if ms <= _last_ms:
                ms = _last_ms
                counter = _last_counter + 1
                if counter >= 1 << 74:
                    ms += 1
                    counter = secrets.randbits(73)
            else:
                counter = secrets.randbits(73)  # MSB 0 leaves room to count up
            _last_ms, _last_counter = ms, counter
    else:
        ms = timestamp * 1000 + (nanos or 0) // 1_000_000
        counter = secrets.randbits(74)

    rand_a = (counter >> 62) & 0xFFF
    rand_b = counter & ((1 << 62) - 1)
    value = (
        ((ms & 0xFFFFFFFFFFFF) << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )
    return uuid.UUID(int=value)


def _install_fallback() -> None:
    for name in [n for n in sys.modules if n == "uuid_utils" or n.startswith("uuid_utils.")]:
        del sys.modules[name]
    package = types.ModuleType("uuid_utils")
    package.__path__ = []  # mark as a package so "uuid_utils.compat" resolves
    compat = types.ModuleType("uuid_utils.compat")
    compat.uuid7 = uuid7
    package.compat = compat
    package.uuid7 = uuid7
    sys.modules["uuid_utils"] = package
    sys.modules["uuid_utils.compat"] = compat


def ensure_uuid_utils() -> bool:
    """Install the fallback if the real module cannot be imported.

    Returns True when the fallback was installed, False when the real module
    (or an already-installed fallback) is in use.
    """
    try:
        import uuid_utils.compat  # noqa: F401
    except ImportError:  # includes "DLL load failed" / Application Control
        _install_fallback()
        _LOGGER.warning(_WARNING)
        return True
    return False
