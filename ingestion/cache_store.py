"""Privacy-conscious exact-match answer cache stores and key helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import re
from typing import Any, Callable, Protocol

import requests


try:
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")
except Exception:  # Windows may have no installed IANA timezone database.
    IST = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DIGIT_RUN_PATTERN = re.compile(r"\d{6,}")
PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
TRAILING_PUNCTUATION = " .,!?:;。،؛؟—–-"


class CacheStore(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...
    def set(self, entry: dict[str, Any]) -> None: ...
    def delete_by_notice(self, notice_id: str) -> None: ...
    def delete(self, key: str) -> None: ...
    def delete_all(self) -> None: ...


class InMemoryCacheStore:
    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        entry = self.entries.get(key)
        return dict(entry) if entry is not None else None

    def set(self, entry: dict[str, Any]) -> None:
        self.entries[entry["cache_key"]] = dict(entry)

    def delete_by_notice(self, notice_id: str) -> None:
        for key, entry in list(self.entries.items()):
            if notice_id in entry.get("source_notice_ids", []):
                del self.entries[key]

    def delete(self, key: str) -> None:
        self.entries.pop(key, None)

    def delete_all(self) -> None:
        self.entries.clear()


@dataclass
class SupabaseCacheStore:
    """PostgREST store. Construct/use only when caching is enabled."""

    project_url: str
    service_role_key: str
    timeout_seconds: int = 15

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        request_headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }
        request_headers.update(kwargs.pop("headers", {}))
        return requests.request(
            method,
            f"{self.project_url.rstrip('/')}/rest/v1/{path}",
            headers=request_headers,
            timeout=self.timeout_seconds,
            **kwargs,
        )

    def get(self, key: str) -> dict[str, Any] | None:
        response = self._request(
            "GET", "answer_cache", params={"select": "*", "cache_key": f"eq.{key}", "limit": 1}
        )
        response.raise_for_status()
        rows = response.json()
        return rows[0] if rows else None

    def set(self, entry: dict[str, Any]) -> None:
        response = self._request(
            "POST", "answer_cache", params={"on_conflict": "cache_key"},
            headers={"Prefer": "resolution=merge-duplicates,return=minimal"}, json=entry,
        )
        response.raise_for_status()

    def delete_by_notice(self, notice_id: str) -> None:
        response = self._request(
            "DELETE", "answer_cache", params={"source_notice_ids": f"cs.{{{notice_id}}}"}
        )
        response.raise_for_status()

    def delete(self, key: str) -> None:
        response = self._request("DELETE", "answer_cache", params={"cache_key": f"eq.{key}"})
        response.raise_for_status()

    def delete_all(self) -> None:
        response = self._request("DELETE", "answer_cache", params={"cache_key": "not.is.null"})
        response.raise_for_status()


def normalize_question(question: str) -> str:
    normalized = " ".join(question.casefold().split())
    return normalized.rstrip(TRAILING_PUNCTUATION).rstrip()


def question_hash(question: str) -> str:
    return hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()


def privacy_guard(question: str) -> bool:
    return bool(
        EMAIL_PATTERN.search(question)
        or DIGIT_RUN_PATTERN.search(question)
        or PHONE_PATTERN.search(question)
    )


def cache_key(
    question: str,
    department: str | None,
    semester: str | None,
    kb_version: int | str,
    prompt_version: str,
    today: date | None = None,
) -> str:
    local_day = today or datetime.now(IST).date()
    payload = json.dumps(
        [normalize_question(question), department, semester, str(kb_version), prompt_version, local_day.isoformat()],
        ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cacheable_answer(result: dict[str, Any], citations_valid: bool) -> bool:
    return result.get("status") == "answered" and citations_valid is True


def cache_answer_if_valid(
    store: CacheStore,
    result: dict[str, Any],
    citations_valid: bool,
    entry: dict[str, Any],
) -> bool:
    """Persist only a successfully answered result whose citations were checked."""
    if not cacheable_answer(result, citations_valid):
        return False
    store.set(entry)
    return True


def validated_cache_hit(
    store: CacheStore,
    key: str,
    *,
    now: datetime,
    kb_version: str | int,
    prompt_version: str,
    source_verifier: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, Any] | None, int | None]:
    """Return a current entry and age, or delete invalid/expired entries."""
    entry = store.get(key)
    if entry is None:
        return None, None
    expires = datetime.fromisoformat(str(entry["expires_at"]).replace("Z", "+00:00"))
    if expires <= now or str(entry.get("kb_version")) != str(kb_version) \
            or entry.get("prompt_version") != prompt_version \
            or entry.get("status") != "answered":
        store.delete(key)
        return None, None
    if not source_verifier(entry):
        store.delete(key)
        return None, None
    created = datetime.fromisoformat(str(entry["created_at"]).replace("Z", "+00:00"))
    return entry, max(0, int((now - created).total_seconds()))


def expiry_at(now: datetime, ttl_seconds: int) -> datetime:
    if now.tzinfo is None:
        now = now.replace(tzinfo=IST)
    now_ist = now.astimezone(IST)
    midnight = datetime.combine(now_ist.date() + timedelta(days=1), time.min, tzinfo=IST)
    return min(now_ist + timedelta(seconds=ttl_seconds), midnight)


def make_cache_entry(
    *,
    cache_key_value: str,
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    source_content_hashes: dict[str, str],
    kb_version: str | int,
    prompt_version: str,
    provider: str | None,
    model: str | None,
    now: datetime,
    ttl_seconds: int,
) -> dict[str, Any]:
    """Build a cache row without retaining the question or source content hashes in cards."""
    safe_sources = [
        {key: source.get(key) for key in ("notice_id", "title", "published_at", "source_url")}
        for source in sources
    ]
    source_notice_ids = list(dict.fromkeys(
        str(source["notice_id"]) for source in safe_sources if source.get("notice_id")
    ))
    return {
        "cache_key": cache_key_value,
        "normalized_question_hash": question_hash(question),
        "answer": answer,
        "status": "answered",
        "sources": safe_sources,
        "source_notice_ids": source_notice_ids,
        "source_content_hashes": source_content_hashes,
        "kb_version": int(kb_version),
        "prompt_version": prompt_version,
        "provider": provider,
        "model": model,
        "created_at": now.isoformat(),
        "expires_at": expiry_at(now, ttl_seconds).isoformat(),
        "hit_count": 0,
    }


def bump_kb_version(project_url: str | None = None, service_role_key: str | None = None) -> None:
    """Increment kb_state through its RPC; called only after real ingestion mutations."""
    import os

    url = (project_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
    key = service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE credentials are required to bump kb_version")
    response = requests.post(
        f"{url}/rest/v1/rpc/bump_kb_version",
        headers={"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={}, timeout=15,
    )
    response.raise_for_status()


def read_kb_version(project_url: str, service_role_key: str) -> str:
    response = requests.post(
        f"{project_url.rstrip('/')}/rest/v1/rpc/get_kb_version",
        headers={"apikey": service_role_key, "Authorization": f"Bearer {service_role_key}", "Content-Type": "application/json"},
        json={}, timeout=15,
    )
    response.raise_for_status()
    return str(response.json())


def verify_cached_sources(project_url: str, service_role_key: str, entry: dict[str, Any]) -> bool:
    """Verify all cached source IDs/hashes in one RPC call; withdrawn/superseded fails."""
    response = requests.post(
        f"{project_url.rstrip('/')}/rest/v1/rpc/verify_cached_sources",
        headers={"apikey": service_role_key, "Authorization": f"Bearer {service_role_key}", "Content-Type": "application/json"},
        json={
            "source_notice_ids": entry.get("source_notice_ids", []),
            "source_content_hashes": entry.get("source_content_hashes", {}),
        }, timeout=15,
    )
    response.raise_for_status()
    return response.json() is True


def invalidate_notice_cache(notice_ids: list[str] | set[str]) -> None:
    """Delete affected answer entries and bump kb_version after ingestion changes."""
    import os

    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE credentials are required to invalidate the answer cache")
    store = SupabaseCacheStore(url, key)
    for notice_id in sorted(set(notice_ids)):
        store.delete_by_notice(notice_id)
    bump_kb_version(url, key)
