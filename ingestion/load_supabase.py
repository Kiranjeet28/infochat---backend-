"""Load enriched notice JSON files into Supabase, ignoring known content hashes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests


DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 30


def _notice_row(notice: dict[str, Any]) -> dict[str, Any] | None:
    """Map the extraction JSON shape to the public.notices table columns."""
    content_hash = notice.get("content_hash")
    if not isinstance(content_hash, str) or not content_hash.strip():
        return None

    metadata = notice.get("metadata") or {}
    source = notice.get("source") or {}
    notice_id = notice.get("notice_id", notice.get("_id"))
    if notice_id is None:
        raise ValueError(f"Notice with content_hash {content_hash!r} has no notice_id or _id")

    row = {
        "notice_id": str(notice_id),
        "title": notice.get("title") or "Untitled notice",
        "source_url": source.get("official_notice_url") or notice.get("source_url") or notice.get("url"),
        "published_at": notice.get("published_at") or metadata.get("date") or notice.get("date"),
        "category": notice.get("category", metadata.get("category")),
        "department": notice.get("department", metadata.get("department")),
        "semester": notice.get("semester", metadata.get("semester")),
        "deadline": notice.get("deadline", metadata.get("deadline")),
        "required_action": notice.get("required_action", metadata.get("required_action")),
        "content_hash": content_hash,
        "fetched_at": notice.get("fetched_at") or source.get("fetched_at"),
        "review_status": notice.get("review_status") or "needs_review",
        "extraction_confidence": notice.get("extraction_confidence"),
        "supersedes_id": notice.get("supersedes_id"),
        "superseded_by": notice.get("superseded_by"),
        "text": notice.get("text"),
    }
    if row["fetched_at"] is None:
        del row["fetched_at"]  # Let the database's now() default apply.
    return row


def read_notice_rows(data_dir: Path = DATA_DIR) -> tuple[list[dict[str, Any]], int, int]:
    """Read notices*.json, dropping unhashed raw records and duplicate input hashes."""
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    ignored_unhashed = 0
    ignored_duplicate_files = 0

    for path in sorted(data_dir.glob("notices*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}")
        for notice in payload:
            if not isinstance(notice, dict):
                raise ValueError(f"Expected notice objects in {path}")
            row = _notice_row(notice)
            if row is None:
                ignored_unhashed += 1
                continue
            if row["content_hash"] in seen_hashes:
                ignored_duplicate_files += 1
                continue
            seen_hashes.add(row["content_hash"])
            rows.append(row)

    if not rows:
        raise ValueError(f"No notices with content_hash found in {data_dir}")
    return rows, ignored_unhashed, ignored_duplicate_files


def load_notices() -> tuple[int, int, int]:
    """Insert new notices through Supabase REST; the unique hash handles races."""
    project_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not project_url or not service_role_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the process environment. "
            "Do not commit the service role key to the repository."
        )

    rows, ignored_unhashed, ignored_duplicate_files = read_notice_rows()
    endpoint = f"{project_url}/rest/v1/notices"
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=representation",
    }

    inserted = 0
    session = requests.Session()
    for start in range(0, len(rows), BATCH_SIZE):
        response = session.post(
            endpoint,
            params={"on_conflict": "content_hash"},
            headers=headers,
            json=rows[start : start + BATCH_SIZE],
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()
        inserted += len(result) if isinstance(result, list) else 0

    skipped_existing = len(rows) - inserted
    print(f"Inserted {inserted} notices")
    print(f"Skipped {skipped_existing} notices with an existing content_hash")
    print(f"Ignored {ignored_unhashed} raw notices without content_hash")
    print(f"Ignored {ignored_duplicate_files} duplicate hashes across JSON files")
    return inserted, skipped_existing, ignored_unhashed


if __name__ == "__main__":
    load_notices()
