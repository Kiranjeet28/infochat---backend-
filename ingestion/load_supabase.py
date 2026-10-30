"""Load enriched notice JSON files into Supabase, ignoring known content hashes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


DATA_DIR = Path(__file__).parent / "data"
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
BATCH_SIZE = 100
PAGE_SIZE = 1000
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
        "superseded_by": notice.get("superseded_by") or None,
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


def notices_missing_published_at(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return identifiers and titles for rows without an explicit publication date."""
    return [
        {"notice_id": str(row.get("notice_id") or "unknown"), "title": str(row.get("title") or "Untitled notice")}
        for row in rows
        if not str(row.get("published_at") or "").strip()
    ]


def _fetch_existing_notices(
    session: requests.Session, endpoint: str, headers: dict[str, str]
) -> tuple[dict[str, str], set[str]]:
    """Return current hashes keyed by notice ID and all hashes already stored."""
    by_id: dict[str, str] = {}
    hashes: set[str] = set()
    offset = 0
    while True:
        response = session.get(
            endpoint,
            params={
                "select": "notice_id,content_hash",
                "order": "notice_id.asc",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        page = response.json()
        for existing in page:
            by_id[existing["notice_id"]] = existing["content_hash"]
            hashes.add(existing["content_hash"])
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return by_id, hashes


def load_notices() -> tuple[int, int, int, int]:
    """Skip known hashes, update changed notice IDs, and insert new notices."""
    project_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not project_url or not service_role_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the process environment. "
            "Do not commit the service role key to the repository."
        )

    rows, ignored_unhashed, ignored_duplicate_files = read_notice_rows()
    missing_publication_dates = notices_missing_published_at(rows)
    if missing_publication_dates:
        print("Notices missing published_at (no dates inferred):")
        for notice in missing_publication_dates:
            print(f"- {notice['notice_id']}: {notice['title']}")
    else:
        print("All loaded notices have published_at.")
    endpoint = f"{project_url}/rest/v1/notices"
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=representation",
    }

    inserted = updated = skipped_existing = 0
    insert_rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        existing_by_id, existing_hashes = _fetch_existing_notices(session, endpoint, headers)
        for row in rows:
            notice_id = row["notice_id"]
            content_hash = row["content_hash"]
            current_hash = existing_by_id.get(notice_id)

            if content_hash in existing_hashes:
                skipped_existing += 1
                continue

            if current_hash is not None:
                # Keep the operator's review decision when refreshing changed notice content.
                update_row = {key: value for key, value in row.items() if key != "review_status"}
                response = session.patch(
                    endpoint,
                    params={"notice_id": f"eq.{notice_id}"},
                    headers={**headers, "Prefer": "return=minimal"},
                    json=update_row,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                existing_hashes.discard(current_hash)
                existing_hashes.add(content_hash)
                existing_by_id[notice_id] = content_hash
                updated += 1
            else:
                insert_rows.append(row)

        for start in range(0, len(insert_rows), BATCH_SIZE):
            response = session.post(
                endpoint,
                params={"on_conflict": "content_hash"},
                headers=headers,
                json=insert_rows[start : start + BATCH_SIZE],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
            inserted += len(result) if isinstance(result, list) else 0

    skipped_existing += len(insert_rows) - inserted
    print(f"Inserted {inserted} notices")
    print(f"Updated {updated} notices with changed content_hash")
    print(f"Skipped {skipped_existing} notices with an existing content_hash")
    print(f"Ignored {ignored_unhashed} raw notices without content_hash")
    print(f"Ignored {ignored_duplicate_files} duplicate hashes across JSON files")
    return inserted, updated, skipped_existing, ignored_unhashed


if __name__ == "__main__":
    load_notices()
