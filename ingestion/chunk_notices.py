"""Split eligible Supabase notices into overlapping token-sized chunks."""

from __future__ import annotations

import os
from typing import Any

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter


PAGE_SIZE = 1000
BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 30
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50


def _credentials() -> tuple[str, dict[str, str]]:
    project_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not project_url or not service_role_key:
        raise RuntimeError(
            "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the process environment. "
            "Do not commit the service role key to the repository."
        )
    return project_url, {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
    }


def _fetch_eligible_notices(
    session: requests.Session, project_url: str, headers: dict[str, str]
) -> list[dict[str, Any]]:
    endpoint = f"{project_url}/rest/v1/notices"
    notices: list[dict[str, Any]] = []
    offset = 0

    while True:
        response = session.get(
            endpoint,
            params={
                "select": "notice_id,title,published_at,source_url,category,department,semester,text,review_status,superseded_by",
                "superseded_by": "is.null",
                "review_status": "neq.needs_review",
                "order": "notice_id.asc",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        page = response.json()
        notices.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    # Keep the eligibility guard here as well as in the REST query.
    return [
        notice
        for notice in notices
        if notice.get("superseded_by") is None
        and notice.get("review_status") != "needs_review"
        and isinstance(notice.get("text"), str)
        and notice["text"].strip()
    ]


def chunk_notice(
    notice: dict[str, Any], splitter: RecursiveCharacterTextSplitter
) -> list[dict[str, Any]]:
    chunks = splitter.split_text(notice["text"].strip())
    return [
        {
            "notice_id": notice["notice_id"],
            "chunk_index": index,
            "text": chunk,
            "title": notice.get("title"),
            "published_at": notice.get("published_at"),
            "source_url": notice.get("source_url"),
            "category": notice.get("category"),
            "department": notice.get("department"),
            "semester": notice.get("semester"),
        }
        for index, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def split_and_load() -> tuple[int, int, int]:
    project_url, headers = _credentials()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    with requests.Session() as session:
        notices = _fetch_eligible_notices(session, project_url, headers)
        if not notices:
            print("No eligible notices found; load reviewed notices first.")
            return 0, 0, 0

        chunk_rows = [
            row
            for notice in notices
            for row in chunk_notice(notice, splitter)
        ]
        endpoint = f"{project_url}/rest/v1/chunks"
        insert_headers = {
            **headers,
            "Prefer": "resolution=ignore-duplicates,return=representation",
        }
        inserted = 0
        for start in range(0, len(chunk_rows), BATCH_SIZE):
            response = session.post(
                endpoint,
                params={"on_conflict": "notice_id,chunk_index"},
                headers=insert_headers,
                json=chunk_rows[start : start + BATCH_SIZE],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            result = response.json()
            inserted += len(result) if isinstance(result, list) else 0

    skipped_existing = len(chunk_rows) - inserted
    print(f"Eligible notices: {len(notices)}")
    print(f"Chunks prepared: {len(chunk_rows)}")
    print(f"Chunks inserted: {inserted}")
    print(f"Chunks skipped because they already exist: {skipped_existing}")
    return len(notices), inserted, skipped_existing


if __name__ == "__main__":
    split_and_load()
