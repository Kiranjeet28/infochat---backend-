"""Split eligible Supabase notices into overlapping token-sized chunks."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


PAGE_SIZE = 1000
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
BATCH_SIZE = 100
REQUEST_TIMEOUT_SECONDS = 30
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
INDEX_VERSION = "notices-v1-rcs400-overlap50-minilm384"


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


def _fetch_notices(
    session: requests.Session, project_url: str, headers: dict[str, str]
) -> list[dict[str, Any]]:
    endpoint = f"{project_url}/rest/v1/notices"
    notices: list[dict[str, Any]] = []
    offset = 0

    while True:
        response = session.get(
            endpoint,
            params={
                "select": "notice_id,title,published_at,source_url,category,department,semester,text,content_hash,review_status,superseded_by,index_version",
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

    return notices


def _fetch_existing_chunks(
    session: requests.Session, project_url: str, headers: dict[str, str]
) -> dict[str, list[dict[str, Any]]]:
    endpoint = f"{project_url}/rest/v1/chunks"
    existing: dict[str, list[dict[str, Any]]] = {}
    offset = 0

    while True:
        response = session.get(
            endpoint,
            params={
                "select": "notice_id,content_hash,chunk_index",
                "order": "notice_id.asc,chunk_index.asc",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        page = response.json()
        for row in page:
            existing.setdefault(row["notice_id"], []).append(row)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    return existing


def chunk_notice(
    notice: dict[str, Any], splitter: RecursiveCharacterTextSplitter
) -> list[dict[str, Any]]:
    chunks = splitter.split_text(notice["text"].strip())
    return [
        {
            "notice_id": notice["notice_id"],
            "chunk_index": index,
            "text": chunk,
            "content_hash": notice["content_hash"],
            "embedding_model": EMBEDDING_MODEL,
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


def _embed_rows(chunk_rows: list[dict[str, Any]]) -> tuple[int, int]:
    """Attach vectors when available; preserve rows and record failures otherwise."""
    model = None
    model_error: str | None = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as error:  # Model download, dependency, or runtime failure.
        model_error = f"{type(error).__name__}: {error}"[:1000]

    succeeded = 0
    failed = 0
    for row in chunk_rows:
        row["embedding_model"] = EMBEDDING_MODEL
        row["embedding"] = None
        if model_error:
            row["embedding_status"] = "failed"
            row["embedding_error"] = model_error
            failed += 1
            continue

        try:
            vector = model.encode(row["text"], normalize_embeddings=True).tolist()
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}"
                )
            row["embedding"] = [float(value) for value in vector]
            row["embedding_status"] = "succeeded"
            row["embedding_error"] = None
            succeeded += 1
        except Exception as error:  # A single chunk failure must not block the batch.
            row["embedding_status"] = "failed"
            row["embedding_error"] = f"{type(error).__name__}: {error}"[:1000]
            failed += 1
    return succeeded, failed


def _chunks_are_current(
    notice: dict[str, Any], existing: list[dict[str, Any]], new_chunks: list[dict[str, Any]]
) -> bool:
    if not existing or len(existing) != len(new_chunks):
        return False
    expected_indexes = set(range(len(new_chunks)))
    return (
        all(row.get("content_hash") == notice.get("content_hash") for row in existing)
        and {row["chunk_index"] for row in existing} == expected_indexes
        and notice.get("index_version") == INDEX_VERSION
    )


def _delete_notice_chunks(
    session: requests.Session,
    project_url: str,
    headers: dict[str, str],
    notice_ids: set[str],
) -> dict[str, str]:
    endpoint = f"{project_url}/rest/v1/chunks"
    failures: dict[str, str] = {}
    for notice_id in sorted(notice_ids):
        try:
            response = session.delete(
                endpoint,
                params={"notice_id": f"eq.{notice_id}"},
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except requests.RequestException as error:
            failures[notice_id] = f"Could not delete old chunks: {error}"[:1000]
    return failures


def _record_index_status(
    session: requests.Session,
    project_url: str,
    headers: dict[str, str],
    notice_id: str,
    status: str,
    reason: str | None = None,
) -> None:
    response = session.patch(
        f"{project_url}/rest/v1/notices",
        params={"notice_id": f"eq.{notice_id}"},
        headers={**headers, "Prefer": "return=minimal"},
        json={
            "index_status": status,
            "index_status_reason": reason,
            "index_version": INDEX_VERSION,
            "index_status_at": datetime.now(timezone.utc).isoformat(),
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()


def _skip_reason(notice: dict[str, Any]) -> str:
    if not isinstance(notice.get("source_url"), str) or not notice["source_url"].strip():
        return "missing source_url"
    if notice.get("superseded_by") is not None:
        return f"Superseded by notice {notice['superseded_by']}"
    if notice.get("review_status") == "needs_review":
        return "Notice requires review"
    return "Notice has no extracted text"


def split_and_load() -> tuple[int, int, int]:
    project_url, headers = _credentials()
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )

    with requests.Session() as session:
        notices = _fetch_notices(session, project_url, headers)
        existing_by_notice = _fetch_existing_chunks(session, project_url, headers)
        eligible_notices = [
            notice
            for notice in notices
            if notice.get("superseded_by") is None
            and notice.get("review_status") != "needs_review"
            and isinstance(notice.get("text"), str)
            and notice["text"].strip()
            and isinstance(notice.get("source_url"), str)
            and notice["source_url"].strip()
        ]
        eligible_ids = {notice["notice_id"] for notice in eligible_notices}

        chunk_rows_by_notice: dict[str, list[dict[str, Any]]] = {}
        delete_ids = set(existing_by_notice) - eligible_ids
        skipped_unchanged = 0
        failed_notices: dict[str, str] = {}
        for notice in eligible_notices:
            notice_id = notice["notice_id"]
            old_chunks = existing_by_notice.get(notice_id, [])
            try:
                new_chunks = chunk_notice(notice, splitter)
            except Exception as error:
                delete_ids.add(notice_id)
                failed_notices[notice_id] = f"Chunking failed: {type(error).__name__}: {error}"[:1000]
                continue
            if _chunks_are_current(notice, old_chunks, new_chunks):
                skipped_unchanged += 1
                try:
                    _record_index_status(session, project_url, headers, notice_id, "indexed")
                except requests.RequestException as error:
                    failed_notices[notice_id] = f"Could not record indexed status: {error}"[:1000]
                continue
            if old_chunks:
                delete_ids.add(notice_id)
            chunk_rows_by_notice[notice_id] = new_chunks

        # Mark notices that are not eligible for chunking and remove any stale chunks.
        for notice in notices:
            if notice["notice_id"] in eligible_ids:
                continue
            reason = _skip_reason(notice)
            try:
                _record_index_status(
                    session, project_url, headers, notice["notice_id"], "skipped", reason
                )
            except requests.RequestException as error:
                failed_notices[notice["notice_id"]] = f"Could not record skipped status: {error}"[:1000]

        chunk_rows = [row for rows in chunk_rows_by_notice.values() for row in rows]
        embedded, failed = _embed_rows(chunk_rows) if chunk_rows else (0, 0)
        # Remove stale chunks after replacements are prepared, before writing new chunks.
        delete_failures = _delete_notice_chunks(session, project_url, headers, delete_ids)
        failed_notices.update(delete_failures)
        endpoint = f"{project_url}/rest/v1/chunks"
        insert_headers = {
            **headers,
            "Prefer": "resolution=merge-duplicates,return=representation",
        }
        saved = 0
        for notice_id, rows in chunk_rows_by_notice.items():
            if notice_id in delete_failures:
                continue
            try:
                for start in range(0, len(rows), BATCH_SIZE):
                    response = session.post(
                        endpoint,
                        params={"on_conflict": "notice_id,chunk_index"},
                        headers=insert_headers,
                        json=rows[start : start + BATCH_SIZE],
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    response.raise_for_status()
                    result = response.json()
                    saved += len(result) if isinstance(result, list) else 0
                _record_index_status(session, project_url, headers, notice_id, "indexed")
            except requests.RequestException as error:
                failed_notices[notice_id] = f"Chunk save failed: {error}"[:1000]

        for notice_id, reason in failed_notices.items():
            try:
                _record_index_status(
                    session, project_url, headers, notice_id, "failed", reason
                )
            except requests.RequestException as error:
                print(f"Could not record failed status for {notice_id}: {error}")

    print(f"Eligible notices: {len(eligible_notices)}")
    print(f"Unchanged notices skipped: {skipped_unchanged}")
    print(f"Old chunk groups removed: {len(delete_ids) - len(delete_failures)}")
    print(f"Chunks prepared: {len(chunk_rows)}")
    print(f"Chunks saved or updated: {saved}")
    print(f"Embeddings succeeded: {embedded}")
    print(f"Embeddings failed; text retained: {failed}")
    print(f"Notices marked failed: {len(failed_notices)}")
    return len(eligible_notices), saved, len(failed_notices)


if __name__ == "__main__":
    split_and_load()
