"""Single entrypoint for the offline-testable CascadeIQ notice pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

SOURCE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SOURCE_ROOT / ".env", override=False)
sys.path.insert(0, str(SOURCE_ROOT))
from ingestion.embedding_config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR, create_embedding_model  # noqa: E402
from ingestion.extract_text import process_notices, link_notice_versions  # noqa: E402
from ingestion.cache_store import bump_kb_version, read_kb_version  # noqa: E402
from ingestion.chunk_notices import (  # noqa: E402
    CHUNK_OVERLAP_TOKENS, CHUNK_SIZE_TOKENS, EMBEDDING_DIMENSIONS,
    INDEX_VERSION, chunk_notice,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

API_TIMEOUT = 90
WARMUP_ATTEMPTS = 4
WARMUP_TOTAL_SECONDS = 240
MIN_TEXT_LENGTH = int(os.getenv("MIN_NOTICE_TEXT_LENGTH", "40"))
ALLOWED_SOURCE_DOMAINS = tuple(filter(None, os.getenv("ALLOWED_SOURCE_DOMAINS", "gndec.ac.in").split(",")))
CURRENT_STAGE = "startup"


def decide_review_status(notice: dict[str, Any], *, duplicate_hash: bool = False) -> tuple[str, list[str]]:
    """Conservative publication gate; reasons stay in the run summary."""
    reasons: list[str] = []
    if not str(notice.get("title") or "").strip():
        reasons.append("missing_title")
    url = str((notice.get("source") or {}).get("official_notice_url") or notice.get("source_url") or notice.get("url") or "")
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https" or not parsed_url.hostname:
        reasons.append("source_url_not_https")
    elif not any(parsed_url.hostname == d or parsed_url.hostname.endswith("." + d) for d in ALLOWED_SOURCE_DOMAINS):
        reasons.append("source_domain_not_allowed")
    date_value = notice.get("published_at") or notice.get("date") or (notice.get("metadata") or {}).get("date")
    try:
        published = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published > datetime.now(timezone.utc):
            reasons.append("published_at_in_future")
    except (TypeError, ValueError):
        reasons.append("published_at_invalid")
    confidence = notice.get("extraction_confidence")
    # validate_notice emits a numeric 0.00–1.00 evidence score, not labels.
    if not isinstance(confidence, (int, float)) or confidence < 0.6:
        reasons.append("extraction_confidence_not_high")
    if len(str(notice.get("text") or "").strip()) < MIN_TEXT_LENGTH:
        reasons.append("text_too_short")
    if notice.get("version_status") == "needs_review" or notice.get("version_review_candidates"):
        reasons.append("version_conflict")
    if duplicate_hash:
        reasons.append("duplicate_hash")
    return ("needs_review", reasons) if reasons else ("ok", [])


class RestStore:
    """Small PostgREST adapter; kept injectable for offline tests."""
    def __init__(self, url: str, key: str, session: Any = None):
        self.url, self.key = url.rstrip("/"), key
        self.session = session or requests.Session()
        self.headers = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        self.notices = f"{self.url}/rest/v1/notices"

    def request(self, method: str, path: str, **kwargs: Any):
        headers = {**self.headers, **kwargs.pop("headers", {})}
        response = self.session.request(method, f"{self.url}/rest/v1/{path}", headers=headers, timeout=30, **kwargs)
        response.raise_for_status()
        return response

    def known(self) -> list[dict[str, Any]]:
        response = self.request("GET", "notices", params={"select":"notice_id,content_hash,title,category,department,published_at,review_status,superseded_by", "limit":10000})
        return response.json()

    def version(self) -> str:
        return read_kb_version(self.url, self.key)

    def persist_notice(self, row: dict[str, Any], is_new: bool) -> None:
        if is_new:
            self.request("POST", "notices", params={"on_conflict":"content_hash"}, headers={**self.headers,"Prefer":"resolution=ignore-duplicates,return=minimal"}, json=row)
        else:
            self.request("PATCH", "notices", params={"notice_id":f"eq.{row['notice_id']}"}, headers={**self.headers,"Prefer":"return=minimal"}, json=row)

    def delete_chunks(self, notice_id: str) -> None:
        self.request("DELETE", "chunks", params={"notice_id":f"eq.{notice_id}"})

    def write_chunks(self, rows: list[dict[str, Any]]) -> int:
        if not rows: return 0
        response = self.request("POST", "chunks", params={"on_conflict":"notice_id,chunk_index"}, headers={**self.headers,"Prefer":"resolution=merge-duplicates,return=representation"}, json=rows)
        return len(response.json()) if isinstance(response.json(), list) else len(rows)

    def mark_superseded(self, notice_id: str, newer_id: str) -> None:
        self.request("PATCH", "notices", params={"notice_id":f"eq.{notice_id}"}, headers={**self.headers,"Prefer":"return=minimal"}, json={"superseded_by":newer_id})

    def withdrawn_available(self) -> bool:
        try:
            self.request("GET", "notices", params={"select":"withdrawn","limit":1})
            return True
        except requests.RequestException:
            return False

    def withdraw_missing(self, known_ids: set[str], source_ids: set[str]) -> None:
        missing = known_ids - source_ids
        for notice_id in missing:
            self.request("PATCH", "notices", params={"notice_id":f"eq.{notice_id}"}, headers={**self.headers,"Prefer":"return=minimal"}, json={"withdrawn":True})

    def invalidate(self, notice_ids: set[str]) -> int:
        invalidated = 0
        for notice_id in sorted(notice_ids):
            response = self.request("DELETE", "answer_cache", params={"source_notice_ids":f"cs.{{{notice_id}}}"}, headers={**self.headers,"Prefer":"return=representation"})
            try: invalidated += len(response.json())
            except ValueError: pass
        return invalidated


def fetch_page(session: Any, api_url: str, page: int) -> list[dict[str, Any]]:
    response = session.get(api_url, params={"page":page,"pageSize":9,"sort":"latest"}, timeout=API_TIMEOUT)
    response.raise_for_status()
    items = response.json().get("data", {}).get("items", [])
    stamp = datetime.now(timezone.utc).isoformat()
    for item in items:
        item["source"] = {"api_url":getattr(response,"url",api_url),"official_notice_url":item.get("url"),"fetched_at":stamp}
    return items


def warm_up(session: Any, api_url: str) -> None:
    started = time.monotonic()
    delay = 2
    last_error: Exception | None = None
    for attempt in range(WARMUP_ATTEMPTS):
        try:
            response = session.get(api_url, params={"page":1,"pageSize":1,"sort":"latest"}, timeout=API_TIMEOUT)
            response.raise_for_status()
            return
        except requests.RequestException as error:
            last_error = error
            if attempt + 1 < WARMUP_ATTEMPTS and time.monotonic() - started + delay < WARMUP_TOTAL_SECONDS:
                time.sleep(delay)
                delay = min(delay * 2, 30)
    raise RuntimeError("Notices API warm-up/fetch failed before database access") from last_error


def _row(notice: dict[str, Any]) -> dict[str, Any]:
    from ingestion.load_supabase import _notice_row
    row = _notice_row(notice)
    if row is None: raise ValueError("notice has no content hash")
    return row


def run(*, mode: str = "incremental", dry_run: bool = False, max_pages: int | None = None,
        api_url: str | None = None, store: Any = None, session: Any = None,
        processor=process_notices, embedder_factory=None) -> tuple[int, dict[str, Any]]:
    global CURRENT_STAGE
    CURRENT_STAGE = "configuration"
    started = time.monotonic()
    api_url = api_url or os.environ["NOTICES_API_URL"]
    session = session or requests.Session()
    pages_cap = max_pages or (5 if mode == "incremental" else 200)
    summary: dict[str, Any] = {"mode":mode,"dry_run":dry_run,"fetched":0,"new":0,"changed":0,"unchanged":0,"superseded":0,"needs_review":{},"length_only":[],"failed":0,"chunks_written":0,"embeddings_written":0,"kb_version_before":None,"kb_version_after":None,"cache_entries_invalidated":0,"warnings":[],"duration_seconds":0}
    CURRENT_STAGE = "notices_api_warmup"
    warm_up(session, api_url)
    if store is None:
        store = RestStore(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    CURRENT_STAGE = "supabase_read_known_notices"
    known_rows = store.known()
    known = {str(r["notice_id"]):r for r in known_rows}
    existing_hashes = {r["content_hash"] for r in known_rows if r.get("content_hash")}
    fetched: list[dict[str, Any]] = []
    processed_incremental: list[dict[str, Any]] = []
    completed = False
    for page in range(1, pages_cap + 1):
        CURRENT_STAGE = "notices_api_fetch"
        items = fetch_page(session, api_url, page)
        if not items:
            completed = True
            break
        fetched.extend(items)
        if mode == "incremental":
            page_processed = processor(items)
            processed_incremental.extend(page_processed)
            if page_processed and all(
                str(n.get("_id", n.get("notice_id"))) in known
                and known[str(n.get("_id", n.get("notice_id")))].get("content_hash") == n.get("content_hash")
                for n in page_processed
            ):
                break
    summary["fetched"] = len(fetched)
    if not completed and mode == "full": summary["warnings"].append("fetch did not reach an empty page before the hard page cap")
    CURRENT_STAGE = "supabase_read_kb_version"
    summary["kb_version_before"] = store.version()
    CURRENT_STAGE = "notice_extraction_and_review"
    processed = processed_incremental if mode == "incremental" else processor(fetched)
    if mode == "incremental" and processed:
        CURRENT_STAGE = "version_linking"
        # The linker compares extracted notices with lightweight stored metadata;
        # stored notice text is deliberately not fetched.
        candidates = []
        for row in known_rows:
            candidates.append({"_id": str(row["notice_id"]), "title": row.get("title") or "",
                "metadata": {"category": row.get("category"), "department": row.get("department"), "date": row.get("published_at")},
                "text": "", "content_hash": row.get("content_hash"), "extraction_confidence": 1.0,
                "version_status": "no_match", "_stored_candidate": True})
        for notice in processed:
            metadata = dict(notice.get("metadata") or {})
            metadata.setdefault("category", notice.get("category"))
            metadata.setdefault("department", notice.get("department"))
            link_date = notice.get("published_at") or notice.get("date")
            if link_date:
                try:
                    link_date = datetime.fromisoformat(str(link_date).replace("Z", "+00:00")).date().isoformat()
                except ValueError:
                    pass
            metadata.setdefault("date", link_date)
            notice["metadata"] = metadata
        linked = link_notice_versions(candidates + processed)
        linked_by_id = {str(n.get("_id")): n for n in linked if not n.get("_stored_candidate")}
        processed = [linked_by_id.get(str(n.get("_id", n.get("notice_id"))), n) for n in processed]
    changed_ids: set[str] = set()
    changed_notices: list[tuple[dict[str, Any], bool]] = []
    seen_hashes: set[str] = set(existing_hashes)
    source_ids: set[str] = set()
    for notice in processed:
        notice_id = str(notice.get("_id", notice.get("notice_id", "")))
        if not notice_id: summary["failed"] += 1; continue
        if notice.get("ingestion_status") == "failed":
            summary["failed"] += 1
            continue
        notice["notice_id"] = notice_id
        source_ids.add(notice_id)
        prior = known.get(notice_id)
        duplicate = notice.get("content_hash") in seen_hashes and (prior is None or prior.get("content_hash") != notice.get("content_hash"))
        seen_hashes.add(notice.get("content_hash"))
        status, reasons = decide_review_status(notice, duplicate_hash=duplicate)
        if reasons == ["text_too_short"]:
            summary["length_only"].append(notice_id)
        notice["review_status"] = status
        if reasons: summary["needs_review"][notice_id] = reasons
        if prior and prior.get("content_hash") == notice.get("content_hash"):
            summary["unchanged"] += 1
            # Preserve a hand-set review decision for identical content.
            continue
        if notice.get("content_hash") in existing_hashes:
            summary["unchanged"] += 1
            continue
        is_new = prior is None
        summary["new" if is_new else "changed"] += 1
        changed_ids.add(notice_id)
        if prior and prior.get("review_status"):
            notice["review_status"] = prior["review_status"]
        changed_notices.append((notice, is_new))
    # Prepare every vector before the first write so a model/dimension error is atomic.
    prepared_chunks: dict[str, list[dict[str, Any]]] = {}
    if changed_notices and not dry_run:
        CURRENT_STAGE = "chunk_and_embedding_preparation"
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS)
        factory = embedder_factory or (lambda: create_embedding_model(allow_download=os.getenv("EMBEDDING_ALLOW_DOWNLOAD") == "1"))
        model = factory()
        for notice, _ in changed_notices:
            row = _row(notice)
            if notice.get("review_status") != "ok" or not str(notice.get("text") or "").strip(): continue
            chunk_rows = chunk_notice(row, splitter)
            vectors = []
            for chunk in chunk_rows:
                vector = list(next(model.embed([chunk["text"]])))
                if len(vector) != EMBEDDING_DIMENSIONS:
                    raise RuntimeError(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(vector)}")
                vectors.append([float(x) for x in vector])
            for chunk, vector in zip(chunk_rows, vectors):
                chunk["embedding"] = vector
                chunk["embedding_model"] = EMBEDDING_MODEL
                chunk["index_version"] = INDEX_VERSION
            prepared_chunks[row["notice_id"]] = chunk_rows
            summary["embeddings_written"] += len(vectors)
    summary["superseded"] = sum(bool(n.get("supersedes_id")) for n in processed)
    for notice, is_new in changed_notices:
        if dry_run: continue
        CURRENT_STAGE = "supabase_write_notices"
        store.persist_notice(_row(notice), is_new)
    if not dry_run:
        CURRENT_STAGE = "supabase_delete_old_chunks"
        for notice, _ in changed_notices:
            store.delete_chunks(str(notice.get("_id", notice.get("notice_id"))))
    for notice in processed:
        older = notice.get("supersedes_id")
        if older:
            CURRENT_STAGE = "supabase_mark_superseded"
            changed_ids.add(str(older))
            if not dry_run: store.mark_superseded(str(older), str(notice.get("_id", notice.get("notice_id"))))
    if mode == "full":
        CURRENT_STAGE = "full_run_withdrawal_check"
        missing = set(known) - source_ids
        if not completed: summary["warnings"].append("withdrawal skipped: fetch was incomplete")
        elif len(missing) > 5 or (known and len(missing) / len(known) > .10): summary["warnings"].append("withdrawal skipped: missing count exceeds 5 or 10 percent threshold")
        elif not store.withdrawn_available(): summary["warnings"].append("withdrawal skipped: notices.withdrawn column is missing")
        elif missing:
            if not dry_run: store.withdraw_missing(set(known), source_ids)
            changed_ids.update(missing)
    for notice_id, chunk_rows in prepared_chunks.items():
        CURRENT_STAGE = "supabase_write_chunks"
        summary["chunks_written"] += store.write_chunks(chunk_rows)
    if changed_ids and not dry_run:
        CURRENT_STAGE = "cache_invalidation_and_kb_version_bump"
        summary["cache_entries_invalidated"] = store.invalidate(changed_ids)
        bump_kb_version(store.url, store.key)
    total = max(1, len(processed))
    if summary["failed"] / total > .20: raise RuntimeError("More than 20 percent of notices failed processing")
    if summary["failed"]:
        summary["warnings"].append("some notices failed processing")
    summary["duration_seconds"] = round(time.monotonic() - started, 2)
    CURRENT_STAGE = "supabase_read_final_kb_version"
    summary["kb_version_after"] = store.version()
    CURRENT_STAGE = "complete"
    return (2 if summary["warnings"] else 0), summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("incremental","full"), default="incremental")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args(argv)
    for name in ("NOTICES_API_URL","SUPABASE_URL","SUPABASE_SERVICE_ROLE_KEY"):
        if not os.getenv(name):
            print(f"Missing required environment variable: {name}", file=sys.stderr)
            return 1
    try:
        code, summary = run(mode=args.mode, dry_run=args.dry_run, max_pages=args.max_pages)
    except Exception as error:
        safe_error = f"Pipeline failed during stage '{CURRENT_STAGE}'; inspect the workflow run summary."
        if isinstance(error, RuntimeError):
            if str(error) == "Notices API warm-up/fetch failed before database access":
                safe_error = "Notices API warm-up failed after retries; check NOTICES_API_URL and API availability."
            elif str(error) == "More than 20 percent of notices failed processing":
                safe_error = "More than 20% of fetched notices failed extraction; inspect source records and extraction status."
            elif str(error).startswith("Embedding dimension mismatch:"):
                safe_error = "Embedding dimension mismatch; verify the configured embedding model output."
        summary = {"mode":args.mode,"dry_run":args.dry_run,"failed":1,"stage":CURRENT_STAGE,"error_type":type(error).__name__,"error":safe_error}
        code = 1
        print(safe_error, file=sys.stderr)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)
    if args.summary_file:
        args.summary_file.parent.mkdir(parents=True, exist_ok=True)
        args.summary_file.write_text(payload + "\n", encoding="utf-8")
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary,"a",encoding="utf-8") as out:
            out.write("\n| Mode | Dry run | Fetched | New | Changed | Unchanged | Superseded | Needs review | Failed | Chunks | Embeddings | KB before | KB after | Cache invalidated | Duration (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|\n")
            out.write(f"| {args.mode} | {args.dry_run} | {summary.get('fetched',0)} | {summary.get('new',0)} | {summary.get('changed',0)} | {summary.get('unchanged',0)} | {summary.get('superseded',0)} | {len(summary.get('needs_review',{}))} | {summary.get('failed',0)} | {summary.get('chunks_written',0)} | {summary.get('embeddings_written',0)} | {summary.get('kb_version_before')} | {summary.get('kb_version_after')} | {summary.get('cache_entries_invalidated',0)} | {summary.get('duration_seconds',0)} |\n")
            for notice_id, reasons in summary.get("needs_review",{}).items(): out.write(f"\n- `{notice_id}`: {', '.join(reasons)}\n")
            for warning in summary.get("warnings",[]): out.write(f"\n- Warning: {warning}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
