"""Offline tests: chunk/embedding backfill limited to named notice ids, and its dry run."""

from __future__ import annotations

import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from ingestion import chunk_notices


class FakeSplitter:
    def split_text(self, text):
        return [text]


class Response:
    def __init__(self, rows):
        self.rows = rows

    def raise_for_status(self):
        pass

    def json(self):
        return self.rows


class FakeSession:
    posts: list = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, _url, *, json, **kwargs):
        FakeSession.posts.append(json)
        return Response(json)


def notice(notice_id, review="ok", superseded=None, text="Some notice text"):
    return {"notice_id": notice_id, "title": f"Notice {notice_id}", "published_at": "2026-09-02",
            "source_url": f"https://erp.gndec.ac.in/{notice_id}", "category": None, "department": None, "semester": None,
            "text": text, "content_hash": f"hash-{notice_id}", "review_status": review, "superseded_by": superseded,
            "index_version": None}


NOTICES = [notice("A"), notice("B"), notice("C", review="needs_review"), notice("D", superseded="A")]


class BackfillTests(unittest.TestCase):
    def run_split(self, only_ids=None, dry_run=False, existing=None):
        FakeSession.posts = []
        calls = {"status": [], "deleted": [], "invalidated": [], "embedded": 0}

        def embed(rows):
            calls["embedded"] += len(rows)
            for row in rows:
                row["embedding"], row["embedding_status"] = [0.0] * 384, "succeeded"
            return len(rows), 0

        with ExitStack() as stack:
            stack.enter_context(patch.object(chunk_notices, "_credentials", return_value=("https://db.example.test", {})))
            stack.enter_context(patch.object(chunk_notices, "_fetch_notices", return_value=[dict(n) for n in NOTICES]))
            stack.enter_context(patch.object(chunk_notices, "_fetch_existing_chunks", return_value=existing or {}))
            stack.enter_context(patch.object(chunk_notices.requests, "Session", FakeSession))
            stack.enter_context(patch.object(chunk_notices.RecursiveCharacterTextSplitter, "from_tiktoken_encoder", return_value=FakeSplitter()))
            stack.enter_context(patch.object(chunk_notices, "_embed_rows", side_effect=embed))
            stack.enter_context(patch.object(chunk_notices, "_record_index_status",
                                             side_effect=lambda _s, _u, _h, nid, status, reason=None: calls["status"].append((nid, status))))
            stack.enter_context(patch.object(chunk_notices, "_delete_notice_chunks",
                                             side_effect=lambda _s, _u, _h, ids: calls["deleted"].append(set(ids)) or {}))
            stack.enter_context(patch.object(chunk_notices, "invalidate_notice_cache", side_effect=lambda ids: calls["invalidated"].append(set(ids))))
            out = StringIO()
            with redirect_stdout(out):
                result = chunk_notices.split_and_load(only_ids, dry_run)
        return result, calls, out.getvalue()

    def test_only_named_notices_are_chunked_embedded_and_marked(self):
        (eligible, saved, failed), calls, _ = self.run_split(only_ids={"A"})
        self.assertEqual((eligible, saved, failed), (1, 1, 0))
        self.assertEqual({row["notice_id"] for rows in FakeSession.posts for row in rows}, {"A"})
        self.assertEqual(calls["status"], [("A", "indexed")])  # B, C, D never touched
        self.assertEqual(calls["invalidated"], [{"A"}])
        self.assertEqual(calls["embedded"], 1)

    def test_other_notices_chunks_are_not_deleted_in_backfill_mode(self):
        existing = {"Z": [{"notice_id": "Z", "chunk_index": 0, "content_hash": "x", "embedding": [0.1]}]}
        _, calls, _ = self.run_split(only_ids={"A"}, existing=existing)
        self.assertTrue(all("Z" not in ids for ids in calls["deleted"]))

    def test_without_ids_every_eligible_notice_is_processed_as_before(self):
        (eligible, saved, _), calls, _ = self.run_split()
        self.assertEqual((eligible, saved), (2, 2))
        self.assertIn(("C", "skipped"), calls["status"])
        self.assertIn(("D", "skipped"), calls["status"])

    def test_dry_run_writes_nothing(self):
        (eligible, saved, _), calls, output = self.run_split(only_ids={"A", "C"}, dry_run=True)
        self.assertEqual((eligible, saved), (1, 0))
        self.assertEqual((FakeSession.posts, calls["status"], calls["deleted"], calls["invalidated"], calls["embedded"]), ([], [], [], [], 0))
        self.assertIn("A: would write 1 chunk", output)
        self.assertIn("C: not eligible (Notice requires review)", output)

    def test_current_chunks_are_left_alone_and_a_dry_run_records_no_status(self):
        current = {"A": [{"notice_id": "A", "chunk_index": 0, "content_hash": "hash-A", "embedding": [0.1] * 384}]}
        _, calls, _ = self.run_split(only_ids={"A"}, existing=current, dry_run=True)
        self.assertEqual((FakeSession.posts, calls["status"]), ([], []))


if __name__ == "__main__":
    unittest.main()
