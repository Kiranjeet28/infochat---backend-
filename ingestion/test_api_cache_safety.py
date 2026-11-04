"""Offline TestClient checks for API cache fail-open and request diagnostics."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
MISSING = [name for name in ("fastapi", "httpx") if importlib.util.find_spec(name) is None]

if not MISSING:
    from fastapi.testclient import TestClient
    import api


@unittest.skipIf(MISSING, "FastAPI TestClient tests skipped: install fastapi and httpx")
class ApiCacheSafetyTests(unittest.TestCase):
    question = "unique offline cache diagnostic question"

    class Chain:
        def __init__(self, status="answered", fail=False):
            self.status = status
            self.fail = fail

        def invoke(self, _values):
            if self.fail:
                raise RuntimeError("offline test exception")
            return {
                "answer": "A checked answer [1]" if self.status == "answered" else "Unavailable",
                "sources": [{
                    "notice_id": "notice-1", "title": "Safe title",
                    "published_at": "2026-11-04", "source_url": "https://example.test/n1",
                    "content_hash": "hash-1",
                }],
                "status": self.status,
                "citations_valid": self.status == "answered",
                "provider": "fake", "model": "fake-model", "attempts": 1,
            }

    class Store:
        project_url = "https://example.test"
        service_role_key = "fake-secret-never-logged"

        def __init__(self, *, get_error=False, set_error=False, hit=False):
            self.get_error = get_error
            self.set_error = set_error
            self.hit = hit
            self.writes = 0

        def get(self, _key):
            if self.get_error:
                raise OSError("fake cache read error")
            if self.hit:
                return {
                    "cache_key": "fake-key", "answer": "Cached answer [1]", "status": "answered",
                    "sources": [{"notice_id": "notice-1", "title": "Safe title"}],
                    "source_notice_ids": ["notice-1"],
                    "source_content_hashes": {"notice-1": "hash-1"},
                    "kb_version": 1, "prompt_version": api.PROMPT_VERSION,
                    "created_at": "2026-11-04T00:00:00+00:00",
                    "expires_at": "2099-01-01T00:00:00+00:00", "hit_count": 0,
                }
            return None

        def set(self, _entry):
            self.writes += 1
            if self.set_error:
                raise OSError("fake cache write error")

        def delete(self, _key):
            return None

    def request(self, *, status="answered", store=None, fail=False):
        store = store or self.Store()
        old_store = api._cache_store
        api._cache_store = store
        try:
            with (
                patch.dict(os.environ, {"CACHE_ENABLED": "true", "SUPABASE_URL": "https://example.test",
                                        "SUPABASE_SERVICE_ROLE_KEY": "fake-secret-never-logged"}),
                patch.object(api, "cache_enabled", return_value=True),
                patch.object(api, "read_kb_version", return_value="1"),
                patch.object(api, "verify_cached_sources", return_value=True),
                patch.object(api, "initialize_embedding_model", return_value=object()),
                patch.object(api, "require_enabled_provider_config", return_value=object()),
                patch.object(api, "_answer_chain", self.Chain(status=status, fail=fail)),
                TestClient(api.app, raise_server_exceptions=False) as client,
            ):
                response = client.post("/api/ask", json={"question": self.question})
            return response, store
        finally:
            api._cache_store = old_store

    def test_cache_get_error_fails_open(self):
        response, _ = self.request(store=self.Store(get_error=True))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")
        self.assertFalse(response.json()["cached"])

    def test_cache_set_error_fails_open(self):
        response, _ = self.request(store=self.Store(set_error=True))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")

    def test_hit_count_update_error_still_returns_cached_answer(self):
        response, _ = self.request(store=self.Store(set_error=True, hit=True))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "answered")
        self.assertTrue(response.json()["cached"])

    def test_request_id_header_is_returned(self):
        response, _ = self.request()
        self.assertTrue(response.headers.get("X-Request-ID"))

    def test_unexpected_error_has_request_id_and_traceback_only_in_log(self):
        response, _ = self.request(fail=True)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])
        self.assertNotIn("Traceback", response.text)
        log_text = (ROOT / "logs" / "api.log").read_text(encoding="utf-8")
        self.assertIn(response.headers["X-Request-ID"], log_text)
        self.assertIn("Traceback", log_text)

    def test_api_logs_do_not_contain_raw_question(self):
        unique_question = "unique raw question must stay out of logs 271104"
        old_store = api._cache_store
        api._cache_store = self.Store()
        try:
            with (
                patch.dict(os.environ, {"CACHE_ENABLED": "false"}),
                patch.object(api, "_answer_chain", self.Chain()),
                patch.object(api, "require_enabled_provider_config", return_value=object()),
                patch.object(api, "initialize_embedding_model", return_value=object()),
                TestClient(api.app) as client,
            ):
                response = client.post("/api/ask", json={"question": unique_question})
            self.assertEqual(response.status_code, 200)
            log_text = (ROOT / "logs" / "api.log").read_text(encoding="utf-8")
            self.assertNotIn(unique_question, log_text)
        finally:
            api._cache_store = old_store

    def test_invalid_citation_ai_unavailable_and_error_are_never_cached(self):
        for status in ("invalid_citation", "ai_unavailable", "error"):
            with self.subTest(status=status):
                response, store = self.request(status=status)
                self.assertEqual(response.status_code, 500 if status == "error" else 200)
                self.assertEqual(store.writes, 0)


if __name__ == "__main__":
    unittest.main()
