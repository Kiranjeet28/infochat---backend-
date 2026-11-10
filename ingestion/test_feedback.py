"""Offline tests for POST /api/feedback (fake store, no database)."""

from __future__ import annotations

import os
import re
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from fastapi.testclient import TestClient

import api
from ingestion.limits import SlidingWindowLimiter, env_int

ORIGIN = "http://127.0.0.1:5173"
VALID = {"request_id": "ab12cd34ef56", "rating": "down", "reason": "wrong_answer", "comment": "Date was wrong"}


class FakeStore:
    def __init__(self, error=None):
        self.rows, self.error = [], error

    def add(self, row):
        if self.error:
            raise self.error
        self.rows.append(row)


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        api._feedback_limiter = SlidingWindowLimiter(lambda: [(86400, env_int("FEEDBACK_PER_DAY", 20))])
        self.store = FakeStore()
        stack = ExitStack()
        stack.enter_context(patch.dict(os.environ, {"RATE_LIMIT_SALT": "t", "FEEDBACK_PER_DAY": "20"}))
        stack.enter_context(patch.object(api, "_feedback_store", self.store))
        stack.enter_context(patch.object(api, "initialize_embedding_model", return_value=object()))
        stack.enter_context(patch.object(api, "require_enabled_provider_config", return_value=object()))
        stack.enter_context(patch.object(api, "_answer_chain", None))
        self.client = stack.enter_context(TestClient(api.app, raise_server_exceptions=False))
        self.addCleanup(stack.close)

    def post(self, payload, **kw):
        return self.client.post("/api/feedback", json=payload, headers={"Origin": ORIGIN}, **kw)

    def test_valid_feedback_is_stored_without_question_or_answer(self):
        for payload in (VALID, {"request_id": "ab12", "rating": "up"}):
            response = self.post(payload)
            self.assertEqual((response.status_code, response.json()), (200, {"status": "ok"}))
            self.assertEqual(response.headers["access-control-allow-origin"], ORIGIN)
        self.assertEqual(self.store.rows[0], VALID)
        self.assertEqual(self.store.rows[1], {"request_id": "ab12", "rating": "up", "reason": None, "comment": None})
        for row in self.store.rows:
            self.assertEqual(set(row), {"request_id", "rating", "reason", "comment"})

    def test_strict_validation(self):
        bad = [
            {**VALID, "rating": "meh"}, {**VALID, "reason": "because"}, {**VALID, "request_id": "ABCDEF"},
            {**VALID, "request_id": "g" * 10}, {**VALID, "request_id": "a" * 65}, {**VALID, "request_id": ""},
            {**VALID, "comment": "x" * 201}, {**VALID, "question": "What is the fee?"}, {**VALID, "answer": "text"},
            {"rating": "up"}, {"request_id": "ab12"}, {**VALID, "request_id": 123}, {**VALID, "comment": ["a"]},
        ]
        for payload in bad:
            with self.subTest(payload=str(payload)[:60]):
                response = self.post(payload)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json(), {"status": "error", "message": "Invalid feedback."})
        self.assertEqual(self.client.post("/api/feedback", content=b"{oops", headers={"content-type": "application/json"}).status_code, 422)
        self.assertEqual(self.store.rows, [])
        self.assertEqual(self.post({**VALID, "request_id": "a" * 64, "comment": "y" * 200}).status_code, 200)

    def test_comment_is_cleaned_and_blank_comment_becomes_null(self):
        self.post({**VALID, "comment": "  line\x00one\x07 \n two  "})
        self.post({**VALID, "comment": "   "})
        self.assertEqual(self.store.rows[0]["comment"], "line one  \n two")
        self.assertIsNone(self.store.rows[1]["comment"])

    def test_limit_is_20_per_client_per_day_with_cors(self):
        self.assertEqual([self.post(VALID).status_code for _ in range(20)], [200] * 20)
        blocked = self.post(VALID)
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.json()["status"], "rate_limited")
        self.assertIn("retry-after", blocked.headers)
        self.assertEqual(blocked.headers["access-control-allow-origin"], ORIGIN)
        self.assertEqual(len(self.store.rows), 20)

    def test_store_failure_is_503_without_details(self):
        self.store.error = OSError("db down https://example.test?key=SECRET")
        response = self.post(VALID)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("SECRET", response.text)
        self.assertNotIn("Traceback", response.text)

    def test_nothing_sensitive_is_logged(self):
        with self.assertLogs(api.logger, level="INFO") as logs:
            self.post({**VALID, "comment": "my roll number is 2210123456"})
        text = "\n".join(logs.output)
        self.assertIn("status=feedback", text)
        self.assertNotIn("2210123456", text)
        self.assertNotIn("roll number", text)

    def test_cors_preflight(self):
        def preflight(origin):
            return self.client.options("/api/feedback", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"})
        allowed = preflight(ORIGIN)
        self.assertEqual((allowed.status_code, allowed.headers.get("access-control-allow-origin")), (200, ORIGIN))
        self.assertNotIn("access-control-allow-origin", preflight("http://evil.example").headers)

    def test_ask_validation_message_is_unchanged(self):
        response = self.client.post("/api/ask", json={"q": "x"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("question must contain", response.json()["message"])


class FeedbackSqlTests(unittest.TestCase):
    sql = (Path(__file__).resolve().parent / "feedback_table.sql").read_text(encoding="utf-8")

    def test_table_is_locked_down_and_idempotent(self):
        sql = re.sub(r"--.*", "", self.sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS public.feedback", sql)
        self.assertIn("ENABLE ROW LEVEL SECURITY", sql)
        self.assertRegex(sql, r"REVOKE ALL ON TABLE public\.feedback FROM PUBLIC, anon, authenticated")
        self.assertIn("GRANT ALL ON TABLE public.feedback TO service_role", sql)
        self.assertNotIn("CREATE POLICY", sql.upper())
        self.assertNotIn("DROP ", sql.upper())
        self.assertNotIn("DELETE", sql.upper())
        self.assertNotRegex(sql.lower(), r"\b(question|answer)\b")
        self.assertIn("char_length(comment) <= 200", sql)


if __name__ == "__main__":
    unittest.main()
