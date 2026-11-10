"""Offline tests: one-line stdout logging, optional file logging, security headers, .env.example."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from fastapi.testclient import TestClient

import api

ROOT = Path(__file__).resolve().parent.parent
QUESTION = "What is the very private question text 9f3a?"


class Chain:
    def invoke(self, _values):
        return {"answer": "Answer [1]", "status": "answered", "citations_valid": True, "attempts": 1,
                "provider": "fake", "model": "m",
                "sources": [{"notice_id": "n1", "title": "T", "published_at": "2026-09-02",
                             "source_url": "https://example.test/n1", "content_hash": "h"}]}


class LoggingTests(unittest.TestCase):
    def fresh_logger(self, name):
        target = logging.getLogger(name)
        target.handlers.clear()

        def cleanup():
            for handler in list(target.handlers):
                handler.close()
            target.handlers.clear()

        self.addCleanup(cleanup)
        return target

    def test_stdout_only_by_default_and_file_only_with_log_file(self):
        target = self.fresh_logger("qa.logging.default")
        with patch.dict(os.environ, {"LOG_FILE": ""}):
            api._configure_logging(target)
            api._configure_logging(target)  # idempotent
        kinds = [type(h).__name__ for h in target.handlers]
        self.assertEqual(kinds, ["StreamHandler"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "api.log"
            other = self.fresh_logger("qa.logging.file")
            with patch.dict(os.environ, {"LOG_FILE": str(path)}):
                api._configure_logging(other)
                api._configure_logging(other)
            self.assertEqual(sorted(type(h).__name__ for h in other.handlers), ["RotatingFileHandler", "StreamHandler"])
            other.info("status=test hello")
            for handler in other.handlers:
                handler.flush()
                handler.close()
            self.assertIn("status=test hello", path.read_text(encoding="utf-8"))

    def test_records_are_one_line_even_for_stack_traces(self):
        formatter = api._OneLineFormatter("%(levelname)s %(message)s")
        try:
            raise ValueError("boom")
        except ValueError:
            import sys
            record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed\nline two", None, sys.exc_info())
        text = formatter.format(record)
        self.assertNotIn("\n", text)
        self.assertNotIn("\r", text)
        self.assertIn("Traceback", text)

    def test_api_logs_never_contain_the_question_or_keys(self):
        secret = "gsk_FAKEKEYFORTESTONLY1234567890"
        with patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "0", "RATE_LIMIT_PER_DAY": "0", "GLOBAL_DAILY_MODEL_CALLS": "0",
                                     "GROQ_API_KEY": secret}), \
                patch.object(api, "cache_enabled", return_value=False), \
                patch.object(api, "initialize_embedding_model", return_value=object()), \
                patch.object(api, "require_enabled_provider_config", return_value=object()), \
                patch.object(api, "_answer_chain", Chain()), \
                TestClient(api.app) as client, self.assertLogs(api.logger, level="INFO") as logs:
            client.post("/api/ask", json={"question": QUESTION})
            client.post("/api/ask", json={"question": "Calculate my SGPA"})
        text = "\n".join(logs.output)
        self.assertNotIn("private question", text)
        self.assertNotIn(secret, text)
        self.assertIn("question_length=", text)
        self.assertTrue(all("\n" not in line for line in logs.output))


class SecurityHeaderTests(unittest.TestCase):
    def test_nosniff_on_every_kind_of_response_and_health_is_unchanged(self):
        with patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "1", "RATE_LIMIT_PER_DAY": "0", "GLOBAL_DAILY_MODEL_CALLS": "0"}), \
                patch.object(api, "cache_enabled", return_value=False), \
                patch.object(api, "initialize_embedding_model", return_value=object()), \
                patch.object(api, "require_enabled_provider_config", return_value=object()), \
                patch.object(api, "_answer_chain", Chain()), \
                patch.object(api, "_ask_limiter", api.SlidingWindowLimiter(api._ask_windows)), \
                TestClient(api.app) as client:
            responses = {
                "health": client.get("/health"),
                "ok": client.post("/api/ask", json={"question": "hello"}),
                "limited": client.post("/api/ask", json={"question": "hello"}),
                "invalid": client.post("/api/ask", json={"q": "x"}),
                "missing": client.get("/nope"),
            }
        self.assertEqual(responses["health"].json(), {"status": "ok"})
        self.assertEqual(responses["limited"].status_code, 429)
        for name, response in responses.items():
            with self.subTest(name):
                self.assertEqual(response.headers.get("x-content-type-options"), "nosniff")
                self.assertEqual(response.headers.get("referrer-policy"), "no-referrer")


class EnvExampleTests(unittest.TestCase):
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    names = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, re.M))

    def test_no_secret_values_are_present(self):
        for line in self.text.splitlines():
            match = re.match(r"^#?\s*([A-Z][A-Z0-9_]+)=(.*)$", line)
            if match and re.search(r"KEY|TOKEN|SECRET|SALT|PASSWORD", match.group(1)):
                self.assertEqual(match.group(2).split("#")[0].strip(), "", match.group(1))
        self.assertNotRegex(self.text, r"gsk_[A-Za-z0-9]{10,}|eyJ[A-Za-z0-9_-]{20,}|sb_secret|AIza[0-9A-Za-z_-]{20,}")

    def test_every_setting_read_by_the_code_is_documented(self):
        used = set()
        sources = [ROOT / "api.py"] + [p for p in (ROOT / "ingestion").glob("*.py") if not p.name.startswith("test_")]
        for path in sources:
            code = path.read_text(encoding="utf-8")
            used |= set(re.findall(r"(?:getenv|environ\.get|environ)\(?\[?[\"']([A-Z][A-Z0-9_]+)[\"']", code))
            used |= set(re.findall(r"(?:env_int|_env_bool|_positive_int|_positive_float)\([\"']([A-Z][A-Z0-9_]+)[\"']", code))
        used -= {"GITHUB_STEP_SUMMARY", "PIPELINE_MODE", "PIPELINE_DRY_RUN"}
        self.assertEqual(sorted(used - self.names), [])


if __name__ == "__main__":
    unittest.main()
