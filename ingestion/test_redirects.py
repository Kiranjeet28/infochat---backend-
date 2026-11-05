"""Offline router and redirect API tests."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from ingestion.redirects import (
    CALCULATOR_LINK, REDIRECT_MESSAGE, TIMETABLE_LINK, route_question,
)

ROOT = Path(__file__).resolve().parent.parent
MISSING = [name for name in ("fastapi", "httpx") if importlib.util.find_spec(name) is None]
if not MISSING:
    from fastapi.testclient import TestClient
    import api


class RouterTests(unittest.TestCase):
    def test_english_and_hinglish_router_phrases_and_notice_flow_exclusions(self):
        calculator = [
            "What is my SGPA?", "help with CGPA", "Explain GPA", "grade point", "grade points",
            "calculate percentage", "calculate my percentage", "percentage calculation",
            "percentage nikal", "percentage nikalo", "mera percentage kitna",
            "percentage kitna hua", "marks se percentage", "convert marks to percentage",
            "work out my percentage", "percentage kaise nikalna",
        ]
        timetable = [
            "show timetable", "show time table", "my class", "what is my next class",
            "class at 10:30 AM", "which class is next", "kaun si class hai", "class kab hai",
            "lecture today", "mera timetable",
        ]
        both = ["My SGPA and timetable", "percentage nikal and next class"]
        notice_flow = [
            "makeup exam schedule", "date sheet", "exam schedule",
            "what is the minimum attendance percentage", "provisional result",
            "result declared", "revaluation result", "When is the exam centre merger?",
        ]
        for phrase in calculator:
            with self.subTest(phrase=phrase):
                self.assertEqual(route_question(phrase)[0], "calculator")
        for phrase in timetable:
            with self.subTest(phrase=phrase):
                self.assertEqual(route_question(phrase)[0], "timetable")
        for phrase in both:
            with self.subTest(phrase=phrase):
                self.assertEqual(route_question(phrase)[0], "calculator+timetable")
        for phrase in notice_flow:
            with self.subTest(phrase=phrase):
                self.assertIsNone(route_question(phrase))
        self.assertIsNone(route_question("Where is the library?"))


@unittest.skipIf(MISSING, "FastAPI TestClient tests skipped: install fastapi and httpx")
class RedirectApiTests(unittest.TestCase):
    class FakeChain:
        def __init__(self, answer):
            self.answer = answer
            self.calls = 0

        def invoke(self, _values):
            self.calls += 1
            return {"answer": self.answer, "sources": [], "status": "redirect", "citations_valid": True}

    def setUp(self):
        self.old_chain = api._answer_chain
        self.old_store = api._cache_store

    def tearDown(self):
        api._answer_chain = self.old_chain
        api._cache_store = self.old_store

    def test_rule_redirect_returns_message_and_link_without_cache_or_chain(self):
        raw_question = "unique private question calculate my SGPA 91723"
        chain = self.FakeChain("should never run")
        with (
            patch.dict(os.environ, {"CACHE_ENABLED": "true"}),
            patch.object(api, "cache_enabled", side_effect=AssertionError("cache was consulted")),
            patch.object(api, "_answer_chain", chain),
            patch.object(api, "build_answer_chain", side_effect=AssertionError("chain was built")),
            patch.object(api, "initialize_embedding_model", return_value=object()),
            patch.object(api, "require_enabled_provider_config", return_value=object()),
            patch.object(api, "_append_audit_event") as audit,
            TestClient(api.app) as client,
        ):
            response = client.post("/api/ask", json={"question": raw_question})
        payload = response.json()
        self.assertEqual(payload["status"], "redirect")
        self.assertEqual(payload["answer"], REDIRECT_MESSAGE + "\n" + CALCULATOR_LINK.url)
        self.assertEqual(payload["links"], [{"label": CALCULATOR_LINK.label, "url": CALCULATOR_LINK.url}])
        self.assertEqual(payload["sources"], [])
        self.assertFalse(payload["cached"])
        self.assertEqual(chain.calls, 0)
        audit.assert_called_once()
        self.assertNotIn(raw_question, str(audit.call_args))

    def test_model_marker_becomes_redirect_and_is_never_cached(self):
        chain = self.FakeChain("REDIRECT_CALCULATOR")
        with (
            patch.dict(os.environ, {"CACHE_ENABLED": "false"}),
            patch.object(api, "cache_enabled", return_value=False),
            patch.object(api, "_answer_chain", chain),
            patch.object(api, "initialize_embedding_model", return_value=object()),
            patch.object(api, "require_enabled_provider_config", return_value=object()),
            patch.object(api, "_append_audit_event") as audit,
            TestClient(api.app) as client,
        ):
            response = client.post("/api/ask", json={"question": "What is announced?"})
        self.assertEqual(response.json()["status"], "redirect")
        self.assertEqual(response.json()["links"][0]["url"], CALCULATOR_LINK.url)
        self.assertEqual(chain.calls, 1)
        audit.assert_called_once()

    def test_redirect_logs_and_audit_record_omit_raw_question(self):
        unique_question = "private phrase mera timetable 48291"
        with (
            patch.dict(os.environ, {"CACHE_ENABLED": "false"}),
            patch.object(api, "cache_enabled", side_effect=AssertionError("cache was consulted")),
            patch.object(api, "_answer_chain", self.FakeChain("no")),
            patch.object(api, "initialize_embedding_model", return_value=object()),
            patch.object(api, "require_enabled_provider_config", return_value=object()),
            TestClient(api.app) as client,
        ):
            response = client.post("/api/ask", json={"question": unique_question})
        self.assertEqual(response.status_code, 200)
        api_log = (ROOT / "logs" / "api.log").read_text(encoding="utf-8")
        audit_log = (ROOT / "logs" / "audit.log").read_text(encoding="utf-8")
        self.assertNotIn(unique_question, api_log)
        self.assertNotIn(unique_question, audit_log)


if __name__ == "__main__":
    unittest.main()
