"""Offline tests for per-client limits, the global model-call cap and their API behaviour."""

from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from fastapi.testclient import TestClient

import api
from ingestion import limits
from ingestion.limits import GlobalModelCap, SlidingWindowLimiter, client_id

ORIGIN = "http://127.0.0.1:5173"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class SlidingWindowTests(unittest.TestCase):
    def make(self, per_minute=10, per_day=80):
        clock = Clock()
        return SlidingWindowLimiter(lambda: [(60, per_minute), (86400, per_day)], clock), clock

    def test_per_minute_limit_then_reset_after_the_window(self):
        limiter, clock = self.make()
        self.assertEqual([limiter.hit("a") for _ in range(10)], [None] * 10)
        wait = limiter.hit("a")
        self.assertIsNotNone(wait)
        self.assertTrue(1 <= wait <= 60)
        self.assertIsNone(limiter.hit("b"))  # other clients are unaffected
        clock.now += 61
        self.assertIsNone(limiter.hit("a"))

    def test_per_day_limit_survives_the_minute_window_and_resets_after_a_day(self):
        limiter, clock = self.make(per_minute=10, per_day=80)
        for _ in range(8):  # 8 bursts of 10 in separate minutes = 80 requests
            for _ in range(10):
                self.assertIsNone(limiter.hit("a"))
            clock.now += 61
        wait = limiter.hit("a")
        self.assertIsNotNone(wait)
        self.assertGreater(wait, 60)
        clock.now += 86400
        self.assertIsNone(limiter.hit("a"))

    def test_rejected_requests_are_not_recorded(self):
        limiter, clock = self.make(per_minute=2, per_day=0)
        limiter.hit("a"); limiter.hit("a")
        for _ in range(20):
            self.assertIsNotNone(limiter.hit("a"))
        clock.now += 61
        self.assertIsNone(limiter.hit("a"))  # spam while blocked did not extend the block

    def test_zero_turns_limits_off_and_table_is_bounded(self):
        off = SlidingWindowLimiter(lambda: [(60, 0), (86400, 0)])
        self.assertTrue(all(off.hit("a") is None for _ in range(500)))
        limiter = SlidingWindowLimiter(lambda: [(60, 5)], Clock(), max_clients=50)
        for i in range(200):
            limiter.hit(f"client-{i}")
        self.assertLessEqual(len(limiter._hits), 51)


class ClientIdTests(unittest.TestCase):
    def test_hashed_stable_and_salted(self):
        with patch.dict(os.environ, {"RATE_LIMIT_SALT": "salt-one"}):
            first = client_id("203.0.113.7")
            self.assertEqual(first, client_id("203.0.113.7"))
            self.assertNotIn("203.0.113.7", first)
        with patch.dict(os.environ, {"RATE_LIMIT_SALT": "salt-two"}):
            self.assertNotEqual(first, client_id("203.0.113.7"))

    def test_forwarded_header_only_when_trusted(self):
        with patch.dict(os.environ, {"RATE_LIMIT_SALT": "s"}, clear=False):
            os.environ.pop("TRUST_PROXY_HEADERS", None)
            self.assertEqual(client_id("10.0.0.1", "1.1.1.1, 10.0.0.1"), client_id("10.0.0.1"))
            with patch.dict(os.environ, {"TRUST_PROXY_HEADERS": "1"}):
                self.assertEqual(client_id("10.0.0.1", "1.1.1.1, 10.0.0.1"), client_id("1.1.1.1"))
                self.assertEqual(client_id("10.0.0.1", None), client_id("10.0.0.1"))


class GlobalCapTests(unittest.TestCase):
    def test_cap_refund_retry_adjustment_and_new_day(self):
        day = ["2026-11-10"]
        cap = GlobalModelCap(lambda: 3, lambda: day[0])
        self.assertEqual([cap.reserve() for _ in range(4)], [True, True, True, False])
        cap.adjust(-1)  # a call that never reached a provider is refunded
        self.assertTrue(cap.reserve())
        cap.adjust(2)   # two retries on the same request
        self.assertFalse(cap.reserve())
        day[0] = "2026-11-11"
        self.assertEqual(cap.used, 0)
        self.assertTrue(cap.reserve())

    def test_zero_disables_the_cap(self):
        cap = GlobalModelCap(lambda: 0)
        self.assertTrue(all(cap.reserve() for _ in range(1000)))


class Chain:
    def __init__(self, status="answered", attempts=1):
        self.status, self.attempts, self.calls = status, attempts, 0

    def invoke(self, _values):
        self.calls += 1
        return {"answer": "Answer [1]" if self.status == "answered" else "x",
                "sources": [{"notice_id": "n1", "title": "T", "published_at": "2026-09-02",
                             "source_url": "https://example.test/n1", "content_hash": "h1"}],
                "status": self.status, "citations_valid": self.status == "answered",
                "provider": "fake" if self.attempts else None, "model": "m", "attempts": self.attempts}


class Store:
    project_url, service_role_key = "https://example.test", "fake-secret"

    def __init__(self, hit=False):
        self.hit = hit

    def get(self, _key):
        if not self.hit:
            return None
        return {"cache_key": "k", "answer": "Cached [1]", "status": "answered",
                "sources": [{"notice_id": "n1", "title": "T"}], "source_notice_ids": ["n1"],
                "source_content_hashes": {"n1": "h1"}, "kb_version": 1, "prompt_version": api.PROMPT_VERSION,
                "created_at": "2026-11-04T00:00:00+00:00", "expires_at": "2099-01-01T00:00:00+00:00", "hit_count": 0}

    def set(self, _entry):
        pass

    def delete(self, _key):
        pass


class ApiLimitTests(unittest.TestCase):
    def setUp(self):
        api._ask_limiter = SlidingWindowLimiter(api._ask_windows)
        api._model_cap = GlobalModelCap(lambda: limits.env_int("GLOBAL_DAILY_MODEL_CALLS", 400))

    def client(self, chain, env=None, store=None, cache=False):
        stack = ExitStack()
        environment = {"RATE_LIMIT_PER_MINUTE": "100", "RATE_LIMIT_PER_DAY": "1000", "GLOBAL_DAILY_MODEL_CALLS": "400",
                       "RATE_LIMIT_SALT": "t", "SUPABASE_URL": "https://example.test", "SUPABASE_SERVICE_ROLE_KEY": "fake-secret",
                       **(env or {})}
        stack.enter_context(patch.dict(os.environ, environment))
        stack.enter_context(patch.object(api, "cache_enabled", return_value=cache))
        stack.enter_context(patch.object(api, "read_kb_version", return_value="1"))
        stack.enter_context(patch.object(api, "verify_cached_sources", return_value=True))
        stack.enter_context(patch.object(api, "initialize_embedding_model", return_value=object()))
        stack.enter_context(patch.object(api, "require_enabled_provider_config", return_value=object()))
        stack.enter_context(patch.object(api, "_answer_chain", chain))
        stack.enter_context(patch.object(api, "_cache_store", store))
        client = stack.enter_context(TestClient(api.app, raise_server_exceptions=False))
        self.addCleanup(stack.close)
        return client

    def ask(self, client, question="What is the makeup schedule?", headers=None):
        return client.post("/api/ask", json={"question": question}, headers={"Origin": ORIGIN, **(headers or {})})

    def test_per_minute_limit_returns_429_with_cors_and_retry_after(self):
        client = self.client(Chain(), {"RATE_LIMIT_PER_MINUTE": "3"})
        self.assertEqual([self.ask(client).status_code for _ in range(3)], [200] * 3)
        blocked = self.ask(client)
        self.assertEqual(blocked.status_code, 429)
        body = blocked.json()
        self.assertEqual(body["status"], "rate_limited")
        self.assertGreaterEqual(body["retry_after_seconds"], 1)
        self.assertEqual(blocked.headers["retry-after"], str(body["retry_after_seconds"]))
        self.assertEqual(blocked.headers["access-control-allow-origin"], ORIGIN)
        self.assertTrue(blocked.headers.get("x-request-id"))

    def test_per_day_limit_blocks_even_when_the_minute_is_free(self):
        clock = Clock()
        api._ask_limiter = SlidingWindowLimiter(api._ask_windows, clock)
        client = self.client(Chain(), {"RATE_LIMIT_PER_MINUTE": "100", "RATE_LIMIT_PER_DAY": "4"})
        for _ in range(4):
            self.assertEqual(self.ask(client).status_code, 200)
            clock.now += 120
        self.assertEqual(self.ask(client).status_code, 429)

    def test_proxy_header_changes_the_client_only_when_trusted(self):
        client = self.client(Chain(), {"RATE_LIMIT_PER_MINUTE": "1"})
        self.assertEqual(self.ask(client, headers={"X-Forwarded-For": "1.1.1.1"}).status_code, 200)
        self.assertEqual(self.ask(client, headers={"X-Forwarded-For": "2.2.2.2"}).status_code, 429)
        api._ask_limiter = SlidingWindowLimiter(api._ask_windows)
        client = self.client(Chain(), {"RATE_LIMIT_PER_MINUTE": "1", "TRUST_PROXY_HEADERS": "1"})
        self.assertEqual(self.ask(client, headers={"X-Forwarded-For": "1.1.1.1"}).status_code, 200)
        self.assertEqual(self.ask(client, headers={"X-Forwarded-For": "2.2.2.2"}).status_code, 200)

    def test_global_cap_returns_busy_ai_unavailable_without_calling_the_provider(self):
        chain = Chain()
        client = self.client(chain, {"GLOBAL_DAILY_MODEL_CALLS": "2"})
        self.assertEqual([self.ask(client).json()["status"] for _ in range(2)], ["answered", "answered"])
        retrieval_only = Chain(status="ai_unavailable", attempts=0)
        with patch.object(api, "build_answer_chain", return_value=retrieval_only) as built:
            busy = self.ask(client)
        self.assertEqual(busy.status_code, 200)
        body = busy.json()
        self.assertEqual(body["status"], "ai_unavailable")
        self.assertEqual(body["answer"], api.BUSY_TODAY_MESSAGE)
        self.assertEqual(body["sources_kind"], "related")
        self.assertTrue(body["sources"])
        self.assertEqual(chain.calls, 2)  # the real chain was not used again
        self.assertIsNotNone(built.call_args.kwargs["answer_generator"])
        self.assertEqual(api._model_cap.used, 2)

    def test_busy_generator_never_calls_a_provider(self):
        result = api._busy_generator("prompt")
        self.assertEqual((result["status"], result["attempts"], result["provider"]), ("ai_unavailable", 0, None))

    def test_redirects_cache_hits_validation_and_retrieval_not_found_do_not_count(self):
        chain = Chain()
        client = self.client(chain, {"GLOBAL_DAILY_MODEL_CALLS": "5"}, store=Store(hit=True), cache=True)
        self.assertEqual(self.ask(client, "Calculate my SGPA").json()["status"], "redirect")
        self.assertTrue(self.ask(client).json()["cached"])
        self.assertEqual(client.post("/api/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(api._model_cap.used, 0)
        self.assertEqual(chain.calls, 0)
        client2 = self.client(Chain(status="not_found_retrieval", attempts=0), {"GLOBAL_DAILY_MODEL_CALLS": "5"})
        self.assertEqual(self.ask(client2, "What is the hostel fee?").json()["status"], "not_found")
        self.assertEqual(api._model_cap.used, 0)

    def test_real_model_calls_and_retries_count(self):
        client = self.client(Chain(attempts=3), {"GLOBAL_DAILY_MODEL_CALLS": "10"})
        self.ask(client)
        self.assertEqual(api._model_cap.used, 3)

    def test_chain_failure_refunds_the_reserved_call(self):
        class Boom:
            def invoke(self, _values):
                raise RuntimeError("retrieval down")

        client = self.client(Boom(), {"GLOBAL_DAILY_MODEL_CALLS": "10"})
        self.assertEqual(self.ask(client).status_code, 500)
        self.assertEqual(api._model_cap.used, 0)


if __name__ == "__main__":
    unittest.main()
