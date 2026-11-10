"""Offline tests for the /api/ask rate limit."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

import api

class RateLimitTests(unittest.TestCase):
    def setUp(self):
        api._rate_hits.clear()

    def request(self, host="1.2.3.4", forwarded=None):
        headers = {"x-forwarded-for": forwarded} if forwarded else {}
        return SimpleNamespace(client=SimpleNamespace(host=host), headers=headers)

    def test_limit_blocks_after_n_requests_per_client(self):
        with patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "3"}):
            self.assertEqual([api._rate_limit_retry_after(self.request()) for _ in range(3)], [None] * 3)
            self.assertGreaterEqual(api._rate_limit_retry_after(self.request()), 1)
            self.assertIsNone(api._rate_limit_retry_after(self.request(host="9.9.9.9")))

    def test_zero_disables_and_forwarded_header_is_opt_in(self):
        with patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "0"}):
            self.assertTrue(all(api._rate_limit_retry_after(self.request()) is None for _ in range(50)))
        with patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "1"}):
            # without the opt-in, the header is ignored and both calls share one client
            self.assertIsNone(api._rate_limit_retry_after(self.request(forwarded="5.5.5.5")))
            self.assertIsNotNone(api._rate_limit_retry_after(self.request(forwarded="6.6.6.6")))
        with patch.dict(os.environ, {"RATE_LIMIT_PER_MINUTE": "1", "RATE_LIMIT_TRUST_FORWARDED": "1"}):
            api._rate_hits.clear()
            self.assertIsNone(api._rate_limit_retry_after(self.request(forwarded="5.5.5.5")))
            self.assertIsNone(api._rate_limit_retry_after(self.request(forwarded="6.6.6.6")))
            self.assertIsNotNone(api._rate_limit_retry_after(self.request(forwarded="5.5.5.5")))


if __name__ == "__main__":
    unittest.main()
