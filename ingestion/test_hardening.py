"""Offline tests for safe fetching of notice links and the /api/ask rate limit."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

import api
from ingestion import extract_text
from ingestion.extract_text import UnsafeFetchError, safe_get

PUBLIC = [(2, 1, 6, "", ("93.184.216.34", 0))]


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b"%PDF-1.4 data", url="https://gndec.ac.in/a.pdf"):
        self.status_code, self.headers, self._body, self.url = status, headers or {}, body, url
        self._content = False
        self.closed = False

    @property
    def content(self):
        return self._content

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        for i in range(0, len(self._body), size):
            yield self._body[i:i + size]

    def close(self):
        self.closed = True


def fetch(response_or_list, url="https://gndec.ac.in/a.pdf", addresses=PUBLIC, **kwargs):
    responses = response_or_list if isinstance(response_or_list, list) else [response_or_list]
    kwargs.setdefault("accept", extract_text.PDF_CONTENT_TYPES)
    kwargs.setdefault("max_bytes", 1024)
    with patch.object(extract_text.socket, "getaddrinfo", return_value=addresses), \
            patch.object(extract_text.requests, "get", side_effect=responses) as get:
        return safe_get(url, **kwargs), get


class SafeGetTests(unittest.TestCase):
    def test_valid_pdf_is_returned_with_body(self):
        response, get = fetch(FakeResponse(headers={"content-type": "application/pdf"}), magic=b"%PDF")
        self.assertEqual(response.content, b"%PDF-1.4 data")
        self.assertIs(get.call_args.kwargs["allow_redirects"], False)

    def test_http_scheme_and_private_hosts_are_blocked(self):
        with self.assertRaises(UnsafeFetchError):
            fetch(FakeResponse(), url="http://gndec.ac.in/a.pdf")
        for ip in ("127.0.0.1", "10.0.0.5", "192.168.1.9", "169.254.169.254", "::1"):
            with self.subTest(ip=ip), self.assertRaises(UnsafeFetchError):
                fetch(FakeResponse(), addresses=[(2, 1, 6, "", (ip, 0))])

    def test_redirect_to_private_host_is_blocked_and_hops_are_capped(self):
        redirect = FakeResponse(302, {"location": "https://internal.example/x"})
        calls = iter([PUBLIC, [(2, 1, 6, "", ("10.0.0.1", 0))]])
        with patch.object(extract_text.socket, "getaddrinfo", side_effect=lambda *a: next(calls)), \
                patch.object(extract_text.requests, "get", side_effect=[redirect]), \
                self.assertRaises(UnsafeFetchError):
            safe_get("https://gndec.ac.in/a.pdf", accept=("application/pdf",), max_bytes=1024)
        loop = [FakeResponse(302, {"location": "https://gndec.ac.in/b"}) for _ in range(10)]
        with self.assertRaisesRegex(UnsafeFetchError, "too many redirects"):
            fetch(loop)

    def test_content_type_size_and_magic_are_enforced(self):
        with self.assertRaisesRegex(UnsafeFetchError, "content type"):
            fetch(FakeResponse(headers={"content-type": "text/html"}))
        with self.assertRaisesRegex(UnsafeFetchError, "too large"):
            fetch(FakeResponse(headers={"content-length": "999999"}))
        with self.assertRaisesRegex(UnsafeFetchError, "too large"):
            fetch(FakeResponse(body=b"%PDF" + b"x" * 5000))
        with self.assertRaisesRegex(UnsafeFetchError, "file type"):
            fetch(FakeResponse(body=b"<html>not a pdf</html>"), magic=b"%PDF")

    def test_non_pdf_notice_link_becomes_a_recorded_failure_not_a_crash(self):
        notice = {"_id": "x", "title": "T", "url": "https://gndec.ac.in/a.pdf", "htmlContent": "", "date": "January 5, 2026"}
        with patch.object(extract_text.socket, "getaddrinfo", return_value=PUBLIC), \
                patch.object(extract_text.requests, "get", return_value=FakeResponse(body=b"<html></html>")):
            result = extract_text.process_notices([notice])[0]
        self.assertEqual(result["ingestion_status"], "failed")
        self.assertEqual(result["extraction_status"], "pdf_error")


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
