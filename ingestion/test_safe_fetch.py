"""Offline tests for safe_get, the official-page fallback rule, PDF caps and the warm-up deadline."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import requests

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from ingestion import extract_text, run_pipeline, safe_fetch
from ingestion.safe_fetch import (BlockedUrlError, FetchDeadlineError, FetchError, FetchTooLargeError,
                                  TooManyRedirectsError, UnsupportedContentTypeError, safe_get)

PUBLIC = [(2, 1, 6, "", ("93.184.216.34", 0))]
PDF = ("application/pdf",)


def addr(ip):
    return [(2, 1, 6, "", (ip, 0))]


class Resp:
    def __init__(self, status=200, headers=None, body=b"%PDF-1.4 ok", url="https://gndec.ac.in/a.pdf", chunks=None):
        self.status_code, self.url = status, url
        self.headers = {"content-type": "application/pdf"} if headers is None else headers
        self._chunks = chunks if chunks is not None else [body]
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError("boom https://gndec.ac.in/a.pdf?token=SECRET123")

    def iter_content(self, _size):
        yield from self._chunks

    def close(self):
        self.closed = True

    @property
    def content(self):
        return self._content

    @property
    def text(self):
        return self._content.decode("utf-8", "replace")


def redirect(location, status=302):
    return Resp(status, {"location": location})


def fetch(responses, url="https://gndec.ac.in/a.pdf", resolver=PUBLIC, **kw):
    kw.setdefault("allowed_content_types", PDF)
    kw.setdefault("max_bytes", 1024)
    seq = responses if isinstance(responses, list) else [responses]
    with patch.object(safe_fetch, "_getaddrinfo", return_value=resolver) as dns, \
            patch.object(safe_fetch.requests, "get", side_effect=seq) as get:
        return safe_get(url, **kw), get, dns


class SafeGetTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"SAFE_FETCH_ALLOWED_HOSTS": "gndec.ac.in"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_valid_fetch_returns_capped_body_and_never_follows_redirects_itself(self):
        resp, get, _ = fetch(Resp())
        self.assertEqual(resp.content, b"%PDF-1.4 ok")
        self.assertIs(get.call_args.kwargs["allow_redirects"], False)

    def test_url_rules(self):
        cases = {
            "http scheme": "http://gndec.ac.in/a.pdf",
            "userinfo": "https://user:pw@gndec.ac.in/a.pdf",
            "other host": "https://example.com/a.pdf",
            "suffix attack": "https://gndec.ac.in.evil.com/a.pdf",
            "prefix attack": "https://evilgndec.ac.in/a.pdf",
            "odd port": "https://gndec.ac.in:8443/a.pdf",
            "no host": "https:///a.pdf",
        }
        for name, url in cases.items():
            with self.subTest(name), self.assertRaises(BlockedUrlError):
                fetch(Resp(), url=url)

    def test_subdomain_and_notices_api_host_are_allowed(self):
        fetch(Resp(), url="https://erp.gndec.ac.in/x.pdf")
        with patch.dict(os.environ, {"NOTICES_API_URL": "https://api.example.test/notices"}):
            fetch(Resp(), url="https://api.example.test/notices")
        fetch(Resp(), url="https://other.test/x", extra_allowed_hosts=("other.test",))

    def test_private_and_special_addresses_are_refused(self):
        for ip in ("127.0.0.1", "10.0.0.5", "172.16.0.1", "192.168.1.1", "169.254.169.254", "0.0.0.0",
                   "224.0.0.1", "100.64.0.1", "::1", "fe80::1", "fc00::1", "ff02::1", "::ffff:10.0.0.1"):
            with self.subTest(ip=ip), self.assertRaises(BlockedUrlError):
                fetch(Resp(), resolver=addr(ip))

    def test_one_private_answer_among_public_ones_is_refused(self):
        with self.assertRaises(BlockedUrlError):
            fetch(Resp(), resolver=addr("93.184.216.34") + addr("10.0.0.1"))

    def test_unresolvable_host_is_refused(self):
        with patch.object(safe_fetch, "_getaddrinfo", side_effect=OSError), self.assertRaises(BlockedUrlError):
            safe_get("https://gndec.ac.in/a.pdf", allowed_content_types=PDF, max_bytes=10)

    def test_redirect_is_rechecked_for_host_ip_and_scheme(self):
        with self.assertRaises(BlockedUrlError):
            fetch([redirect("https://evil.example/x")])
        with self.assertRaises(BlockedUrlError):  # https -> http
            fetch([redirect("http://gndec.ac.in/b.pdf")])
        calls = iter([PUBLIC, addr("10.0.0.9")])
        with patch.object(safe_fetch, "_getaddrinfo", side_effect=lambda *a: next(calls)), \
                patch.object(safe_fetch.requests, "get", side_effect=[redirect("https://erp.gndec.ac.in/b.pdf")]), \
                self.assertRaises(BlockedUrlError):
            safe_get("https://gndec.ac.in/a.pdf", allowed_content_types=PDF, max_bytes=10)

    def test_relative_redirect_followed_and_hops_capped_at_five(self):
        resp, get, _ = fetch([redirect("/b.pdf"), Resp()])
        self.assertEqual(get.call_count, 2)
        loop = [redirect("https://gndec.ac.in/loop") for _ in range(10)]
        with self.assertRaises(TooManyRedirectsError):
            fetch(loop)

    def test_size_cap_counts_streamed_and_declared_bytes(self):
        with self.assertRaises(FetchTooLargeError):
            fetch(Resp(headers={"content-type": "application/pdf", "content-length": "999999"}))
        with self.assertRaises(FetchTooLargeError):
            fetch(Resp(chunks=[b"x" * 600, b"x" * 600]))

    def test_content_type_allowlist(self):
        for headers in ({"content-type": "text/plain"}, {"content-type": "application/octet-stream"}, {}):
            with self.subTest(headers=headers), self.assertRaises(UnsupportedContentTypeError):
                fetch(Resp(headers=headers))
        fetch(Resp(headers={"content-type": "Application/PDF; charset=binary"}))

    def test_deadline_stops_a_slow_stream_and_shrinks_timeouts(self):
        now = [0.0]

        def slow_stream():
            for part in (b"a", b"b", b"c"):
                now[0] += 5
                yield part

        resp = Resp()
        resp.iter_content = lambda _size: slow_stream()
        with patch.object(safe_fetch.time, "monotonic", side_effect=lambda: now[0]), self.assertRaises(FetchDeadlineError):
            fetch(resp, deadline_seconds=8)
        _, get, _ = fetch(Resp(), timeout=(10, 60), deadline_seconds=3)
        self.assertLessEqual(get.call_args.kwargs["timeout"][0], 3)

    def test_error_messages_hold_no_url_query_or_secret(self):
        secret_url = "https://gndec.ac.in/a.pdf?token=SECRET123"
        errors = []
        for case in (lambda: fetch(Resp(status=500), url=secret_url),
                     lambda: fetch(Resp(), url="http://gndec.ac.in/a.pdf?token=SECRET123"),
                     lambda: fetch(Resp(headers={"content-type": "text/html"}), url=secret_url)):
            try:
                case()
            except FetchError as error:
                errors.append(str(error))
        with patch.object(safe_fetch, "_getaddrinfo", return_value=PUBLIC), \
                patch.object(safe_fetch.requests, "get", side_effect=requests.ConnectionError("failed " + secret_url)):
            try:
                safe_get(secret_url, allowed_content_types=PDF, max_bytes=10)
            except FetchError as error:
                errors.append(str(error))
        self.assertEqual(len(errors), 4)
        for message in errors:
            self.assertNotIn("SECRET123", message)
            self.assertNotIn("?", message)


class ExtractionRuleTests(unittest.TestCase):
    notice = {"_id": "n1", "title": "Teaser notice", "url": "https://erp.gndec.ac.in/noticeboard/x",
              "htmlContent": "<p>Check the ERP noticeboard for full details.</p>", "date": "September 2, 2026"}

    def setUp(self):
        for patcher in (patch.object(safe_fetch, "_getaddrinfo", return_value=PUBLIC),
                        patch.dict(os.environ, {"SAFE_FETCH_ALLOWED_HOSTS": "gndec.ac.in"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_unreachable_official_page_fails_the_notice_instead_of_using_the_teaser(self):
        with patch.object(safe_fetch.requests, "get", side_effect=requests.ConnectTimeout("t")):
            result = extract_text.process_notices([dict(self.notice), dict(self.notice, _id="n2", title="Other")])
        self.assertEqual([r["ingestion_status"] for r in result], ["failed", "failed"])
        self.assertEqual(result[0]["extraction_status"], "official_page_unreachable")
        self.assertEqual(result[0]["text"], "")

    def test_page_without_notice_body_also_fails(self):
        page = Resp(headers={"content-type": "text/html"}, body=b"<html><body>app shell</body></html>")
        with patch.object(safe_fetch.requests, "get", return_value=page):
            result = extract_text.process_notices([dict(self.notice)])
        self.assertEqual(result[0]["extraction_status"], "official_page_unreachable")

    def test_official_page_text_is_used_when_readable(self):
        html = b"<html><div class='ql-editor'><p>Exam on 5 January 2027 at the main hall.</p></div></html>"
        page = Resp(headers={"content-type": "text/html"}, body=html)
        with patch.object(safe_fetch.requests, "get", return_value=page):
            result = extract_text.process_notices([dict(self.notice)])
        self.assertEqual(result[0]["ingestion_status"], "success")
        self.assertIn("Exam on 5 January 2027", result[0]["text"])

    def test_pdf_page_and_text_caps(self):
        class Page:
            def __init__(self, number):
                self.number = number

            def extract_text(self):
                return f"w{self.number} " * 20000

        class FakePdf:
            pages = [Page(i) for i in range(150)]

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with patch.object(safe_fetch.requests, "get", return_value=Resp()), \
                patch.object(extract_text.pdfplumber, "open", return_value=FakePdf()):
            text, status = extract_text._extract_pdf("https://gndec.ac.in/a.pdf")
        self.assertEqual(status, "extracted")
        self.assertEqual(len(text), extract_text.MAX_TEXT_CHARS)
        self.assertNotIn("w100 ", text)  # pages beyond the 100-page cap are never read

    def test_non_pdf_file_is_rejected(self):
        with patch.object(safe_fetch.requests, "get", return_value=Resp(body=b"<html></html>")), self.assertRaises(FetchError):
            extract_text._extract_pdf("https://gndec.ac.in/a.pdf")


class ApiOnlyTests(unittest.TestCase):
    """Default configuration: notices come only from the notices API host; no college host is contacted."""

    API = "https://infocascade-database.onrender.com/api/notices"
    teaser = {"_id": "n1", "title": "Teaser notice", "url": "https://erp.gndec.ac.in/noticeboard/x",
              "htmlContent": "<p>Check the ERP noticeboard for full details.</p>", "date": "September 2, 2026"}

    def setUp(self):
        for name in ("SAFE_FETCH_ALLOWED_HOSTS", "NOTICES_API_URL"):
            os.environ.pop(name, None)
        patcher = patch.dict(os.environ, {"NOTICES_API_URL": self.API})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_college_hosts_are_refused_without_any_connection_or_dns_lookup(self):
        for url in ("https://erp.gndec.ac.in/noticeboard/x", "https://gndec.ac.in/a.pdf", "https://example.com/a.pdf"):
            with self.subTest(url=url), patch.object(safe_fetch, "_getaddrinfo") as dns,                     patch.object(safe_fetch.requests, "get") as get, self.assertRaises(safe_fetch.HostNotAllowedError):
                safe_get(url, allowed_content_types=PDF, max_bytes=10)
            dns.assert_not_called()
            get.assert_not_called()

    def test_only_the_notices_api_host_is_allowed_by_default(self):
        self.assertEqual(safe_fetch.allowed_hosts(), ["infocascade-database.onrender.com"])
        with patch.object(safe_fetch, "_getaddrinfo", return_value=PUBLIC),                 patch.object(safe_fetch.requests, "get", return_value=Resp(headers={"content-type": "application/json"}, body=b"{}")):
            safe_get(self.API, allowed_content_types=("application/json",), max_bytes=10)

    def test_the_college_host_can_be_allowed_again_with_one_variable(self):
        with patch.dict(os.environ, {"SAFE_FETCH_ALLOWED_HOSTS": "gndec.ac.in"}):
            self.assertEqual(safe_fetch.allowed_hosts(), ["gndec.ac.in", "infocascade-database.onrender.com"])

    def test_teaser_notice_fails_with_a_clear_status_and_never_touches_the_network(self):
        with patch.object(safe_fetch, "_getaddrinfo") as dns, patch.object(safe_fetch.requests, "get") as get:
            result = extract_text.process_notices([dict(self.teaser)])[0]
        dns.assert_not_called()
        get.assert_not_called()
        self.assertEqual((result["ingestion_status"], result["extraction_status"], result["text"]),
                         ("failed", "official_page_not_allowed", ""))
        self.assertIn("SAFE_FETCH_ALLOWED_HOSTS", result["ingestion_failure_reason"])

    def test_pdf_links_on_blocked_hosts_are_failures_too(self):
        notice = dict(self.teaser, htmlContent='<a href="https://erp.gndec.ac.in/files/a.pdf">pdf</a>')
        with patch.object(safe_fetch.requests, "get") as get:
            result = extract_text.process_notices([notice])[0]
        get.assert_not_called()
        self.assertEqual(result["ingestion_status"], "failed")
        self.assertEqual(result["extraction_status"], "pdf_error")

    def test_pipeline_explains_the_blocked_host_failure_and_writes_nothing(self):
        from ingestion.test_run_pipeline import FakeAPI, FakeStore

        def blocked(items):
            return [dict(n, ingestion_status="failed", extraction_status="official_page_not_allowed") for n in items]

        api = FakeAPI({1: [dict(self.teaser)]})
        store = FakeStore()
        with patch.object(run_pipeline, "warm_up"), patch.object(safe_fetch, "_getaddrinfo", return_value=PUBLIC), \
                self.assertRaisesRegex(RuntimeError, "only on blocked hosts"):
            run_pipeline.run(api_url="https://api.example.test", store=store, session=api, processor=blocked)
        self.assertEqual(store.writes, [])

        with patch.dict(os.environ, {"SUPABASE_URL": "https://db.example.test", "SUPABASE_SERVICE_ROLE_KEY": "fake"}),                 patch.object(run_pipeline, "run", side_effect=RuntimeError("Notice text is only on blocked hosts")),                 patch("sys.stderr") as stderr:
            code = run_pipeline.main(["--dry-run"])
        self.assertEqual(code, 1)
        printed = "".join(call.args[0] for call in stderr.write.call_args_list if call.args)
        self.assertIn("SAFE_FETCH_ALLOWED_HOSTS=gndec.ac.in", printed)


class WarmUpTests(unittest.TestCase):
    def test_unreachable_api_stops_within_the_deadline_and_raises(self):
        class Session:
            calls = 0

            def get(self, *args, **kwargs):
                Session.calls += 1
                raise requests.ConnectTimeout("t")

        now = [0.0]
        with patch.dict(os.environ, {"WARMUP_DEADLINE_SECONDS": "30"}), \
                patch.object(safe_fetch, "_getaddrinfo", return_value=PUBLIC), \
                patch.object(run_pipeline.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(run_pipeline.time, "sleep", side_effect=lambda s: now.__setitem__(0, now[0] + s)):
            with self.assertRaisesRegex(RuntimeError, "warm-up"):
                run_pipeline.warm_up(Session(), "https://api.example.test/notices")
        self.assertLessEqual(Session.calls, run_pipeline.WARMUP_ATTEMPTS)
        self.assertLessEqual(now[0], 30)


if __name__ == "__main__":
    unittest.main()
