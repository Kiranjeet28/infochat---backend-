"""Offline API response tests using an in-process fake answer chain."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from starlette.requests import Request

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

import api


def _request() -> Request:
    return Request({
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/ask",
        "raw_path": b"/api/ask",
        "query_string": b"",
        "headers": [],
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    })


class FakeChain:
    def __init__(self, result):
        self.result = result

    async def ainvoke(self, _values):
        return self.result


class ApiResponseTests(unittest.TestCase):
    def call_api(self, result, *, plain_text="false"):
        with (
            patch.object(api, "_answer_chain", FakeChain(result)),
            patch.dict(os.environ, {"CASCADEIQ_PLAIN_TEXT_ANSWERS": plain_text}),
        ):
            return api.ask(_request(), api.AskRequest(question="test question"))

    @staticmethod
    def source(title):
        return {"notice_id": title, "title": title, "published_at": "2026-09-02",
                "source_url": f"https://example.test/{title}"}

    def test_answer_returns_only_cited_sources_in_citation_order(self):
        response = self.call_api({
            "answer": "Second [2], then first [1], and second again [2].",
            "sources": [self.source("first"), self.source("second"), self.source("related")],
            "status": "answered",
            "citations_valid": True,
        })
        self.assertEqual([source.title for source in response.sources], ["second", "first"])
        self.assertEqual(response.sources_kind, "cited")

    def test_citations_are_renumbered_to_match_trimmed_sources(self):
        response = self.call_api({
            "answer": "Results online [1]; dates [3].",
            "sources": [self.source("first"), self.source("second"), self.source("third")],
            "status": "answered",
            "citations_valid": True,
        })
        self.assertEqual([source.title for source in response.sources], ["first", "third"])
        self.assertEqual(response.answer, "Results online [1]; dates [2].")

        response = self.call_api({
            "answer": "Second [2], then first [1], and second again [2].",
            "sources": [self.source("first"), self.source("second"), self.source("related")],
            "status": "answered",
            "citations_valid": True,
        })
        self.assertEqual(response.answer, "Second [1], then first [2], and second again [1].")

    def test_renumber_citations_handles_groups_and_ranges(self):
        from ingestion.answer_chain import renumber_citations

        self.assertEqual(renumber_citations("x [2, 4] y", {2: 1, 4: 2}), "x [1, 2] y")
        self.assertEqual(renumber_citations("x [3-5] y", {3: 1, 4: 2, 5: 3}), "x [1-3] y")
        self.assertEqual(renumber_citations("x [3-5] y", {3: 3, 4: 1, 5: 2}), "x [3][1][2] y")
        self.assertEqual(renumber_citations("keep [note] 2026 [x]", {}), "keep [note] 2026 [x]")

    def test_not_found_returns_related_sources(self):
        response = self.call_api({
            "answer": "The available notices do not establish this.",
            "sources": [self.source("related")],
            "status": "not_found",
            "citations_valid": None,
        })
        self.assertEqual([source.title for source in response.sources], ["related"])
        self.assertEqual(response.sources_kind, "related")

    def test_ai_unavailable_and_invalid_citation_keep_related_sources(self):
        for status in ("ai_unavailable", "invalid_citation"):
            with self.subTest(status=status):
                response = self.call_api({
                    "answer": "I could not verify this.",
                    "sources": [self.source("related")],
                    "status": status,
                    "citations_valid": False,
                })
                self.assertEqual([source.title for source in response.sources], ["related"])
                self.assertEqual(response.sources_kind, "related")

    def test_markdown_is_preserved_by_default(self):
        response = self.call_api({
            "answer": "**Bold** [1]",
            "sources": [self.source("notice")],
            "status": "answered",
            "citations_valid": True,
        })
        self.assertEqual(response.answer, "**Bold** [1]")

    def test_configured_plain_text_strips_common_markdown(self):
        response = self.call_api({
            "answer": "# **Important**\n\n- *first*\n- [source](https://example.test/a)\n- `code`\n1. ~~old~~ [1]",
            "sources": [self.source("notice")],
            "status": "answered",
            "citations_valid": True,
        }, plain_text="true")
        self.assertEqual(
            response.answer,
            "Important\n\nfirst\nsource (https://example.test/a)\ncode\nold [1]",
        )


if __name__ == "__main__":
    unittest.main()
