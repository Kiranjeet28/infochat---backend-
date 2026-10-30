"""Offline unit tests for normalization, citations, retrieval filters, and chain routing."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from ingestion.answer_chain import _validate_answer, build_answer_chain, normalize_answer
from ingestion.evaluate_chain import RESULT_COLUMNS, cases
from ingestion.load_supabase import _notice_row
from ingestion.retrieve import retrieve
from ingestion.retrieval_config import SIMILARITY_DISTANCE_CUTOFF


class FakeResponse:
    status_code = 200

    def __init__(self, rows):
        self.rows = rows

    def json(self):
        return self.rows


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def post(self, *_args, **_kwargs):
        return next(self.responses)


class NormalizationTests(unittest.TestCase):
    def test_narrow_no_break_spaces_between_words_and_in_date(self):
        self.assertEqual(
            normalize_answer("The\u202fpublished on 2\u202fSeptember\u202f2026"),
            "The published on 2 September 2026",
        )

    def test_unicode_space_separators_become_ascii_spaces(self):
        source = "a\u00a0b\u2009c\u2002d\u2003e\u3000f"
        self.assertEqual(normalize_answer(source), "a b c d e f")

    def test_dash_variants_become_hyphens(self):
        self.assertEqual(normalize_answer("A\u2011B C\u2013D E\u2014F G\u2212H"), "A-B C-D E-F G-H")

    def test_zero_width_characters_are_deleted_and_joined_words_repaired(self):
        source = "The\u200bpublished Makeup\u200cExam 2\u200dSeptember 2026\ufeff"
        self.assertEqual(normalize_answer(source), "The published Makeup Exam 2September 2026")

    def test_normal_text_is_unchanged(self):
        text = "Regular answer text with [1] and 2 September 2026."
        self.assertEqual(normalize_answer(text), text)

    def test_repeated_and_line_trailing_spaces_are_cleaned(self):
        self.assertEqual(normalize_answer("two  spaces  \nnext line   \n"), "two spaces\nnext line")


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.document = Document(page_content="Notice text", metadata={"title": "Notice A"})

    def validate(self, answer):
        return _validate_answer({"documents": [self.document, self.document], "answer": answer})

    def test_single_and_grouped_citations_are_valid(self):
        self.assertTrue(self.validate("Fact [1]")["citations_valid"])
        self.assertTrue(self.validate("Fact [1,2]")["citations_valid"])
        self.assertTrue(self.validate("Facts [1][2]")["citations_valid"])

    def test_out_of_range_citation_is_invalid(self):
        result = self.validate("Fact [9]")
        self.assertEqual(result["status"], "invalid_citation")
        self.assertEqual(result["answer"], "not found")
        self.assertEqual(len(result["sources"]), 2)

    def test_no_citation_is_invalid(self):
        self.assertEqual(self.validate("Fact without a citation")["status"], "invalid_citation")

    def test_citation_inside_not_found_answer_is_still_checked(self):
        result = self.validate("not found; the available notices do not establish this [1]")
        self.assertEqual(result["status"], "not_found")
        self.assertTrue(result["citations_valid"])

    def test_out_of_range_citation_inside_not_found_is_invalid(self):
        result = self.validate("not found [9]")
        self.assertEqual(result["status"], "invalid_citation")
        self.assertFalse(result["citations_valid"])


class RetrievalAndRoutingTests(unittest.TestCase):
    def test_full_evaluation_has_12_cases_and_required_columns(self):
        self.assertEqual(len(cases()), 12)
        self.assertEqual(
            RESULT_COLUMNS,
            (
                "case", "question", "expected_status", "actual_status", "stage", "answer",
                "cited_titles", "citations_valid", "best_distance", "latency", "error", "pass/fail",
            ),
        )

    def test_cutoff_applies_per_candidate_and_excludes_unmeasured_keyword_rows(self):
        similarity = [
            {"chunk_id": "a", "chunk_text": "A", "score": 0.20},
            {"chunk_id": "b", "chunk_text": "B", "score": 0.66},
            {"chunk_id": "c", "chunk_text": "C", "score": 0.64},
        ]
        keywords = [
            {"chunk_id": "a", "chunk_text": "A", "score": 0.9},
            {"chunk_id": "b", "chunk_text": "B", "score": 0.8},
            {"chunk_id": "d", "chunk_text": "D", "score": 0.7},
        ]
        fake_model = type("FakeEmbedding", (), {"embed": lambda _self, _texts: iter([FakeVector([0.1, 0.2])])})()
        fake_session = FakeSession([FakeResponse(similarity), FakeResponse(keywords)])
        with patch.dict(os.environ, {"SUPABASE_URL": "https://offline.invalid", "SUPABASE_SERVICE_ROLE_KEY": "test-key"}):
            with patch("ingestion.retrieve._embedding_model", fake_model):
                with patch("ingestion.retrieve.requests.Session", return_value=fake_session):
                    result = retrieve("test", diagnostics=True)
        self.assertEqual({row["chunk_id"] for row in result["results"]}, {"a", "c"})
        self.assertTrue(
            all(
                row["similarity_distance"] < SIMILARITY_DISTANCE_CUTOFF
                for row in result["results"]
            )
        )

    def test_no_passing_chunks_returns_not_found(self):
        similarity = [{"chunk_id": "a", "chunk_text": "A", "score": 0.8}]
        keywords = [{"chunk_id": "a", "chunk_text": "A", "score": 0.9}]
        fake_model = type("FakeEmbedding", (), {"embed": lambda _self, _texts: iter([FakeVector([0.1])])})()
        with patch.dict(os.environ, {"SUPABASE_URL": "https://offline.invalid", "SUPABASE_SERVICE_ROLE_KEY": "test-key"}):
            with patch("ingestion.retrieve._embedding_model", fake_model):
                with patch(
                    "ingestion.retrieve.requests.Session",
                    return_value=FakeSession([FakeResponse(similarity), FakeResponse(keywords)]),
                ):
                    self.assertEqual(retrieve("test"), "not found")

    def test_no_evidence_does_not_invoke_model(self):
        calls = []
        model = RunnableLambda(lambda _prompt: calls.append(True) or AIMessage(content="unexpected [1]"))
        chain = build_answer_chain(
            retriever=RunnableLambda(lambda _question: []),
            model=model,
        )
        result = chain.invoke({"question": "no evidence"})
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(calls, [])

    def test_empty_superseded_by_is_stored_as_null(self):
        row = _notice_row({"content_hash": "hash", "_id": "n1", "superseded_by": ""})
        self.assertIsNone(row["superseded_by"])


if __name__ == "__main__":
    unittest.main()
