"""Offline tests for the manual fact-check sheet generator."""

from __future__ import annotations

import unittest

from ingestion.make_fact_check import build_fact_check, extract_claims


class FactCheckSheetTests(unittest.TestCase):
    def test_extracts_checkable_spans_without_evaluating_them(self):
        claims = extract_claims(
            "Notice COE/155/772, dated 2 September 2026, applies w.e.f. 3 Sept 2026; "
            "submit by 12 p.m. at https://example.test/notice."
        )
        self.assertTrue(any("COE/155/772" in claim for claim in claims))
        self.assertTrue(any("2 September 2026" in claim for claim in claims))
        self.assertTrue(any("12 p.m." in claim for claim in claims))
        self.assertTrue(any("https://example.test/notice" in claim for claim in claims))
        self.assertTrue(any("w.e.f." in claim for claim in claims))

    def test_sheet_uses_cited_source_metadata_and_leaves_review_cells_blank(self):
        sheet = build_fact_check([{
            "case": "schedule",
            "actual_status": "answered",
            "question": "When?",
            "answer": "It starts 2 September 2026 [1].",
            "sources": [{
                "title": "Exam notice", "published_at": "2026-09-01",
                "source_url": "https://example.test/source",
            }],
        }])
        self.assertIn("Exam notice", sheet)
        self.assertIn("2026-09-01", sheet)
        self.assertIn("https://example.test/source", sheet)
        self.assertIn("| date: 2 September 2026 |  |  |", sheet)
        self.assertIn("does not judge correctness", sheet)


if __name__ == "__main__":
    unittest.main()
