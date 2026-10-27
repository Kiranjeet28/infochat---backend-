import unittest
from unittest.mock import patch

import requests

from extract_text import process_notices


def make_notice(**overrides):
    notice = {
        "_id": "good-1",
        "title": "Fee payment notice",
        "date": "October 1, 2026",
        "url": "",
        "htmlContent": "<p>Pay the fee by 15 October 2026.</p>",
        "source": {
            "api_url": "https://api.example.test/api/notices?page=1&pageSize=9&sort=latest",
            "official_notice_url": "https://erp.gndec.ac.in/noticeboard/fee-payment",
            "fetched_at": "2026-10-27T10:00:00+05:30",
        },
    }
    notice.update(overrides)
    return notice


class IngestionPipelineTests(unittest.TestCase):
    def test_good_notice_is_saved_with_clean_text_and_hash(self):
        [result] = process_notices([make_notice()])

        self.assertEqual(result["ingestion_status"], "success")
        self.assertEqual(result["text"], "Pay the fee by 15 October 2026.")
        self.assertEqual(len(result["content_hash"]), 64)

    def test_missing_date_is_flagged_without_stopping_batch(self):
        [result] = process_notices([make_notice(_id="missing-date", date=None)])

        self.assertEqual(result["ingestion_status"], "success")
        self.assertEqual(result["review_status"], "needs_review")
        self.assertIn("missing_or_invalid_date", result["review_flags"])

    @patch("extract_text.requests.get", side_effect=requests.ConnectionError("host unavailable"))
    def test_pdf_fetch_failure_is_recorded_and_other_notice_continues(self, _mock_get):
        bad_pdf = make_notice(
            _id="bad-pdf",
            title="Scanned notice",
            url="https://invalid.example.test/missing.pdf",
            htmlContent="",
        )
        results = process_notices([make_notice(), bad_pdf])

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["ingestion_status"], "success")
        self.assertEqual(results[1]["ingestion_status"], "failed")
        self.assertIn("PDF extraction failed", results[1]["ingestion_failure_reason"])


if __name__ == "__main__":
    unittest.main()
