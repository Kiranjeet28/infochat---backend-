"""Offline tests: restricted notices (student lists) never reach the model with names or roll numbers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from langchain_core.documents import Document

from ingestion.answer_chain import format_evidence
from ingestion.answer_config import restricted_notice_patterns
from ingestion.prompt import PROMPT_VERSION, SYSTEM_PROMPT

NAMES = "Rajinder Singh Gill 2210123456 CSE\nAnita Kumari Sharma 2210123457 ECE"


def doc(title, text=NAMES, category=None):
    return Document(page_content=text, metadata={"notice_id": "n1", "title": title, "category": category,
                                                 "published_at": "2026-08-04", "source_url": "https://erp.gndec.ac.in/n1"})


class RestrictedNoticeTests(unittest.TestCase):
    def test_default_patterns_withhold_names_and_digit_runs_but_keep_the_source_link(self):
        for title in ("List of OE sem 3rd(25-29)  session july-dec.2026", "Readmission_Notice 12_04.08.26",
                      "Struck_off_Notice_42_07.07.26", "LIST OF OE SEM 5TH", "readmission-notice"):
            with self.subTest(title=title):
                evidence = format_evidence([doc(title)])
                for secret in ("Rajinder", "Anita", "2210123456", "2210123457"):
                    self.assertNotIn(secret, evidence)
                self.assertIn("withheld", evidence)
                self.assertIn("https://erp.gndec.ac.in/n1", evidence)
                self.assertIn(title, evidence)

    def test_category_match_and_ordinary_notices(self):
        self.assertNotIn("Rajinder", format_evidence([doc("Some notice", category="Readmission")]))
        plain = format_evidence([doc("Schedule for End Semester Makeup Examination", text="Exam on 11.9.2026 onwards")])
        self.assertIn("Exam on 11.9.2026 onwards", plain)

    def test_struck_off_keeps_its_original_summary(self):
        self.assertIn("students struck off the rolls", format_evidence([doc("Struck_off_Notice_42")]))

    def test_patterns_are_configurable_by_environment(self):
        with patch.dict(os.environ, {"RESTRICTED_NOTICE_PATTERNS": "Merit List, hostel allotment"}):
            self.assertEqual(restricted_notice_patterns(), ("merit list", "hostel allotment"))
            self.assertNotIn("Rajinder", format_evidence([doc("Final Merit-List 2026")]))
            self.assertIn("Rajinder", format_evidence([doc("List of OE sem 3rd")]))  # defaults replaced
        with patch.dict(os.environ, {"RESTRICTED_NOTICE_PATTERNS": ""}):
            self.assertEqual(restricted_notice_patterns(), ())
        os.environ.pop("RESTRICTED_NOTICE_PATTERNS", None)
        self.assertEqual(restricted_notice_patterns(), ("struck off", "list of oe", "readmission"))

    def test_prompt_covers_withheld_notices_and_version_was_bumped(self):
        self.assertIn("names and roll numbers are withheld", SYSTEM_PROMPT)
        self.assertNotEqual(PROMPT_VERSION, "2026-11-05-v1")


if __name__ == "__main__":
    unittest.main()
