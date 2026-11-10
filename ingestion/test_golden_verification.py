"""Offline tests: verified and unverified golden cases are reported separately."""
from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from ingestion.evaluate_chain import _golden_exit_code, _run_golden_evaluation, _summarize_golden
from ingestion.validate_golden import validate_payload


def case(case_id, verified=None, status="not_found", kind="no_answer"):
    value = {"id": case_id, "question": "A test?", "type": kind, "expected_status": status,
             "must_contain_facts": [], "expected_notice_titles": [], "notes": "test"}
    if verified is not None:
        value["verified"] = verified
    return value


def row(case_id, verified, ok=True, status_match=True):
    return {"id": case_id, "type": "no_answer", "verified": verified, "status_match": status_match,
            "top1_retrieval_hit": None, "top5_retrieval_hit": None, "citation_correctness": True,
            "no_answer_correctness": status_match, "redirect_correctness": None, "fact_match": "needs manual check",
            "freshness_incident": False, "model_call_latency": 0, "attempts": 0, "fallback": False, "cached": False,
            "actual_status": "not_found", "pass/fail": "pass" if ok else "fail"}


class ValidatorTests(unittest.TestCase):
    def test_verified_must_be_a_boolean_and_unverified_cases_warn(self):
        errors, warnings, _ = validate_payload([case("a", verified="yes")])
        self.assertTrue(any("verified must be true or false" in error for error in errors))
        errors, warnings, _ = validate_payload([case("a", verified=False), case("b", verified=True), case("c")])
        self.assertEqual(errors, [])
        self.assertTrue(any("1 of 3 cases are unverified" in warning for warning in warnings))


class SummaryTests(unittest.TestCase):
    def test_counts_and_metrics_are_split_and_a_missing_flag_counts_as_verified(self):
        rows = [row("a", True), row("b", None), row("c", False, ok=False, status_match=False), row("c", False, ok=False, status_match=False)]
        rows[1].pop("verified")
        summary = _summarize_golden(rows, repeat=2)
        verification = summary["verification"]
        self.assertEqual((verification["verified_cases"], verification["unverified_cases"]), (2, 1))
        self.assertEqual(verification["verified"]["status_match_rate"], 1.0)
        self.assertEqual(verification["unverified"]["status_match_rate"], 0.0)
        self.assertEqual(summary["overall"]["cases"], 4)

    def test_unverified_failures_do_not_gate_the_exit_code(self):
        rows = [row("a", True), row("c", False, ok=False, status_match=False)]
        self.assertEqual(_golden_exit_code(_summarize_golden(rows, 1), []), 0)
        self.assertEqual(_golden_exit_code(_summarize_golden(rows, 1), ["unverified cases"]), 2)
        failing = [row("a", True, ok=False, status_match=False), row("c", False)]
        self.assertEqual(_golden_exit_code(_summarize_golden(failing, 1), []), 1)

    def test_with_no_verified_case_the_overall_metrics_gate(self):
        rows = [row("c", False, ok=False, status_match=False)]
        self.assertEqual(_golden_exit_code(_summarize_golden(rows, 1), []), 1)


class RunnerTests(unittest.TestCase):
    def test_runner_reports_counts_marks_a_warning_and_saves_the_flag(self):
        runner = lambda item, **_kw: {**item, "actual_status": "not_found_retrieval", "answer": "", "cited_titles": [],
                                      "sources": [], "retrieved_titles": [], "attempts": 0, "model_call_latency": 0}
        many = [case(f"n-{i}", verified=True) for i in range(49)] + [case("draft-1", verified=False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "golden.json"
            path.write_text(json.dumps(many), encoding="utf-8")
            out = StringIO()
            with redirect_stdout(out):
                code = _run_golden_evaluation(golden_path=path, provider_config=None, results_dir=Path(directory) / "r", case_runner=runner)
            self.assertEqual(code, 2)  # the verified cases pass; the unverified count is a warning
            text = out.getvalue()
            self.assertIn("Golden set verified: 49, unverified: 1", text)
            self.assertIn("Verified cases: 49, unverified cases: 1", text)
            saved = json.loads(next((Path(directory) / "r").glob("golden_*.json")).read_text(encoding="utf-8"))
            self.assertEqual(saved["summary"]["verification"]["unverified_cases"], 1)
            self.assertFalse(next(r for r in saved["results"] if r["id"] == "draft-1")["verified"])
            markdown = next((Path(directory) / "r").glob("golden_*.md")).read_text(encoding="utf-8")
            self.assertIn("Unverified cases (not used for the exit code): 1", markdown)


if __name__ == "__main__":
    unittest.main()
