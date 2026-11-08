"""Offline tests for golden-set validation, scoring, and saved artifacts."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ingestion.compat_uuid_utils import ensure_uuid_utils

ensure_uuid_utils()  # Windows Application Control workaround; before langchain imports

from ingestion.evaluate_chain import (
    _detect_regressions,
    _evaluate_golden_case,
    _golden_exit_code,
    _normalize_fact,
    _run_golden_evaluation,
    _summarize_golden,
    _write_golden_results,
)
from ingestion.validate_golden import validate_payload


def case(case_id="case-1", kind="no_answer", status="not_found", facts=None):
    return {
        "id": case_id, "question": "A test?", "type": kind,
        "expected_status": status, "must_contain_facts": facts or [],
        "expected_notice_titles": [], "notes": "test",
    }


class GoldenSetTests(unittest.TestCase):
    def test_validator_counts_types_and_warns_for_small_set_and_empty_answerable_facts(self):
        payload = [case(kind="answerable", status="answered"), case("n2")]
        errors, warnings, counts = validate_payload(payload)
        self.assertEqual(errors, [])
        self.assertEqual(counts["answerable"], 1)
        self.assertEqual(counts["no_answer"], 1)
        self.assertTrue(any("fewer" in warning or "recommended minimum" in warning for warning in warnings))
        self.assertTrue(any("answerable case has no" in warning for warning in warnings))

    def test_validator_rejects_duplicate_ids_missing_fields_and_bad_status(self):
        first = case()
        second = {**case(), "expected_status": "unknown"}
        second.pop("notes")
        errors, _warnings, _counts = validate_payload([first, second])
        self.assertTrue(any("duplicate id" in error for error in errors))
        self.assertTrue(any("missing fields" in error for error in errors))

    def test_fact_normalization_ignores_punctuation_case_and_unicode_dash(self):
        self.assertEqual(_normalize_fact("May—2026,  12 NOON!"), "may 2026 12 noon")

    def test_golden_metrics_have_expected_rates_latency_and_fallback(self):
        rows = [
            {"id": "a", "type": "answerable", "status_match": True, "top1_retrieval_hit": True, "top5_retrieval_hit": True, "citation_correctness": True, "no_answer_correctness": None, "redirect_correctness": None, "fact_match": True, "freshness_incident": False, "model_call_latency": 1.0, "attempts": 1, "fallback": False, "cached": False, "actual_status": "answered", "pass/fail": "pass"},
            {"id": "b", "type": "answerable", "status_match": False, "top1_retrieval_hit": False, "top5_retrieval_hit": True, "citation_correctness": False, "no_answer_correctness": None, "redirect_correctness": None, "fact_match": False, "freshness_incident": False, "model_call_latency": 3.0, "attempts": 2, "fallback": True, "cached": True, "actual_status": "error", "pass/fail": "fail"},
        ]
        summary = _summarize_golden(rows, repeat=1)["overall"]
        self.assertEqual(summary["status_match_rate"], .5)
        self.assertEqual(summary["retrieval_top5_hit_rate"], 1.0)
        self.assertEqual(summary["latency_median"], 2.0)
        self.assertEqual(summary["latency_p95"], 3.0)
        self.assertEqual(summary["fallback_rate"], .5)
        self.assertEqual(summary["cache_hit_rate"], .5)

    def test_case_metrics_check_title_facts_and_freshness(self):
        golden = {**case("answer", "answerable", "answered", ["fee paid"]), "expected_notice_titles": ["Fee Notice"]}
        raw = {"actual_status": "answered", "answer": "Fee—PAID [1]", "cited_titles": ["Fee Notice"], "retrieved_titles": ["Fee Notice"], "sources": [{"title": "Fee Notice", "superseded_by": None, "withdrawn": False}], "attempts": 1, "provider": "fake", "model_call_latency": .1}
        result = _evaluate_golden_case(golden, raw, provider_config=None)
        self.assertTrue(result["status_match"])
        self.assertTrue(result["top1_retrieval_hit"])
        self.assertTrue(result["citation_correctness"])
        self.assertTrue(result["fact_match"])
        self.assertFalse(result["freshness_incident"])
        raw["sources"][0]["withdrawn"] = True
        stale = _evaluate_golden_case(golden, raw, provider_config=None)
        self.assertTrue(stale["freshness_incident"])
        self.assertEqual(stale["pass/fail"], "fail")

    def test_regression_and_fix_detection(self):
        previous = [{"id": "regressed", "pass/fail": "pass"}, {"id": "fixed", "pass/fail": "fail"}]
        current = [{"id": "regressed", "pass/fail": "fail"}, {"id": "fixed", "pass/fail": "pass"}]
        self.assertEqual(_detect_regressions(previous, current), (["regressed"], ["fixed"]))

    def test_exit_codes_for_threshold_failure_success_and_warning_only(self):
        good = {"overall": {"status_match_rate": .95, "freshness_incidents": 0, "no_answer_correctness_rate": 1.0}}
        bad = {"overall": {"status_match_rate": .80, "freshness_incidents": 0, "no_answer_correctness_rate": 1.0}}
        self.assertEqual(_golden_exit_code(good, []), 0)
        self.assertEqual(_golden_exit_code(good, ["small set"]), 2)
        self.assertEqual(_golden_exit_code(bad, []), 1)

    def test_results_files_redact_environment_secrets(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_secret_value_1234567890"}):
            results = [{"id": "x", "answer": "key gsk_test_secret_value_1234567890", "type": "answerable", "pass/fail": "pass"}]
            summary = {"overall": {"cases": 1}, "by_type": {}}
            json_path, md_path = _write_golden_results(results, summary, [], [], results_dir=Path(directory), timestamp="20261107_120000")
            self.assertIn("[REDACTED]", json_path.read_text(encoding="utf-8"))
            self.assertNotIn("gsk_test_secret_value_1234567890", json_path.read_text(encoding="utf-8"))
            self.assertNotIn("gsk_test_secret_value_1234567890", md_path.read_text(encoding="utf-8"))

    def test_runner_returns_warning_code_and_pass_code_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            golden_path = Path(directory) / "golden.json"
            golden_path.write_text(json.dumps([case()]), encoding="utf-8")
            runner = lambda item, **_kwargs: {**item, "actual_status": "not_found_retrieval", "answer": "", "cited_titles": [], "sources": [], "retrieved_titles": [], "attempts": 0, "model_call_latency": 0, "pass/fail": "pass"}
            with redirect_stdout(StringIO()):
                warning_code = _run_golden_evaluation(golden_path=golden_path, provider_config=None, results_dir=Path(directory) / "warning", case_runner=runner)
            self.assertEqual(warning_code, 2)
            many = [case(f"n-{index}") for index in range(50)]
            golden_path.write_text(json.dumps(many), encoding="utf-8")
            with redirect_stdout(StringIO()):
                success_code = _run_golden_evaluation(golden_path=golden_path, provider_config=None, results_dir=Path(directory) / "success", case_runner=runner)
            self.assertEqual(success_code, 0)


if __name__ == "__main__":
    unittest.main()
