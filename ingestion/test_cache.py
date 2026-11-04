"""Offline cache behavior tests; all persistence uses an in-memory store."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import unittest
from unittest.mock import patch

from ingestion.cache_store import (
    IST, InMemoryCacheStore, cache_answer_if_valid, cache_key, cacheable_answer,
    expiry_at, make_cache_entry, normalize_question, privacy_guard, question_hash,
    validated_cache_hit,
)


def entry(key: str, notice_id: str = "n1", content_hash: str = "h1") -> dict:
    return {
        "cache_key": key, "answer": "Answer [1]", "status": "answered",
        "sources": [], "source_notice_ids": [notice_id],
        "source_content_hashes": {notice_id: content_hash}, "kb_version": "1",
        "prompt_version": "v1", "provider": "local", "model": "test",
        "created_at": "2026-11-03T00:00:00+00:00",
        "expires_at": "2026-11-03T06:00:00+00:00", "hit_count": 0,
    }


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.store = InMemoryCacheStore()

    def test_miss_then_hit(self):
        self.assertIsNone(self.store.get("k"))
        self.store.set(entry("k"))
        self.assertEqual(self.store.get("k")["answer"], "Answer [1]")

    def test_expiry_ttl(self):
        now = datetime(2026, 11, 3, 10, 0, tzinfo=IST)
        self.assertEqual(expiry_at(now, 3600), now + timedelta(hours=1))

    def test_expired_entry_is_deleted_and_misses(self):
        self.store.set(entry("k"))
        current = datetime(2026, 11, 3, 13, 0, tzinfo=IST)
        hit, _ = validated_cache_hit(
            self.store, "k", now=current, kb_version=1, prompt_version="v1",
            source_verifier=lambda _entry: True,
        )
        self.assertIsNone(hit)
        self.assertIsNone(self.store.get("k"))

    def test_expiry_is_clamped_to_midnight_ist(self):
        now = datetime(2026, 11, 3, 23, 30, tzinfo=IST)
        self.assertEqual(expiry_at(now, 6 * 3600), datetime(2026, 11, 4, 0, 0, tzinfo=now.tzinfo))

    def test_different_filter_and_date_change_key(self):
        base = cache_key(" Q? ", None, None, 1, "v1", date(2026, 11, 3))
        self.assertNotEqual(base, cache_key("Q?", "EE", None, 1, "v1", date(2026, 11, 3)))
        self.assertNotEqual(base, cache_key("Q?", None, None, 1, "v1", date(2026, 11, 4)))

    def test_normalization_is_exact_casefolded_space_and_terminal_punctuation(self):
        self.assertEqual(normalize_question("  WHAT   is this?!  "), "what is this")
        self.assertEqual(question_hash(" WHAT is this? "), question_hash("what is this"))

    def test_kb_version_bump_changes_key(self):
        self.assertNotEqual(cache_key("q", None, None, 4, "v1"), cache_key("q", None, None, 5, "v1"))

    def test_changed_content_hash_invalidates_entry(self):
        self.store.set(entry("k", content_hash="old"))
        hit, _ = validated_cache_hit(
            self.store, "k", now=datetime(2026, 11, 3, 1, tzinfo=IST),
            kb_version=1, prompt_version="v1",
            source_verifier=lambda cached: cached["source_content_hashes"] == {"n1": "new"},
        )
        self.assertIsNone(hit)
        self.assertIsNone(self.store.get("k"))

    def test_superseded_notice_invalidates_only_its_entries(self):
        self.store.set(entry("a", notice_id="n1"))
        self.store.set(entry("b", notice_id="n2"))
        hit, _ = validated_cache_hit(
            self.store, "b", now=datetime(2026, 11, 3, 1, tzinfo=IST),
            kb_version=1, prompt_version="v1",
            source_verifier=lambda cached: not cached["source_notice_ids"] == ["n1"],
        )
        self.assertIsNotNone(hit)
        self.store.delete_by_notice("n1")
        self.assertIsNone(self.store.get("a"))
        self.assertIsNotNone(self.store.get("b"))

    def test_superseded_source_invalidates_cached_answer(self):
        self.store.set(entry("k", notice_id="n1"))
        hit, _ = validated_cache_hit(
            self.store, "k", now=datetime(2026, 11, 3, 1, tzinfo=IST),
            kb_version=1, prompt_version="v1", source_verifier=lambda _cached: False,
        )
        self.assertIsNone(hit)
        self.assertIsNone(self.store.get("k"))

    def test_failures_and_not_found_are_never_cacheable(self):
        for status in ("ai_unavailable", "invalid_citation", "error"):
            result = {"status": status}
            self.assertFalse(cacheable_answer(result, True))
            self.assertFalse(cache_answer_if_valid(self.store, result, True, entry("failed")))
        for status in ("not_found_model", "not_found_retrieval"):
            self.assertFalse(cache_answer_if_valid(self.store, {"status": status}, True, entry("not-found")))
        self.assertFalse(cacheable_answer({"status": "answered"}, False))
        self.assertTrue(cacheable_answer({"status": "answered"}, True))
        self.assertEqual(self.store.entries, {})

    def test_cache_entry_keeps_question_hash_but_not_question(self):
        raw_question = "Which exam centre was combined with another one?"
        cached = make_cache_entry(
            cache_key_value="k", question=raw_question, answer="C1M [1]",
            sources=[{"notice_id": "n1", "title": "Centre notice", "content_hash": "h1"}],
            source_content_hashes={"n1": "h1"}, kb_version=1, prompt_version="p1",
            provider="test", model="test", now=datetime(2026, 11, 3, tzinfo=IST),
            ttl_seconds=3600,
        )
        self.assertNotIn("question", cached)
        self.assertNotIn(raw_question, repr(cached))
        self.assertEqual(cached["normalized_question_hash"], question_hash(raw_question))

    def test_privacy_guard_skips_email_long_digit_run_and_phone(self):
        for question in ("contact me at a@b.edu", "roll 123456", "call +1 (555) 212-3456"):
            self.assertTrue(privacy_guard(question), question)
        self.assertFalse(privacy_guard("when do classes begin"))

    def test_evaluation_runner_bypasses_cache_by_default(self):
        from ingestion import evaluate_chain
        previous = evaluate_chain.USE_API_CACHE
        evaluate_chain.USE_API_CACHE = False
        calls = []
        try:
            with patch.object(evaluate_chain, "_run_api_cache_case") as cache_runner, \
                 patch.object(evaluate_chain, "retrieve", side_effect=lambda *_a, **_kw: calls.append(True) or {
                     "results": [], "best_distance": None, "keyword_found": False, "not_found": True,
                 }):
                result = evaluate_chain._run_case({"case": "offline", "question": "test", "expected_status": "not_found"})
                cache_runner.assert_not_called()
            self.assertEqual(result["actual_status"], "not_found_retrieval")
            self.assertEqual(calls, [True])
        finally:
            evaluate_chain.USE_API_CACHE = previous

    def test_evaluation_cache_path_requires_explicit_opt_in(self):
        from ingestion import evaluate_chain
        previous = evaluate_chain.USE_API_CACHE
        evaluate_chain.USE_API_CACHE = True
        case = {"case": "offline", "question": "test", "expected_status": "not_found"}
        try:
            with patch.object(evaluate_chain, "_run_api_cache_case", return_value={"actual_status": "not_found"}) as cache_runner:
                result = evaluate_chain._run_case(case)
            cache_runner.assert_called_once_with(case)
            self.assertEqual(result["actual_status"], "not_found")
        finally:
            evaluate_chain.USE_API_CACHE = previous

    def test_sql_permissions_notify_and_destructive_statement_checks(self):
        root = __import__("pathlib").Path(__file__).resolve().parent
        cache_sql = (root / "cache_tables.sql").read_text(encoding="utf-8").casefold()
        search_sql = (root / "search_chunks_freshness.sql").read_text(encoding="utf-8").casefold()
        for name, signature in (
            ("bump_kb_version", ""),
            ("get_kb_version", ""),
            ("verify_cached_sources", "text[], jsonb"),
        ):
            revoke = f"revoke execute on function public.{name}({signature})"
            if name != "bump_kb_version":
                revoke = f"revoke all on function public.{name}({signature})"
            self.assertIn(revoke, cache_sql)
            self.assertIn(f"grant execute on function public.{name}({signature})", cache_sql)
        for name, signature in (
            ("search_chunks_similarity", "extensions.vector, integer, text, text"),
            ("search_chunks_keyword", "text, integer, text, text"),
        ):
            self.assertIn(f"revoke execute on function public.{name}({signature})", search_sql)
            self.assertIn(f"grant execute on function public.{name}({signature})", search_sql)
        self.assertIn("notify pgrst, 'reload schema';", search_sql)
        forbidden = ("drop table", "truncate", "delete")
        for sql in (cache_sql, search_sql):
            for statement in forbidden:
                self.assertNotIn(statement, sql)


if __name__ == "__main__":
    unittest.main()
