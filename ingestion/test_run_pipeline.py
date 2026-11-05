"""Offline orchestration checks using fake API and Supabase adapters."""
import unittest
from pathlib import Path
from unittest.mock import patch

from ingestion import run_pipeline as pipeline


class Response:
    url = "https://api.example.test/notices"
    def __init__(self, items=None, error=None): self.items, self.error = items or [], error
    def raise_for_status(self):
        if self.error: raise self.error
    def json(self): return {"data":{"items":self.items}}


class FakeAPI:
    def __init__(self, pages): self.pages, self.calls = pages, []
    def get(self, _url, *, params, timeout):
        self.calls.append(params["page"])
        value = self.pages.get(params["page"], [])
        if isinstance(value, Exception): raise value
        return Response(value)


class FakeStore:
    url, key = "https://db.example.test", "not-a-secret"
    def __init__(self, known=()): self.rows=list(known); self.writes=[]
    def known(self): return self.rows
    def version(self): return "4"
    def persist_notice(self, *args): self.writes.append(("notice",args))
    def delete_chunks(self, *args): self.writes.append(("delete",args))
    def write_chunks(self, *args): self.writes.append(("chunks",args)); return 0
    def mark_superseded(self, *args): self.writes.append(("superseded",args))
    def withdrawn_available(self): return True
    def withdraw_missing(self, *args): self.writes.append(("withdraw",args))
    def invalidate(self, *args): self.writes.append(("invalidate",args)); return 0


class FakeSplitter:
    def split_text(self, text): return [text]


def processed(notices):
    return [dict(n, notice_id=n.get("_id"), text=n.get("text", "x"), content_hash=n.get("content_hash", "h"),
                 extraction_confidence=.9, published_at="2026-10-01T00:00:00Z", version_status="no_match") for n in notices]


class RunPipelineOfflineTests(unittest.TestCase):
    def notice(self, notice_id="n1", content_hash="h1"):
        return {"_id":notice_id,"title":"Notice","url":"https://erp.gndec.ac.in/n","text":"content",
                "content_hash":content_hash}

    def test_incremental_stops_after_all_known_page(self):
        api=FakeAPI({1:[self.notice()]}); store=FakeStore([{"notice_id":"n1","content_hash":"h1"}])
        with patch.object(pipeline,"warm_up"):
            code, summary=pipeline.run(api_url="https://api.example.test",store=store,session=api,processor=processed)
        self.assertEqual(api.calls,[1]); self.assertEqual(summary["unchanged"],1); self.assertEqual(code,0)

    def test_full_fetches_through_empty_page_and_honors_cap(self):
        api=FakeAPI({1:[self.notice()],2:[]}); store=FakeStore()
        with patch.object(pipeline,"warm_up"), patch.object(pipeline,"bump_kb_version"), patch.object(pipeline.RecursiveCharacterTextSplitter,"from_tiktoken_encoder",return_value=FakeSplitter()):
            _, summary=pipeline.run(mode="full",max_pages=3,api_url="https://api.example.test",store=store,session=api,processor=processed)
        self.assertEqual(api.calls,[1,2]); self.assertTrue(summary["fetched"])

    def test_dry_run_performs_no_writes(self):
        api=FakeAPI({1:[self.notice()],2:[]}); store=FakeStore()
        with patch.object(pipeline,"warm_up"):
            _, summary=pipeline.run(mode="full",dry_run=True,api_url="https://api.example.test",store=store,session=api,processor=processed)
        self.assertEqual(store.writes,[]); self.assertEqual(summary["kb_version_before"],"4")

    def test_review_gate_lists_missing_conditions(self):
        status,reasons=pipeline.decide_review_status({"title":"","url":"http://bad.test/x","text":"short","extraction_confidence":.1,"published_at":"bad"})
        self.assertEqual(status,"needs_review")
        self.assertIn("missing_title",reasons)
        self.assertIn("source_url_not_https",reasons)
        self.assertIn("published_at_invalid",reasons)

    def test_domain_accepts_subdomains_and_rejects_suffix_attack(self):
        base={"title":"T","published_at":"2026-10-01T00:00:00Z","text":"x"*40,"extraction_confidence":.6}
        self.assertEqual(pipeline.decide_review_status(dict(base,url="https://gndec.ac.in/a"))[0],"ok")
        self.assertEqual(pipeline.decide_review_status(dict(base,url="https://erp.gndec.ac.in/a"))[0],"ok")
        self.assertIn("source_domain_not_allowed",pipeline.decide_review_status(dict(base,url="https://gndec.ac.in.evil.com/a"))[1])

    def test_default_minimum_length_is_40_and_length_only_is_reported(self):
        self.assertEqual(pipeline.MIN_TEXT_LENGTH,40)
        status,reasons=pipeline.decide_review_status({"title":"T","url":"https://erp.gndec.ac.in/a","published_at":"2026-10-01T00:00:00Z","text":"x","extraction_confidence":.6})
        self.assertEqual((status,reasons),("needs_review",["text_too_short"]))

    def test_incremental_links_new_notice_to_stored_candidate(self):
        old={"notice_id":"old","title":"Exam date sheet semester 5","category":"Examinations","department":"CSE","published_at":"2026-09-01","review_status":"ok","superseded_by":None,"content_hash":"oldhash"}
        new=self.notice("new","newhash")
        new.update(title="Updated exam date sheet semester 5",text="This notice replaces the previous schedule.",category="Examinations",department="CSE")
        api=FakeAPI({1:[new]}); store=FakeStore([old])
        with patch.object(pipeline,"warm_up"), patch.object(pipeline,"bump_kb_version"), patch.object(pipeline.RecursiveCharacterTextSplitter,"from_tiktoken_encoder",return_value=FakeSplitter()):
            _,_=pipeline.run(api_url="https://api.example.test",store=store,session=api,processor=processed)
        self.assertIn(("superseded",("old","new")),store.writes)

    def test_workflow_has_schedule_guard_and_structural_order(self):
        workflow=Path(__file__).parents[1]/".github/workflows/cascadeiq-ingest.yml"
        source=workflow.read_text(encoding="utf-8")
        self.assertIn("schedule:",source)
        self.assertIn("if: github.event_name == 'workflow_dispatch' || vars.INGEST_ENABLED == 'true'",source)
        self.assertLess(source.index("jobs:"),source.index("  ingest:"))
        self.assertLess(source.index("  ingest:"),source.index("    steps:"))


if __name__ == "__main__": unittest.main()
