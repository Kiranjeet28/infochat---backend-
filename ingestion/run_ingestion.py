"""Fetch, enrich, validate, and save every notice in one command."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from extract_text import OUTPUT_PATH as ENRICHED_PATH, process_notices
from fetch_notices import fetch_notices


RAW_PATH = Path(__file__).parent / "data" / "notices.json"


def run() -> list[dict]:
    notices = fetch_notices()
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_text(json.dumps(notices, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    processed = process_notices(notices)
    ENRICHED_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENRICHED_PATH.write_text(
        json.dumps(processed, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ingestion_counts = Counter(item["ingestion_status"] for item in processed)
    review_counts = Counter(item["review_status"] for item in processed)
    print(f"Fetched {len(notices)} notices")
    print(f"Saved enriched notices to {ENRICHED_PATH}")
    print("Ingestion: " + ", ".join(f"{key}={value}" for key, value in sorted(ingestion_counts.items())))
    print("Review: " + ", ".join(f"{key}={value}" for key, value in sorted(review_counts.items())))
    return processed


if __name__ == "__main__":
    run()
