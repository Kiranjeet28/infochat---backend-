"""Fetch GNDEC notices from the existing InfoCascade API."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import requests


API_ENDPOINT = "https://infocascade-database.onrender.com/api/notices"
PAGE_SIZE = 9
REQUEST_TIMEOUT_SECONDS = 60
OUTPUT_PATH = Path(__file__).parent / "data" / "notices.json"


def fetch_notices() -> list[dict]:
    """Fetch every non-empty page and attach source provenance per notice."""
    notices: list[dict] = []
    page = 1

    with requests.Session() as session:
        while True:
            response = session.get(
                API_ENDPOINT,
                params={"page": page, "pageSize": PAGE_SIZE, "sort": "latest"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
            items = payload.get("data", {}).get("items", [])
            if not items:
                break

            api_url = response.url
            fetched_at = datetime.now(timezone.utc).isoformat()
            for item in items:
                notice = dict(item)
                notice["source"] = {
                    "api_url": api_url,
                    "official_notice_url": item.get("url"),
                    "fetched_at": fetched_at,
                }
                notices.append(notice)

            page += 1

    return notices


def main() -> None:
    notices = fetch_notices()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(notices, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(notices)} notices to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
