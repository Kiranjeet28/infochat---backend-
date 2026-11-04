"""Create a human fact-check sheet from offline evaluation results."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "results.json"
OUTPUT_PATH = ROOT / "results" / "fact_check.md"

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*|\d+\s*-\s*\d+)\]")
_CLAIM_PATTERNS = (
    ("w.e.f. date", re.compile(r"\bw\.?e\.?f\.?\s*[^\n.;,]*", re.IGNORECASE)),
    ("URL", re.compile(r"https?://[^\s)\]>]+", re.IGNORECASE)),
    ("notice number", re.compile(r"\b[A-Z]{2,}(?:/[A-Z0-9-]+){1,}\b")),
    ("date", re.compile(
        r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{4})\b", re.IGNORECASE)),
    ("time", re.compile(r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)(?![A-Za-z])|\bnoon\b", re.IGNORECASE)),
    ("number", re.compile(r"\b\d+(?:[.,]\d+)*\b")),
)


def _indexes(answer: str, source_count: int) -> list[int]:
    indexes: list[int] = []
    for match in _CITATION.finditer(answer):
        body = match.group(1).replace(" ", "")
        if "-" in body:
            first, last = (int(part) for part in body.split("-", 1))
            indexes.extend(range(first, last + 1))
        else:
            indexes.extend(int(part) for part in body.split(","))
    return list(dict.fromkeys(index for index in indexes if 1 <= index <= source_count))


def extract_claims(answer: str) -> list[str]:
    """Collect checkable spans without assessing their truth."""
    found: list[str] = []
    seen: set[str] = set()
    for label, pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(answer):
            claim = f"{label}: {match.group(0).strip()}"
            if claim not in seen:
                found.append(claim)
                seen.add(claim)
    return found


def _cell(value: Any) -> str:
    if value is None or value == "":
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def build_fact_check(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Manual fact-check sheet",
        "",
        "Automated extraction only. Verify every claim against the official notice; this sheet does not judge correctness.",
        "",
    ]
    for item in results:
        if item.get("actual_status") != "answered":
            continue
        answer = str(item.get("answer") or "")
        sources = item.get("sources") or []
        cited_sources = [sources[index - 1] for index in _indexes(answer, len(sources))]
        if not cited_sources:
            cited_sources = [
                {"title": title, "published_at": None, "source_url": None}
                for title in item.get("cited_titles", [])
            ]
        titles = list(dict.fromkeys(source.get("title") or "Untitled notice" for source in cited_sources))
        source_lines = [
            f"- **{source.get('title') or 'Untitled notice'}** — published: "
            f"{source.get('published_at') or 'not available'}; URL: "
            f"{source.get('source_url') or 'not available'}"
            for source in cited_sources
        ] or ["- No cited source metadata is present in this result."]
        lines.extend([
            f"## {item.get('case') or 'Evaluation case'}",
            "",
            f"**Question:** {_cell(item.get('question'))}",
            "",
            f"**Answer:** {_cell(answer)}",
            "",
            f"**Cited notice titles:** {_cell(', '.join(titles)) or 'None'}",
            "",
            "**Sources:**",
            *source_lines,
            "",
            "| Checkable claim (extracted; not verified) | Matches official notice? (Y/N) | My notes |",
            "|---|---|---|",
        ])
        claims = extract_claims(answer)
        for claim in claims or ["No date, time, number, URL, notice number, or w.e.f. span detected."]:
            lines.append(f"| {_cell(claim)} |  |  |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(build_fact_check(results), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
