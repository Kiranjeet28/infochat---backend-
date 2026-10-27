"""Extract and clean text from API notice records."""

from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

import pdfplumber
import requests
from bs4 import BeautifulSoup


INPUT_PATH = Path(__file__).parent / "data" / "notices.json"
OUTPUT_PATH = Path(__file__).parent / "data" / "notices_extracted.json"
REQUEST_TIMEOUT_SECONDS = 60
PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?$", re.IGNORECASE)


def clean_text(text: str, page_texts: list[str] | None = None) -> str:
    """Normalize whitespace and remove page numbers and repeated page furniture."""
    repeated: set[str] = set()
    if page_texts and len(page_texts) > 1:
        normalized_pages = [
            {re.sub(r"\s+", " ", line).strip().casefold() for line in page.splitlines() if line.strip()}
            for page in page_texts
        ]
        counts = Counter(line for lines in normalized_pages for line in lines)
        repeated = {
            line
            for line, count in counts.items()
            if count >= 2 and count / len(normalized_pages) >= 0.5
        }

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        if not line or PAGE_NUMBER.fullmatch(line):
            continue
        if line.casefold() in repeated:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _text_and_pdf_link(notice: dict) -> tuple[str, str | None]:
    html = notice.get("htmlContent") or ""
    soup = BeautifulSoup(html, "html.parser")
    pdf_link = None
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if ".pdf" in href.casefold():
            pdf_link = urljoin(notice.get("url") or "", href)
            break

    notice_url = notice.get("url") or ""
    if pdf_link is None and ".pdf" in notice_url.casefold():
        pdf_link = notice_url

    html_text = soup.get_text("\n", strip=True)
    return html_text, pdf_link


def _extract_pdf(pdf_url: str) -> tuple[str, str]:
    response = requests.get(pdf_url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    with pdfplumber.open(BytesIO(response.content)) as pdf:
        pages = [(page.extract_text() or "") for page in pdf.pages]
    text = clean_text("\n".join(pages), page_texts=pages)
    return text, "extracted" if text else "needs_ocr"


def extract_notice_text(notice: dict) -> dict:
    """Return a copy with cleaned text, extraction status, and PDF provenance."""
    result = dict(notice)
    existing_text, pdf_url = _text_and_pdf_link(notice)

    if existing_text.strip():
        result["text"] = clean_text(existing_text)
        result["extraction_status"] = "text_available"
        result["content_hash"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
        return result

    if not pdf_url:
        result["text"] = ""
        result["extraction_status"] = "no_text_or_pdf"
        result["content_hash"] = hashlib.sha256(b"").hexdigest()
        return result

    result["pdf_url"] = pdf_url
    result["text"], result["extraction_status"] = _extract_pdf(pdf_url)
    result["content_hash"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
    return result


def main() -> None:
    notices = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    extracted = []
    for notice in notices:
        try:
            extracted.append(extract_notice_text(notice))
        except Exception as error:
            result = dict(notice)
            _, pdf_url = _text_and_pdf_link(notice)
            result["pdf_url"] = pdf_url
            result["text"] = ""
            result["extraction_status"] = "pdf_error"
            result["content_hash"] = hashlib.sha256(b"").hexdigest()
            result["extraction_error"] = str(error)
            extracted.append(result)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(extracted, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = Counter(item["extraction_status"] for item in extracted)
    print(f"Saved {len(extracted)} notices to {OUTPUT_PATH}")
    print("Statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))


if __name__ == "__main__":
    main()
