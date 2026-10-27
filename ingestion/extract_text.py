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
DATE_PATTERN = re.compile(
    r"\b(?:\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+(?:\s+20\d{2})?|"
    r"[A-Z][a-z]+\s+\d{1,2}(?:,?\s+20\d{2})?|\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b"
)
DEADLINE_CUE = re.compile(r"\b(?:deadline|last date|on or before|by|due date|submit by)\b", re.IGNORECASE)
SEMESTER_PATTERN = re.compile(
    r"\b(?:semester\s*(?:no\.?\s*)?\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s*semester|"
    r"(?:odd|even)\s*(?:semester|term)|(?:january|jan|july|jul)[- ](?:june|jun|december|dec)"
    r"(?:\s*,?\s*20\d{2})?)\b",
    re.IGNORECASE,
)
DEPARTMENT_PATTERN = re.compile(
    r"\b(?:department\s+of\s+[A-Z][A-Za-z&()/ -]{1,50}|"
    r"[A-Z][A-Za-z&()/ -]{1,40}\s+Department)\b"
)
ACTION_PATTERN = re.compile(
    r"^\s*(?:please\s+|students?\s+(?:are|should|must)\s+(?:advised|required|requested)\s+to\s+)?"
    r"(?:submit|fill(?:\s+in|\s+out)?|register|apply|pay|deposit|upload|collect|"
    r"report|appear|attend|contact|verify|download|complete|present)\b[^.!?\n]*",
    re.IGNORECASE,
)


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


def extract_metadata(notice: dict, text: str) -> dict:
    """Keep supplied values and infer only missing notice metadata."""
    metadata = {
        "date": notice.get("date"),
        "category": notice.get("category"),
        "department": notice.get("department"),
        "semester": notice.get("semester"),
        "deadline": notice.get("deadline"),
        "deadline_source_sentence": notice.get("deadline_source_sentence"),
        "required_action": notice.get("required_action"),
    }
    source = "\n".join(value for value in (notice.get("title", ""), text) if value)
    sentences = [line.strip() for line in re.split(r"(?<=[.!?])\s+|\n+", source) if line.strip()]

    if not metadata["deadline"]:
        for sentence in sentences:
            if DEADLINE_CUE.search(sentence):
                match = DATE_PATTERN.search(sentence)
                if match:
                    metadata["deadline"] = match.group(0)
                    metadata["deadline_source_sentence"] = sentence
                    break

    if not metadata["semester"]:
        match = SEMESTER_PATTERN.search(source)
        if match:
            metadata["semester"] = re.sub(r"\s+", " ", match.group(0)).strip()

    if not metadata["department"]:
        match = DEPARTMENT_PATTERN.search(source)
        if match:
            metadata["department"] = re.sub(r"\s+", " ", match.group(0)).strip()

    if not metadata["required_action"]:
        for sentence in sentences:
            match = ACTION_PATTERN.search(sentence)
            if match:
                metadata["required_action"] = re.sub(r"\s+", " ", match.group(0)).strip()
                break

    return metadata


def extract_notice_text(notice: dict) -> dict:
    """Return a copy with cleaned text, extraction status, and PDF provenance."""
    result = dict(notice)
    existing_text, pdf_url = _text_and_pdf_link(notice)

    if existing_text.strip():
        result["text"] = clean_text(existing_text)
        result["extraction_status"] = "text_available"
        result["content_hash"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
        result["metadata"] = extract_metadata(notice, result["text"])
        return result

    if not pdf_url:
        result["text"] = ""
        result["extraction_status"] = "no_text_or_pdf"
        result["content_hash"] = hashlib.sha256(b"").hexdigest()
        result["metadata"] = extract_metadata(notice, "")
        return result

    result["pdf_url"] = pdf_url
    result["text"], result["extraction_status"] = _extract_pdf(pdf_url)
    result["content_hash"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
    result["metadata"] = extract_metadata(notice, result["text"])
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
            result["metadata"] = extract_metadata(notice, "")
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
