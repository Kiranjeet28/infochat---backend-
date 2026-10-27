"""Extract and clean text from API notice records."""

from __future__ import annotations

import json
import hashlib
import re
from collections import Counter
from datetime import datetime
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
    r"[A-Z][a-z]+\s+\d{1,2}(?:,?\s+20\d{2})?|\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b"
)
DEADLINE_CUE = re.compile(
    r"\b(?:deadline|last date|on or before|by|due date|submit by|up\s*to|upto|till|until|before)\b",
    re.IGNORECASE,
)
SEMESTER_PATTERN = re.compile(
    r"\b(?:semester\s*(?:no\.?\s*)?\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s*semester|"
    r"(?:odd|even)\s*(?:semester|term)|(?:january|jan|july|jul)[- ](?:june|jun|december|dec)"
    r"(?:\s*,?\s*20\d{2})?)\b",
    re.IGNORECASE,
)
DEPARTMENT_PATTERN = re.compile(
    r"\b(?:department\s+of\s+[A-Z][A-Za-z&()/ -]{1,50}|"
    r"[A-Z][A-Za-z&()/ -]{1,40}\s+Department|[A-Z]{2,6}\s+Dept\.?|"
    r"[A-Z][A-Za-z&()/ -]{1,40}\s+Dept\.?)\b"
)
ACTION_PATTERN = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?:please\s+|(?:eligible\s+)?students?\b[^.!?\n]{0,100}\b"
    r"(?:can|may|must|should|are\s+(?:advised|directed|required|requested))\s+(?:to\s+)?|"
    r"interested(?:\s+students)?\s+may\s+|students?\s+(?:are|should|must)\s+"
    r"(?:advised|required|requested)\s+to\s+)?"
    r"(?:submit|fill(?:ing|\s+in|\s+out)?|register|apply|pay|deposit|upload|collect|"
    r"report|appear|attend|contact|verify|download|complete|present)\b[^.!?\n]*",
    re.IGNORECASE,
)
STUDENT_ACTION_PATTERN = re.compile(
    r"\b(?:students?|applicants?|candidates?)\b[^.!?\n]{0,180}?"
    r"\b(?:can|may|must|should|are\s+(?:advised|directed|required|requested))\s+(?:to\s+)?"
    r"(?:submit|fill(?:ing|\s+in|\s+out)?|register|apply|pay|deposit|upload|collect|"
    r"report|appear|attend|contact|verify|download|complete|present|view|make)\b[^.!?\n]*",
    re.IGNORECASE,
)
TOPIC_STOPWORDS = {
    "a", "an", "and", "for", "from", "in", "is", "of", "on", "or", "the", "to",
    "notice", "regarding", "2026", "2025", "2024", "may", "june", "july", "august",
}
REPLACEMENT_CUE = re.compile(
    r"\b(?:revised|corrigendum|supersed(?:e|es|ed|ing)|in place of|replaces?|"
    r"dates? extended|extension|re-opening|reopening|corrected)\b",
    re.IGNORECASE,
)


def clean_text(text: str, page_texts: list[str] | None = None) -> str:
    """Normalize whitespace and remove page numbers and repeated page furniture."""
    text = text.replace("\ufffd", " ")
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
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
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

    if pdf_link:
        return "", pdf_link

    if notice_url and len(html_text) >= 160 and not _is_api_summary(html_text):
        return clean_text(html_text), None

    # API htmlContent can be a short generated teaser. Prefer the official
    # notice body and its PDF attachment whenever that teaser points students
    # elsewhere for the actual details.
    if notice_url:
        try:
            response = requests.get(notice_url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            page = BeautifulSoup(response.text, "html.parser")
            editor = page.select_one(".ql-editor")
            if editor:
                for anchor in editor.select("a[href]"):
                    href = anchor.get("href", "").strip()
                    if ".pdf" in href.casefold():
                        return "", urljoin(response.url, href)
                official_text = editor.get_text("\n", strip=True)
                if _is_link_instructions(official_text):
                    return "", None
                if official_text:
                    return clean_text(official_text), None
        except requests.RequestException:
            # Keep the API text as a fallback when the official page is down.
            pass

    return html_text, pdf_link


def _is_api_summary(text: str) -> bool:
    lowered = text.casefold()
    return any(
        phrase in lowered
        for phrase in (
            "check the erp noticeboard",
            "check the erp notice",
            "check erp for",
            "refer to the complete",
            "refer to the erp notice",
            "refer to erp for",
            "refer to the complete notice",
            "read the complete notice",
            "for full details",
            "for all details and instructions",
            "for further details",
            "full notice on erp",
            "essential document is now available for review",
            "this schedule is for",
            "has been issued",
        )
    )


def _is_link_instructions(text: str) -> bool:
    lowered = text.casefold()
    return "click on the link" in lowered or (
        "notice link" in lowered and ".pdf" in lowered
    )


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
    sentences = [
        line.strip()
        for line in re.split(r"(?<=[!?])\s+|(?<!\d)\.(?!\d)\s+|\n+", source)
        if line.strip()
    ]

    if not metadata["deadline"]:
        for sentence in sentences:
            if DEADLINE_CUE.search(sentence):
                matches = list(DATE_PATTERN.finditer(sentence))
                if matches:
                    cue = DEADLINE_CUE.search(sentence)
                    dates_after_cue = [match for match in matches if match.start() >= cue.end()]
                    match = dates_after_cue[0] if dates_after_cue else matches[-1]
                    metadata["deadline"] = match.group(0)
                    metadata["deadline_source_sentence"] = sentence
                    break
    if not metadata["deadline"] and re.search(r"\b(?:submit|submission)\b", source, re.IGNORECASE):
        for line in text.splitlines():
            matches = list(DATE_PATTERN.finditer(line))
            if len(matches) >= 2 and re.search(r"\bto\b|[-–—]", line, re.IGNORECASE):
                metadata["deadline"] = matches[-1].group(0)
                metadata["deadline_source_sentence"] = line.strip()
                break

    if not metadata["semester"]:
        match = SEMESTER_PATTERN.search(source)
        if match:
            metadata["semester"] = re.sub(r"\s+", " ", match.group(0)).strip()

    if not metadata["department"]:
        if "concerned department" not in source.casefold():
            for match in DEPARTMENT_PATTERN.finditer(source):
                department = re.sub(r"\s+", " ", match.group(0)).strip()
                if not re.search(r"\b(?:schedule|submission|submitted)\b", department, re.IGNORECASE):
                    metadata["department"] = department
                    break

    if not metadata["required_action"]:
        for sentence in sentences:
            match = ACTION_PATTERN.search(sentence) or STUDENT_ACTION_PATTERN.search(sentence)
            if match:
                action = re.sub(r"\s+", " ", match.group(0)).strip()
                action = re.sub(r"^\d+[.)]\s*", "", action)
                action = re.sub(r"\s+\d{1,2}$", "", action)
                metadata["required_action"] = action
                break

    return metadata


def _valid_notice_date(value: str | None) -> bool:
    if not value:
        return False
    for date_format in (
        "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y", "%m/%d/%y", "%d.%m.%y", "%d-%m-%y",
    ):
        try:
            datetime.strptime(value.strip(), date_format)
            return True
        except ValueError:
            continue
    return False


def validate_notice(notice: dict) -> dict:
    """Flag incomplete or inconsistent records and score their evidence quality."""
    metadata = notice.get("metadata") or {}
    source = notice.get("source") or {}
    flags: list[str] = []
    score = 0.0

    if notice.get("title", "").strip():
        score += 0.15
    else:
        flags.append("missing_title")

    notice_date = metadata.get("date") or notice.get("date")
    if _valid_notice_date(notice_date):
        score += 0.20
    else:
        flags.append("missing_or_invalid_date")

    if notice.get("text", "").strip():
        score += 0.20
    else:
        flags.append("missing_text")

    if source.get("official_notice_url"):
        score += 0.16
    else:
        flags.append("missing_official_notice_url")

    if source.get("api_url"):
        score += 0.04
    else:
        flags.append("missing_api_url")

    fetched_at = source.get("fetched_at")
    try:
        parsed_fetched_at = datetime.fromisoformat(fetched_at) if fetched_at else None
        if parsed_fetched_at and parsed_fetched_at.tzinfo:
            score += 0.04
        else:
            flags.append("missing_or_invalid_fetched_at")
    except (TypeError, ValueError):
        flags.append("missing_or_invalid_fetched_at")

    expected_hash = hashlib.sha256(notice.get("text", "").encode("utf-8")).hexdigest()
    if notice.get("content_hash") == expected_hash:
        score += 0.15
    else:
        flags.append("missing_or_mismatched_content_hash")

    if metadata.get("category"):
        score += 0.06
    else:
        flags.append("missing_category")

    deadline = metadata.get("deadline")
    if deadline and not metadata.get("deadline_source_sentence"):
        flags.append("deadline_missing_source_sentence")
        score = max(0.0, score - 0.15)
    elif deadline and metadata["deadline"] not in metadata["deadline_source_sentence"]:
        flags.append("deadline_not_found_in_source_sentence")
        score = max(0.0, score - 0.15)
    elif deadline:
        deadline_date = _parse_notice_date(deadline)
        published_date = _parse_notice_date(notice_date)
        if deadline_date and published_date and deadline_date.date() < published_date.date():
            flags.append("deadline_precedes_notice_date")
            score = max(0.0, score - 0.15)

    notice["review_flags"] = flags
    notice["review_status"] = "needs_review" if flags else "ok"
    notice["extraction_confidence"] = round(max(0.0, min(1.0, score)), 2)
    return notice


def _topic_tokens(title: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", title.casefold())
        if len(token) > 2 and token not in TOPIC_STOPWORDS
    }


def link_notice_versions(notices: list[dict]) -> list[dict]:
    """Link only evidenced replacements; flag similar, unresolved notice pairs."""
    for notice in notices:
        notice["supersedes_id"] = None
        notice["superseded_by"] = None
        notice["version_status"] = "no_match"

    ambiguous_pairs: list[tuple[dict, dict]] = []
    for index, first in enumerate(notices):
        first_tokens = _topic_tokens(first.get("title", ""))
        for second in notices[index + 1 :]:
            second_tokens = _topic_tokens(second.get("title", ""))
            shared = first_tokens & second_tokens
            union = first_tokens | second_tokens
            topic_match = len(shared) >= 2 and bool(union) and len(shared) / len(union) >= 0.18
            if not topic_match:
                continue

            first_meta = first.get("metadata") or {}
            second_meta = second.get("metadata") or {}
            same_category_and_department = (
                first_meta.get("category")
                and second_meta.get("category")
                and first_meta.get("category") == second_meta.get("category")
                and first_meta.get("department")
                and second_meta.get("department")
                and first_meta.get("department") == second_meta.get("department")
            )
            replacement_evidence = bool(
                REPLACEMENT_CUE.search(first.get("title", "") + "\n" + first.get("text", ""))
            ) or bool(REPLACEMENT_CUE.search(second.get("title", "") + "\n" + second.get("text", "")))

            # Similar titles alone are common for separate notices (for example,
            # different students' presentations or different examination groups).
            # Consider them version candidates only with explicit update language,
            # or when category and department already establish the shared scope.
            if not replacement_evidence and not same_category_and_department:
                continue

            if replacement_evidence and same_category_and_department:
                first_date = first_meta.get("date") or first.get("date")
                second_date = second_meta.get("date") or second.get("date")
                first_parsed = _parse_notice_date(first_date)
                second_parsed = _parse_notice_date(second_date)
                if first_parsed and second_parsed and first_parsed != second_parsed:
                    older, newer = (first, second) if first_parsed < second_parsed else (second, first)
                    newer["supersedes_id"] = older.get("_id")
                    older["superseded_by"] = newer.get("_id")
                    older["version_status"] = "superseded"
                    newer["version_status"] = "supersedes"
                    continue

            ambiguous_pairs.append((first, second))

    for first, second in ambiguous_pairs:
        for notice, other in ((first, second), (second, first)):
            notice["version_status"] = "needs_review"
            notice.setdefault("version_review_candidates", []).append(other.get("_id"))
            flags = notice.setdefault("review_flags", [])
            if "version_relationship_ambiguous" not in flags:
                flags.append("version_relationship_ambiguous")
            notice["review_status"] = "needs_review"
            notice["extraction_confidence"] = round(
                max(0.0, notice.get("extraction_confidence", 0.0) - 0.10), 2
            )

    return notices


def add_ingestion_status(notice: dict) -> dict:
    """Record whether usable notice text was produced, with a failure reason."""
    if notice.get("extraction_status") in {"text_available", "extracted"} and notice.get("text", "").strip():
        notice["ingestion_status"] = "success"
        notice.pop("ingestion_failure_reason", None)
        return notice

    status = notice.get("extraction_status")
    if status == "needs_ocr":
        reason = "PDF has no extractable text; OCR is required."
    elif status == "pdf_error":
        error = notice.get("extraction_error", "unknown PDF extraction error")
        reason = f"PDF extraction failed: {error}"
    elif status == "no_text_or_pdf":
        reason = "Notice has no text and no direct PDF link."
    else:
        reason = "No usable notice text was produced."

    notice["ingestion_status"] = "failed"
    notice["ingestion_failure_reason"] = reason
    return notice


def _parse_notice_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for date_format in (
        "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%d.%m.%Y", "%d-%m-%Y", "%d/%m/%y", "%m/%d/%y", "%d.%m.%y", "%d-%m-%y",
    ):
        try:
            return datetime.strptime(value.strip(), date_format)
        except ValueError:
            continue
    return None


def process_notices(notices: list[dict]) -> list[dict]:
    """Process a batch independently so one malformed notice cannot stop it."""
    extracted = []
    for notice in notices:
        try:
            result = extract_notice_text(notice)
        except Exception as error:
            result = dict(notice)
            _, pdf_url = _text_and_pdf_link(notice)
            result["pdf_url"] = pdf_url
            result["text"] = ""
            result["extraction_status"] = "pdf_error"
            result["content_hash"] = hashlib.sha256(b"").hexdigest()
            result["metadata"] = extract_metadata(notice, "")
            result["extraction_error"] = str(error)
            result = validate_notice(result)
        extracted.append(add_ingestion_status(result))

    return link_notice_versions(extracted)


def extract_notice_text(notice: dict) -> dict:
    """Return a copy with cleaned text, extraction status, and PDF provenance."""
    result = dict(notice)
    existing_text, pdf_url = _text_and_pdf_link(notice)

    if existing_text.strip():
        result["text"] = clean_text(existing_text)
        result["extraction_status"] = "text_available"
        result["content_hash"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
        result["metadata"] = extract_metadata(notice, result["text"])
        return validate_notice(result)

    if not pdf_url:
        result["text"] = ""
        result["extraction_status"] = "no_text_or_pdf"
        result["content_hash"] = hashlib.sha256(b"").hexdigest()
        result["metadata"] = extract_metadata(notice, "")
        return validate_notice(result)

    result["pdf_url"] = pdf_url
    result["text"], result["extraction_status"] = _extract_pdf(pdf_url)
    result["content_hash"] = hashlib.sha256(result["text"].encode("utf-8")).hexdigest()
    result["metadata"] = extract_metadata(notice, result["text"])
    return validate_notice(result)


def main() -> None:
    notices = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    extracted = process_notices(notices)
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
