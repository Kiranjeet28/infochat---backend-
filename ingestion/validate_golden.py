"""Validate CascadeIQ's offline golden-set definition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ALLOWED_TYPES = {"answerable", "near_miss", "no_answer", "paraphrase", "redirect", "adversarial"}
ALLOWED_STATUSES = {"answered", "not_found", "redirect"}
REQUIRED_FIELDS = {
    "id", "question", "type", "expected_status", "must_contain_facts",
    "expected_notice_titles", "notes",
}


def validate_payload(payload: Any) -> tuple[list[str], list[str], dict[str, int]]:
    errors: list[str] = []
    warnings: list[str] = []
    counts = {kind: 0 for kind in sorted(ALLOWED_TYPES)}
    if not isinstance(payload, list):
        return ["top-level JSON value must be a list"], warnings, counts
    seen: set[str] = set()
    for index, case in enumerate(payload):
        label = f"case[{index}]"
        if not isinstance(case, dict):
            errors.append(f"{label} must be an object")
            continue
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            errors.append(f"{label}.id must be a non-empty string")
        elif case_id in seen:
            errors.append(f"duplicate id: {case_id}")
        else:
            seen.add(case_id)
        missing = REQUIRED_FIELDS - case.keys()
        if missing:
            errors.append(f"{label} missing fields: {', '.join(sorted(missing))}")
            continue
        if not isinstance(case["question"], str) or not case["question"].strip():
            errors.append(f"{label}.question must be a non-empty string")
        kind = case["type"]
        if not isinstance(kind, str) or kind not in ALLOWED_TYPES:
            errors.append(f"{label}.type is not allowed: {kind!r}")
        else:
            counts[kind] += 1
        if not isinstance(case["expected_status"], str) or case["expected_status"] not in ALLOWED_STATUSES:
            errors.append(f"{label}.expected_status is not allowed: {case['expected_status']!r}")
        for field in ("must_contain_facts", "expected_notice_titles"):
            value = case[field]
            if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
                errors.append(f"{label}.{field} must be a list of strings")
        if not isinstance(case["notes"], str):
            errors.append(f"{label}.notes must be a string")
        if "verified" in case and not isinstance(case["verified"], bool):
            errors.append(f"{label}.verified must be true or false")
        if kind == "answerable" and isinstance(case["must_contain_facts"], list) and not case["must_contain_facts"]:
            warnings.append(f"{case_id}: answerable case has no must_contain_facts")
    unverified = sum(1 for case in payload if isinstance(case, dict) and case.get("verified") is False)
    if unverified:
        warnings.append(f"{unverified} of {len(payload)} cases are unverified (verified: false); check their facts before trusting results")
    if len(payload) < 50:
        warnings.append(f"golden set has {len(payload)} cases; recommended minimum is 50")
    return errors, warnings, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).parent / "golden" / "golden_set.json")
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"Invalid golden set: {type(error).__name__}")
        return 1
    errors, warnings, counts = validate_payload(payload)
    print(f"Cases: {len(payload) if isinstance(payload, list) else 0}")
    for kind, count in counts.items():
        print(f"{kind}: {count}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    return 2 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
