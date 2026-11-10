"""Answer-time privacy settings for notice evidence."""

import os

# Matching chunks are replaced with a privacy-safe synopsis before model input.
MODEL_INPUT_EXCLUDED_NOTICE_IDS: tuple[str, ...] = ()
MODEL_INPUT_EXCLUDED_CATEGORIES: tuple[str, ...] = ("struck_off", "struck off")
MODEL_INPUT_EXCLUDED_TITLE_MARKERS: tuple[str, ...] = ("struck_off_notice", "struck off notice")

# Notices whose title or category matches are withheld from the model like struck-off notices.
# Override with RESTRICTED_NOTICE_PATTERNS (comma separated, case-insensitive).
DEFAULT_RESTRICTED_NOTICE_PATTERNS: tuple[str, ...] = ("struck off", "list of oe", "readmission")


def normalize_marker(value: str) -> str:
    return " ".join(value.casefold().replace("_", " ").replace("-", " ").split())


def restricted_notice_patterns() -> tuple[str, ...]:
    configured = os.environ.get("RESTRICTED_NOTICE_PATTERNS")
    values = configured.split(",") if configured is not None else DEFAULT_RESTRICTED_NOTICE_PATTERNS
    return tuple(p for p in (normalize_marker(v) for v in values) if p)


FRIENDLY_NOT_FOUND_MESSAGE = "The available notices do not establish this."
FRIENDLY_INVALID_CITATION_MESSAGE = "I could not verify an answer from the notices."


def plain_text_answers_enabled() -> bool:
    """Return whether API answers should be rendered without Markdown syntax."""
    return os.environ.get("CASCADEIQ_PLAIN_TEXT_ANSWERS", "false").strip().casefold() in {
        "1", "true", "yes", "on",
    }
