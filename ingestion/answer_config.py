"""Answer-time privacy settings for notice evidence."""

# Matching chunks are replaced with a privacy-safe synopsis before model input.
MODEL_INPUT_EXCLUDED_NOTICE_IDS: tuple[str, ...] = ()
MODEL_INPUT_EXCLUDED_CATEGORIES: tuple[str, ...] = ("struck_off", "struck off")
MODEL_INPUT_EXCLUDED_TITLE_MARKERS: tuple[str, ...] = ("struck_off_notice", "struck off notice")

FRIENDLY_NOT_FOUND_MESSAGE = "The available notices do not establish this."
FRIENDLY_INVALID_CITATION_MESSAGE = "I could not verify an answer from the notices."
