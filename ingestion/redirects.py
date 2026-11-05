"""Rule-based redirects for calculator and timetable questions."""

from __future__ import annotations

from dataclasses import dataclass
import re


INFOCASCADE_BASE_URL = "https://infocascadebuild.vercel.app"
REDIRECT_MESSAGE = "I can't provide this information through my AI model. Please visit this page for the relevant information:"
REDIRECT_CALCULATOR = "REDIRECT_CALCULATOR"
REDIRECT_TIMETABLE = "REDIRECT_TIMETABLE"


@dataclass(frozen=True)
class RedirectLink:
    label: str
    url: str


CALCULATOR_LINK = RedirectLink("Calculator", f"{INFOCASCADE_BASE_URL}/calculator")
TIMETABLE_LINK = RedirectLink("Timetable", f"{INFOCASCADE_BASE_URL}/timetable")

_CALCULATOR_TERM = re.compile(r"\b(?:sgpa|cgpa|gpa|grade\s+points?)\b", re.IGNORECASE)
_PERCENTAGE_INTENT = re.compile(
    r"(?:\b(?:calculate|calculating|calculation|compute|find|work\s+out|convert)\b.{0,40}\bpercentage\b"
    r"|\bpercentage\b.{0,40}\b(?:calculate|calculation|compute|find|work\s+out|convert)\b"
    r"|\bpercentage\s+(?:nikal(?:o|na)?|kitna|kitni|hua|batao)\b"
    r"|\bpercentage\s+kaise\s+nikal(?:o|na)?\b"
    r"|\bmera\s+percentage\s+(?:kitna|kitni)\b"
    r"|\bmarks\s+se\s+percentage\b"
    r"|\bconvert\s+marks\s+to\s+percentage\b)",
    re.IGNORECASE,
)
_TIMETABLE = re.compile(
    r"\b(?:timetable|time\s+table|my\s+class|next\s+class|which\s+class|"
    r"kaun\s+si\s+class|class\s+kab|lecture\s+today|mera\s+timetable)\b"
    r"|\bclass\s+at\s+\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?\b",
    re.IGNORECASE,
)


def route_question(question: str) -> tuple[str, list[RedirectLink]] | None:
    """Return matching redirect routes and links, or None for notice flow."""
    calculator = bool(_CALCULATOR_TERM.search(question) or _PERCENTAGE_INTENT.search(question))
    timetable = bool(_TIMETABLE.search(question))
    if not calculator and not timetable:
        return None
    routes: list[str] = []
    links: list[RedirectLink] = []
    if calculator:
        routes.append("calculator")
        links.append(CALCULATOR_LINK)
    if timetable:
        routes.append("timetable")
        links.append(TIMETABLE_LINK)
    return "+".join(routes), links


def redirect_response(question: str) -> dict | None:
    match = route_question(question)
    if match is None:
        return None
    route, links = match
    return {
        "route": route,
        "answer": REDIRECT_MESSAGE + "\n" + "\n".join(link.url for link in links),
        "links": [{"label": link.label, "url": link.url} for link in links],
    }


def marker_response(marker: str) -> dict | None:
    route_links = {
        REDIRECT_CALCULATOR: ("calculator", [CALCULATOR_LINK]),
        REDIRECT_TIMETABLE: ("timetable", [TIMETABLE_LINK]),
    }
    selected = route_links.get(marker)
    if selected is None:
        return None
    route, links = selected
    return {
        "route": route,
        "answer": REDIRECT_MESSAGE + "\n" + "\n".join(link.url for link in links),
        "links": [{"label": link.label, "url": link.url} for link in links],
    }
