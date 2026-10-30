"""FastAPI application exposing notice question answering."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ingestion.answer_chain import build_answer_chain


app = FastAPI(title="Infochat API")
MAX_QUESTION_LENGTH = 1000
_answer_chain = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)


class SourceCard(BaseModel):
    notice_id: str | None = None
    title: str | None = None
    published_at: str | None = None
    source_url: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCard]
    status: Literal["answered", "not_found", "invalid_citation"]


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """Answer a question from current notices and return its source cards."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    global _answer_chain
    try:
        if _answer_chain is None:
            _answer_chain = build_answer_chain()
        result = await _answer_chain.ainvoke({"question": question})
    except Exception as error:
        # Keep provider credentials and upstream error details out of responses.
        raise HTTPException(
            status_code=502,
            detail="The answer service is temporarily unavailable.",
        ) from error

    answer = result.get("answer", "not found")
    sources = result.get("sources", [])
    status = result.get("status", "not_found" if answer == "not found" else "answered")
    return AskResponse(answer=answer, sources=sources, status=status)
