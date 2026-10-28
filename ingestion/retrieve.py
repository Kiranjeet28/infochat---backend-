"""Hybrid semantic and keyword retrieval for indexed notice chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60
REQUEST_TIMEOUT_SECONDS = 30

_embedding_model: TextEmbedding | None = None


def retrieve(
    question: str,
    department: str | None = None,
    semester: str | None = None,
) -> list[dict[str, Any]]:
    """Return the five best chunks using reciprocal rank fusion.

    Requires ``SUPABASE_URL`` and ``SUPABASE_SERVICE_ROLE_KEY`` in the
    environment (or the repository's local ``.env`` file).
    """
    global _embedding_model

    question = question.strip()
    if not question:
        raise ValueError("question must not be empty")

    project_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not project_url or not service_role_key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.")

    if _embedding_model is None:
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    query_vector = next(_embedding_model.embed([question])).tolist()

    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
    }
    with requests.Session() as session:
        similarity_response = session.post(
            f"{project_url}/rest/v1/rpc/search_chunks_similarity",
            headers=headers,
            json={
                "query_embedding": query_vector,
                "match_count": TOP_K,
                "filter_department": department,
                "filter_semester": semester,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        similarity_response.raise_for_status()
        similarity_rows = similarity_response.json()

        keyword_response = session.post(
            f"{project_url}/rest/v1/rpc/search_chunks_keyword",
            headers=headers,
            json={
                "search_query": question,
                "match_count": TOP_K,
                "filter_department": department,
                "filter_semester": semester,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        keyword_response.raise_for_status()
        keyword_rows = keyword_response.json()

    fused: dict[str, dict[str, Any]] = {}
    for rows in (similarity_rows, keyword_rows):
        for rank, row in enumerate(rows, start=1):
            chunk_id = row.get("chunk_id")
            if chunk_id is None:
                continue
            item = fused.setdefault(chunk_id, {**row, "rrf_score": 0.0})
            item["rrf_score"] += 1.0 / (RRF_K + rank)

    return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:TOP_K]
