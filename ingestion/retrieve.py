"""Hybrid semantic and keyword retrieval for indexed notice chunks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from ingestion.embedding_config import EMBEDDING_MODEL, create_embedding_model


load_dotenv(Path(__file__).resolve().parent.parent / ".env")

TOP_K = 5
RRF_K = 60
SIMILARITY_DISTANCE_CUTOFF = 0.65
REQUEST_TIMEOUT_SECONDS = 30

_embedding_model: Any | None = None


def retrieve(
    question: str,
    department: str | None = None,
    semester: str | None = None,
    *,
    diagnostics: bool = False,
) -> list[dict[str, Any]] | str | dict[str, Any]:
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
        _embedding_model = create_embedding_model()
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
        if similarity_response.status_code >= 400:
            _raise_rpc_error("search_chunks_similarity", similarity_response,
                             {"query_embedding", "match_count", "filter_department", "filter_semester"})
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
        if keyword_response.status_code >= 400:
            _raise_rpc_error("search_chunks_keyword", keyword_response,
                             {"search_query", "match_count", "filter_department", "filter_semester"})
        keyword_rows = keyword_response.json()

    best_distance = similarity_rows[0].get("score") if similarity_rows else None
    similarity_distances: dict[Any, float] = {}
    for row in similarity_rows:
        chunk_id = row.get("chunk_id")
        distance = row.get("score")
        if chunk_id is not None and distance is not None:
            similarity_distances[chunk_id] = float(distance)

    fused: dict[str, dict[str, Any]] = {}
    for search_type, rows in (("similarity", similarity_rows), ("keyword", keyword_rows)):
        for rank, row in enumerate(rows, start=1):
            chunk_id = row.get("chunk_id")
            if chunk_id is None:
                continue
            item = fused.setdefault(chunk_id, {**row, "rrf_score": 0.0})
            if search_type == "similarity":
                item["similarity_distance"] = row.get("score")
            else:
                item["keyword_score"] = row.get("score")
            item["rrf_score"] += 1.0 / (RRF_K + rank)

    # A result must have a vector distance below the cutoff, even when it was
    # also found by keyword. Keyword-only rows have no measured distance.
    eligible = [
        row for chunk_id, row in fused.items()
        if similarity_distances.get(chunk_id, float("inf")) < SIMILARITY_DISTANCE_CUTOFF
    ]
    results = sorted(eligible, key=lambda item: item["rrf_score"], reverse=True)[:TOP_K]
    not_found = not results
    if diagnostics:
        return {
            "results": results,
            "best_distance": best_distance,
            "keyword_found": bool(keyword_rows),
            "not_found": not_found,
        }
    return "not found" if not_found else results


def _raise_rpc_error(function: str, response: requests.Response, parameters: set[str]) -> None:
    """Raise useful PostgREST errors without ever exposing request credentials."""
    try:
        payload = response.json()
        code = payload.get("code") if isinstance(payload, dict) else None
    except ValueError:
        payload, code = None, None
    if code in {"PGRST202", "PGRST203"}:
        raise RuntimeError(
            f"{code} from {function}; parameter names sent: {', '.join(sorted(parameters))}. "
            f"Exact response: {response.status_code} {response.text}"
        )
    response.raise_for_status()
