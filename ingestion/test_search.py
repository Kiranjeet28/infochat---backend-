"""Read-only similarity and full-text checks for CascadeIQ notice chunks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding


load_dotenv(Path(__file__).resolve().parent.parent / ".env")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
TOP_K = 5
REQUEST_TIMEOUT_SECONDS = 30
KEYWORD_RPC = "search_chunks_keyword"
SIMILARITY_RPC = "search_chunks_similarity"

TESTS = [
    {
        "question": "What was announced about the provisional results of the end semester examinations in May 2026?",
        "keyword": "provisional",
        "expected": "The provisional results of the End Semester Examinations May 2026",
        "related": [
            "The provisional result of B. Tech. Final year for ESE May-2026",
            "The provisional result of BBA and BCA Final year (Regular) for ESE May-2026",
        ],
    },
    {
        "question": "What schedule was published for the May-2026 makeup examination?",
        "keyword": "Makeup",
        "expected": "Schedule for End Semester Makeup Examination May-2026.",
    },
    {
        "question": "Which exam centre was combined with another one?",
        "keyword": "C1M",
        "expected": "Merger of Examination Centres for End Semester Examinations May 2026",
    },
    {
        "question": "When is Ritu's six-month PhD progress presentation?",
        "keyword": "Ritu",
        "expected": "Six Months Progress Presentation of Ms. Ritu, PhD. Regd. No.1903009, URN 1935251",
    },
    {
        "question": "When is Gitanjali Sharda's six-month PhD progress presentation?",
        "keyword": "Gitanjali",
        "expected": "Six Months Progress Presentation of Ms. Gitanjali Sharda PhD Research Scholar.",
    },
    {
        "question": "Which students were struck off the rolls?",
        "keyword": "struck",
        "expected": "Struck_off_Notice_42_07.07.26",
    },
    {
        "question": "What is the hostel fee for the next session?",
        "keyword": "hostel",
        "expected": None,
    },
]


def _credentials() -> tuple[str, dict[str, str]]:
    project_url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
    if not project_url or not service_role_key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env.")
    return project_url, {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
    }


def _parse_vector(value: Any) -> list[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        raise ValueError("Stored embedding is not a vector array")
    vector = [float(item) for item in value]
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"Expected {EMBEDDING_DIMENSIONS} dimensions, got {len(vector)}")
    return vector


def _fetch_chunks(
    session: requests.Session, project_url: str, headers: dict[str, str]
) -> list[dict[str, Any]]:
    response = session.get(
        f"{project_url}/rest/v1/chunks",
        params={
            "select": "chunk_id,embedding,embedding_model",
            "order": "chunk_id.asc",
            "limit": 1000,
        },
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _similarity_search(
    session: requests.Session,
    project_url: str,
    headers: dict[str, str],
    query_vector: list[float],
) -> list[dict[str, Any]]:
    response = session.post(
        f"{project_url}/rest/v1/rpc/{SIMILARITY_RPC}",
        headers=headers,
        json={"query_embedding": query_vector, "match_count": TOP_K},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _keyword_search(
    session: requests.Session,
    project_url: str,
    headers: dict[str, str],
    keyword: str,
) -> list[dict[str, Any]]:
    response = session.post(
        f"{project_url}/rest/v1/rpc/{KEYWORD_RPC}",
        headers=headers,
        json={"search_query": keyword, "match_count": TOP_K},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def _rank(rows: list[dict[str, Any]], title: str | None) -> int | None:
    if title is None:
        return None
    for index, row in enumerate(rows, start=1):
        if row.get("title") == title:
            return index
    return None


def _print_results(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"  {label} top {TOP_K}:")
    for rank, row in enumerate(rows, start=1):
        print(
            f"    {rank}. {row.get('title')} | {row.get('published_at')} | "
            f"{row.get('source_url')} | cosine_distance={row.get('score', 0.0):.6f}"
        )
    if not rows:
        print("    (no results)")


def _rank_label(rank: int | None) -> str:
    return str(rank) if rank is not None else "not in top 5"


def run() -> None:
    project_url, headers = _credentials()
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    with requests.Session() as session:
        chunks = _fetch_chunks(session, project_url, headers)
        if not chunks:
            raise RuntimeError("No chunks were returned from Supabase")
        for chunk in chunks:
            if chunk.get("embedding_model") != EMBEDDING_MODEL:
                raise ValueError(
                    f"Unexpected stored embedding model: {chunk.get('embedding_model')}"
                )
            if chunk.get("embedding") is None:
                raise ValueError(f"Chunk {chunk.get('chunk_id')} has a null embedding")
            _parse_vector(chunk["embedding"])

        summaries = []
        for test in TESTS:
            question = test["question"]
            query_vector = next(model.query_embed(question)).tolist()
            if len(query_vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"Expected {EMBEDDING_DIMENSIONS} query dimensions, got {len(query_vector)}"
                )

            similarity_rows: list[dict[str, Any]] = []
            similarity_error: str | None = None
            try:
                similarity_rows = _similarity_search(
                    session, project_url, headers, query_vector
                )
            except requests.RequestException as error:
                detail = error.response.text[:500] if error.response is not None else str(error)
                similarity_error = f"{type(error).__name__}: {detail}"

            keyword_rows: list[dict[str, Any]] = []
            keyword_error: str | None = None
            try:
                keyword_rows = _keyword_search(
                    session, project_url, headers, test["keyword"]
                )
            except requests.RequestException as error:
                detail = error.response.text[:500] if error.response is not None else str(error)
                keyword_error = f"{type(error).__name__}: {detail}"

            print(f"\nQuestion: {question}")
            if similarity_error:
                print(f"  Similarity search unavailable: {similarity_error}")
            else:
                _print_results("Similarity", similarity_rows)
            if keyword_error:
                print(f"  Keyword search unavailable: {keyword_error}")
            else:
                _print_results("Keyword", keyword_rows)

            related_ranks = {
                title: _rank(similarity_rows, title) if not similarity_error else None
                for title in test.get("related", [])
            }
            summaries.append(
                {
                    "question": question,
                    "expected": test["expected"] or "No matching notice",
                    "similarity_rank": _rank(similarity_rows, test["expected"])
                    if not similarity_error
                    else None,
                    "keyword_rank": _rank(keyword_rows, test["expected"])
                    if not keyword_error
                    else None,
                    "related_ranks": related_ranks,
                    "keyword_error": keyword_error,
                    "similarity_error": similarity_error,
                }
            )

    print("\nSummary ranks:")
    print("question | expected notice | similarity rank | keyword rank")
    for item in summaries:
        print(
            f"{item['question']} | {item['expected']} | "
            f"{_rank_label(item['similarity_rank']) if not item['similarity_error'] else 'unavailable'} | "
            f"{_rank_label(item['keyword_rank']) if not item['keyword_error'] else 'unavailable'}"
        )
        for title, rank in item["related_ranks"].items():
            label = _rank_label(rank) if not item["similarity_error"] else "unavailable"
            print(f"  related: {title} | similarity rank {label}")
        if item["keyword_error"]:
            print(f"  keyword failure: {item['keyword_error']}")
        if item["similarity_error"]:
            print(f"  similarity failure: {item['similarity_error']}")


if __name__ == "__main__":
    run()
