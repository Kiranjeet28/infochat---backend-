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
            "select": "chunk_id,title,published_at,source_url,embedding,embedding_model",
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
    if response.status_code >= 400:
        if "PGRST202" in response.text:
            raise RuntimeError(
                "PGRST202 from search_chunks_similarity; called with "
                "query_embedding: extensions.vector(384), match_count: integer. "
                f"Exact response: {response.status_code} {response.text}"
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
    if response.status_code >= 400:
        if "PGRST202" in response.text:
            raise RuntimeError(
                "PGRST202 from search_chunks_keyword; called with "
                "search_query: text, match_count: integer. "
                f"Exact response: {response.status_code} {response.text}"
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


def _print_results(
    label: str, rows: list[dict[str, Any]], score_label: str = "score"
) -> None:
    print(f"  {label} top {TOP_K}:")
    for rank, row in enumerate(rows, start=1):
        print(
            f"    {rank}. {row.get('title')} | {row.get('published_at')} | "
            f"{row.get('source_url')} | {score_label}={row.get('score', 0.0):.6f}"
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

            similarity_rows = _similarity_search(
                session, project_url, headers, query_vector
            )
            keyword_rows = _keyword_search(
                session, project_url, headers, test["keyword"]
            )

            print(f"\nQuestion: {question}")
            _print_results("Similarity (database pgvector)", similarity_rows, "cosine_distance")
            _print_results("Keyword", keyword_rows, "ts_rank_cd")

            related_ranks = {
                title: {
                    "similarity": _rank(similarity_rows, title),
                    "keyword": _rank(keyword_rows, title),
                }
                for title in test.get("related", [])
            }
            expected_row = next(
                (row for row in similarity_rows if row.get("title") == test["expected"]),
                None,
            )
            summaries.append(
                {
                    "question": question,
                    "expected": test["expected"] or "No matching notice",
                    "similarity_rank": _rank(similarity_rows, test["expected"]),
                    "keyword_rank": _rank(keyword_rows, test["expected"]),
                    "expected_distance": expected_row.get("score") if expected_row else None,
                    "similarity_rows": similarity_rows,
                    "related_ranks": related_ranks,
                }
            )

    print("\nSummary ranks:")
    print("question | expected notice | similarity rank | keyword rank")
    for item in summaries:
        print(
            f"{item['question']} | {item['expected']} | "
            f"{_rank_label(item['similarity_rank'])} | "
            f"{_rank_label(item['keyword_rank'])}"
        )
        for title, ranks in item["related_ranks"].items():
            print(
                f"  related: {title} | similarity rank {_rank_label(ranks['similarity'])} | "
                f"keyword rank {_rank_label(ranks['keyword'])}"
            )

    print("\nCosine distances:")
    print("case | notice/result | cosine distance")
    for item in summaries:
        if item["expected"] == "No matching notice":
            for rank, row in enumerate(item["similarity_rows"], start=1):
                print(f"hostel top {rank} | {row.get('title')} | {row.get('score'):.6f}")
        else:
            distance = item["expected_distance"]
            rendered = f"{distance:.6f}" if distance is not None else "not in top 5"
            print(f"expected | {item['expected']} | {rendered}")


if __name__ == "__main__":
    run()
