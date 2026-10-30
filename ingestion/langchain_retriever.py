"""LangChain adapter for the hybrid notice retrieval function."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ingestion.retrieve import retrieve


class NoticeRetriever(BaseRetriever):
    """Retrieve current notice chunks as LangChain documents."""

    department: str | None = None
    semester: str | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = retrieve(
            query,
            department=self.department,
            semester=self.semester,
        )
        if isinstance(results, str):  # retrieve returns this for "not found".
            return []

        documents: list[Document] = []
        for row in results:
            metadata: dict[str, Any] = {
                key: row.get(key)
                for key in (
                    "notice_id",
                    "title",
                    "published_at",
                    "source_url",
                    "category",
                    "chunk_id",
                    "department",
                    "semester",
                    "rrf_score",
                    "similarity_distance",
                    "keyword_score",
                )
                if row.get(key) is not None
            }
            documents.append(
                Document(page_content=row.get("chunk_text") or "", metadata=metadata)
            )
        return documents
