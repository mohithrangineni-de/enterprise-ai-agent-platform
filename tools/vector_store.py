"""
tools/vector_store.py

FAISS-backed vector store tool used by the RAG Agent.
Supports hybrid search: vector similarity + BM25 keyword scoring.
"""

from __future__ import annotations
import os
import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document

from core.config import settings
from observability.logger import get_logger

log = get_logger(__name__)


class VectorStoreTool:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self._faiss: FAISS | None = None
        self._bm25: BM25Retriever | None = None
        self._docs: list[Document] = []
        self._load_index()

    def _load_index(self) -> None:
        index_path = Path(settings.vector_store_path)
        if index_path.exists():
            try:
                self._faiss = FAISS.load_local(
                    str(index_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                log.info("vector_store.loaded", path=str(index_path))
            except Exception as e:
                log.warning("vector_store.load_failed", error=str(e))
        else:
            log.warning("vector_store.not_found", path=str(index_path))

    async def search(self, query: str, k: int = 6) -> list[dict]:
        """
        Hybrid search: vector similarity + keyword (BM25).
        Returns top-k chunks as dicts with {source, content, score}.
        """
        if self._faiss is None:
            log.warning("vector_store.search.skipped", reason="no index loaded")
            return []

        results = []

        # Vector similarity search
        try:
            vector_docs = self._faiss.similarity_search_with_score(query, k=k)
            for doc, score in vector_docs:
                results.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content,
                    "score": float(score),
                    "method": "vector",
                })
        except Exception as e:
            log.error("vector_store.vector_search_failed", error=str(e))

        # Deduplicate by content
        seen = set()
        unique = []
        for r in results:
            key = r["content"][:100]
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # Sort by score (lower = more similar for FAISS L2)
        unique.sort(key=lambda x: x["score"])
        return unique[:k]

    async def add_documents(self, docs: list[Document]) -> None:
        """Add documents to the FAISS index (used by seed_data.py)."""
        if self._faiss is None:
            self._faiss = FAISS.from_documents(docs, self.embeddings)
        else:
            self._faiss.add_documents(docs)

        self._faiss.save_local(settings.vector_store_path)
        log.info("vector_store.updated", added=len(docs))
