"""
In-memory document store — orchestrates ingest → index → ask.

This replaces the React state (`documents`, `corpusIdf`, `flatChunks`) with a
simple Python object both FastAPI and Flask can share.
"""

from __future__ import annotations

from typing import Any

from .gemini_client import answer_question, answer_question_sync
from .pdf_parser import ingest_file
from .retrieval import resolve_retrieval_query, retrieve_relevant_chunks
from .tfidf import build_tfidf_index_for_chunks


class DocumentStore:
    """Session-scoped corpus: documents + shared IDF + flat chunk list."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.corpus_idf: dict[str, float] = {}
        self.flat_chunks: list[dict[str, Any]] = []

    def clear(self) -> None:
        self.documents.clear()
        self.corpus_idf = {}
        self.flat_chunks = []

    def _reindex(self) -> None:
        """
        Rebuild IDF over *all* chunks.

        Why full rebuild (not incremental):
          IDF depends on N and every term's DF. Adding one PDF changes weights
          for the whole corpus, so the JS app also re-ran buildTfidfIndexForChunks
          on the flattened list after each ingest/merge.
        """
        self.flat_chunks = [c for d in self.documents for c in d["chunks"]]
        self.corpus_idf = build_tfidf_index_for_chunks(self.flat_chunks)

    def add_file(
        self,
        filename: str,
        data: bytes,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        doc = ingest_file(filename, data, content_type)
        self.documents.append(doc)
        self._reindex()
        return {
            "id": doc["id"],
            "name": doc["name"],
            "chunk_count": len(doc["chunks"]),
            "page_count": doc["page_count"],
            "total_chunks": len(self.flat_chunks),
        }

    def stats(self) -> dict[str, Any]:
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.flat_chunks),
            "vocabulary_size": len(self.corpus_idf),
            "documents": [
                {
                    "id": d["id"],
                    "name": d["name"],
                    "chunk_count": len(d["chunks"]),
                    "page_count": d.get("page_count"),
                }
                for d in self.documents
            ],
        }

    def retrieve(self, query: str, top_k: int = 7) -> dict[str, Any]:
        return retrieve_relevant_chunks(
            query, self.flat_chunks, top_k=top_k, idf_map=self.corpus_idf
        )

    async def ask(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int = 7,
    ) -> dict[str, Any]:
        """
        End-to-end Q&A: resolve follow-up → retrieve → (maybe) Gemini.

        If nothing clears the similarity floor, skip the LLM and return the
        same out-of-scope message the UI used — saves quota and avoids
        hallucinated "I couldn't find…" after the model sees empty context.
        """
        history = chat_history or []
        resolved = resolve_retrieval_query(question, history)
        retrieval = self.retrieve(resolved["retrievalQuery"], top_k=top_k)
        results = retrieval["results"]
        confidence = retrieval["confidence"]

        if not results:
            return {
                "answer": (
                    "I couldn't find information about this in the uploaded "
                    "document(s)."
                ),
                "sources": [],
                "confidence": confidence,
                "confidence_tier": "out_of_scope",
                "is_follow_up": resolved["isFollowUp"],
            }

        # Strip to fields Gemini + the UI need (avoid shipping tfidf blobs).
        chunk_payloads = [
            {
                "text": r["chunk"]["text"],
                "docName": r["chunk"]["docName"],
                "page": r["chunk"]["page"],
                "similarity": r["similarity"],
                "chunkIndex": r["chunk"]["chunkIndex"],
            }
            for r in results
        ]

        answer = await answer_question(
            question,
            chunk_payloads,
            history,
            is_follow_up=resolved["isFollowUp"],
        )

        avg = confidence["avg"]
        if avg > 0.3:
            tier = "high"
        elif avg >= 0.1:
            tier = "medium"
        else:
            tier = "low"

        return {
            "answer": answer,
            # UI historically showed the top 3 sources.
            "sources": chunk_payloads[:3],
            "retrieved": chunk_payloads,
            "confidence": confidence,
            "confidence_tier": tier,
            "is_follow_up": resolved["isFollowUp"],
        }

    def ask_sync(
        self,
        question: str,
        chat_history: list[dict[str, str]] | None = None,
        top_k: int = 7,
    ) -> dict[str, Any]:
        """Flask-friendly sync path (same logic, blocking Gemini call)."""
        history = chat_history or []
        resolved = resolve_retrieval_query(question, history)
        retrieval = self.retrieve(resolved["retrievalQuery"], top_k=top_k)
        results = retrieval["results"]
        confidence = retrieval["confidence"]

        if not results:
            return {
                "answer": (
                    "I couldn't find information about this in the uploaded "
                    "document(s)."
                ),
                "sources": [],
                "confidence": confidence,
                "confidence_tier": "out_of_scope",
                "is_follow_up": resolved["isFollowUp"],
            }

        chunk_payloads = [
            {
                "text": r["chunk"]["text"],
                "docName": r["chunk"]["docName"],
                "page": r["chunk"]["page"],
                "similarity": r["similarity"],
                "chunkIndex": r["chunk"]["chunkIndex"],
            }
            for r in results
        ]
        answer = answer_question_sync(
            question,
            chunk_payloads,
            history,
            is_follow_up=resolved["isFollowUp"],
        )
        avg = confidence["avg"]
        tier = "high" if avg > 0.3 else "medium" if avg >= 0.1 else "low"
        return {
            "answer": answer,
            "sources": chunk_payloads[:3],
            "retrieved": chunk_payloads,
            "confidence": confidence,
            "confidence_tier": tier,
            "is_follow_up": resolved["isFollowUp"],
        }
