from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import RAGError
from app.core.logging import logger
from app.services.embedding_service import EmbeddingService
from app.vectorstore.base import BaseVectorStore, VectorSearchResult
from app.vectorstore.chroma_store import ChromaStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int]
    content: str
    similarity_score: float
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    query: str
    chunks: List[RetrievedChunk]
    has_sufficient_context: bool
    highest_similarity: float
    total_candidates_found: int


class RetrievalService:
    """Retrieval engine: converts query to embedding, queries vector store, filters by score."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or ChromaStore(settings=self.settings)
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.similarity_threshold = self.settings.RAG_SIMILARITY_THRESHOLD
        self.default_top_k = self.settings.RAG_TOP_K

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """Retrieves and filters most relevant chunks for a given query."""
        k = top_k or self.default_top_k
        min_threshold = threshold if threshold is not None else self.similarity_threshold

        try:
            # 1. Embed query vector
            query_embedding = self.embedding_service.embed_text(query)

            # 2. Vector search in ChromaDB
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=k,
                filter_metadata=filter_metadata,
            )

            if not search_results:
                return RetrievalResult(
                    query=query,
                    chunks=[],
                    has_sufficient_context=False,
                    highest_similarity=0.0,
                    total_candidates_found=0,
                )

            # 3. Transform to RetrievedChunk domain model and filter by similarity threshold
            chunks: List[RetrievedChunk] = []
            highest_sim = search_results[0].similarity_score if search_results else 0.0

            for res in search_results:
                # Include chunk if it meets minimum similarity
                if res.similarity_score >= min_threshold:
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=res.id,
                            document_id=res.metadata.get("document_id", "unknown"),
                            filename=res.metadata.get("filename", "unknown"),
                            page_number=res.metadata.get("page_number"),
                            content=res.text,
                            similarity_score=res.similarity_score,
                            metadata=res.metadata,
                        )
                    )

            has_sufficient = len(chunks) > 0 and highest_sim >= min_threshold

            logger.info(
                f"Retrieved {len(chunks)} chunks for query (top_score={highest_sim:.3f}, threshold={min_threshold}, sufficient={has_sufficient})"
            )

            return RetrievalResult(
                query=query,
                chunks=chunks,
                has_sufficient_context=has_sufficient,
                highest_similarity=highest_sim,
                total_candidates_found=len(search_results),
            )
        except Exception as exc:
            if isinstance(exc, RAGError):
                raise
            raise RAGError(f"Retrieval operation failed: {exc}") from exc

