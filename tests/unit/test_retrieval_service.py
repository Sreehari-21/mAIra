from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from app.core.config import Settings
from app.services.retrieval_service import RetrievalService
from app.vectorstore.base import VectorSearchResult


def test_retrieval_service_filters_by_threshold():
    settings = Settings(RAG_SIMILARITY_THRESHOLD=0.5, RAG_TOP_K=3)
    mock_store = MagicMock()
    mock_embedder = MagicMock()

    mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
    mock_store.similarity_search.return_value = [
        VectorSearchResult(
            id="doc1#c00",
            text="High similarity chunk",
            metadata={"document_id": "doc1", "filename": "spec.txt", "page_number": 1},
            distance=0.2,
            similarity_score=0.8,
        ),
        VectorSearchResult(
            id="doc1#c01",
            text="Low similarity chunk",
            metadata={"document_id": "doc1", "filename": "spec.txt", "page_number": 2},
            distance=0.7,
            similarity_score=0.3,  # Below threshold 0.5
        ),
    ]

    service = RetrievalService(
        vector_store=mock_store,
        embedding_service=mock_embedder,
        settings=settings,
    )

    result = service.retrieve("How does the system work?")
    assert result.has_sufficient_context is True
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "doc1#c00"
    assert result.chunks[0].similarity_score == 0.8

