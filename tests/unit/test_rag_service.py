from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from app.api.schemas.chat import ChatRequest
from app.core.config import Settings
from app.services.rag_service import RAGService
from app.services.retrieval_service import RetrievalResult, RetrievedChunk


def test_rag_service_grounded_answer():
    mock_retrieval = MagicMock()
    mock_llm = MagicMock()

    chunk = RetrievedChunk(
        chunk_id="doc1#c00",
        document_id="doc1",
        filename="platform.txt",
        page_number=1,
        content="Phase 1 uses Ollama and qwen2.5:7b exclusively.",
        similarity_score=0.92,
        metadata={},
    )
    mock_retrieval.retrieve.return_value = RetrievalResult(
        query="What runtime is used?",
        chunks=[chunk],
        has_sufficient_context=True,
        highest_similarity=0.92,
        total_candidates_found=1,
    )

    mock_llm.generate_response.return_value = {
        "text": "Phase 1 uses Ollama with qwen2.5:7b [Document: platform.txt, Page: 1 | Chunk ID: doc1#c00].",
        "model": "qwen2.5:7b",
    }

    rag = RAGService(retrieval_service=mock_retrieval, llm_service=mock_llm)
    req = ChatRequest(query="What runtime is used?")
    res = rag.answer_query(req)

    assert res.is_grounded is True
    assert res.has_sufficient_context is True
    assert len(res.sources) == 1
    assert res.sources[0].chunk_id == "doc1#c00"
    assert "qwen2.5:7b" in res.answer


def test_rag_service_fallback_on_insufficient_context():
    mock_retrieval = MagicMock()
    mock_llm = MagicMock()

    mock_retrieval.retrieve.return_value = RetrievalResult(
        query="Unrelated query",
        chunks=[],
        has_sufficient_context=False,
        highest_similarity=0.1,
        total_candidates_found=0,
    )

    rag = RAGService(retrieval_service=mock_retrieval, llm_service=mock_llm)
    req = ChatRequest(query="What is the speed of light in jupiter core?")
    res = rag.answer_query(req)

    assert res.is_grounded is False
    assert res.has_sufficient_context is False
    assert len(res.sources) == 0
    assert "sufficient information" in res.answer

