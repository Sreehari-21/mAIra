from __future__ import annotations

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.api.routes.chat import get_rag_service
from app.api.schemas.chat import ChatResponse, SourceReference
from app.main import app


def test_chat_endpoint_success(client: TestClient):
    mock_rag = MagicMock()
    mock_rag.answer_query.return_value = ChatResponse(
        answer="The system uses local Ollama runtime.",
        conversation_id="conv-test-123",
        sources=[
            SourceReference(
                document_id="doc-test-1",
                filename="arch.txt",
                chunk_id="doc-test-1#c0000",
                page_number=1,
                similarity_score=0.91,
                snippet="The system uses local Ollama runtime.",
            )
        ],
        is_grounded=True,
        has_sufficient_context=True,
        latency_seconds=0.12,
        model_used="qwen2.5:7b",
    )

    app.dependency_overrides[get_rag_service] = lambda: mock_rag
    try:
        response = client.post(
            "/api/v1/chat",
            json={"query": "What runtime is used?", "conversation_id": "conv-test-123"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The system uses local Ollama runtime."
        assert data["conversation_id"] == "conv-test-123"
        assert len(data["sources"]) == 1
        assert data["is_grounded"] is True
    finally:
        app.dependency_overrides.pop(get_rag_service, None)


def test_chat_endpoint_validation_error(client: TestClient):
    response = client.post("/api/v1/chat", json={"query": ""})
    assert response.status_code == 422  # Pydantic min_length validation error

