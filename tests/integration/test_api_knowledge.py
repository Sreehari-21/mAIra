from __future__ import annotations

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from app.api.routes.knowledge import get_chroma_store, get_pipeline
from app.core.exceptions import DocumentNotFoundError
from app.main import app


def test_knowledge_ingest_text_endpoint(client: TestClient):
    mock_pipeline = MagicMock()
    mock_pipeline.ingest_raw_text.return_value = {
        "document_id": "doc-manual-text-1234",
        "filename": "Platform Overview",
        "chunk_count": 2,
        "character_count": 600,
        "status": "indexed",
    }

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    try:
        response = client.post(
            "/api/v1/knowledge/ingest",
            json={
                "title": "Platform Overview",
                "content": "This is detailed text content for platform overview knowledge base testing.",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["document_id"] == "doc-manual-text-1234"
        assert data["chunk_count"] == 2
        assert data["status"] == "indexed"
    finally:
        app.dependency_overrides.pop(get_pipeline, None)


def test_knowledge_list_endpoint(client: TestClient):
    mock_store = MagicMock()
    mock_store.list_documents.return_value = [
        {
            "document_id": "doc-manual-text-1234",
            "filename": "Platform Overview",
            "chunk_count": 2,
            "source_type": "text",
            "ingested_at": "2026-11-06T00:00:00Z",
            "metadata": {},
        }
    ]

    app.dependency_overrides[get_chroma_store] = lambda: mock_store
    try:
        response = client.get("/api/v1/knowledge")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 1
        assert data["documents"][0]["document_id"] == "doc-manual-text-1234"
    finally:
        app.dependency_overrides.pop(get_chroma_store, None)


def test_knowledge_delete_endpoint(client: TestClient):
    mock_pipeline = MagicMock()
    mock_pipeline.delete_document.return_value = 2

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    try:
        response = client.delete("/api/v1/knowledge/doc-manual-text-1234")
        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["chunks_removed"] == 2
    finally:
        app.dependency_overrides.pop(get_pipeline, None)


def test_knowledge_delete_not_found(client: TestClient):
    mock_pipeline = MagicMock()
    mock_pipeline.delete_document.side_effect = DocumentNotFoundError("doc-missing")

    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    try:
        response = client.delete("/api/v1/knowledge/doc-missing")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.pop(get_pipeline, None)

