from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from app.api.routes.health import get_ollama_client
from app.core.exceptions import OllamaConnectionError
from app.main import app


def test_health_endpoint(client: TestClient):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"
    assert "timestamp" in data


def test_ollama_health_endpoint_healthy(client: TestClient, monkeypatch):
    class MockHealthyClient:
        def check_health(self):
            return {"status": "connected"}

        def list_local_models(self):
            return ["qwen2.5:7b", "nomic-embed-text"]

    app.dependency_overrides[get_ollama_client] = lambda: MockHealthyClient()
    try:
        response = client.get("/api/v1/health/ollama")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "connected"
        assert data["is_available"] is True
    finally:
        app.dependency_overrides.pop(get_ollama_client, None)


def test_ollama_health_endpoint_offline(client: TestClient):
    class MockOfflineClient:
        def check_health(self):
            raise OllamaConnectionError("Daemon offline")

        def list_local_models(self):
            return []

    app.dependency_overrides[get_ollama_client] = lambda: MockOfflineClient()
    try:
        response = client.get("/api/v1/health/ollama")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "offline"
        assert data["is_available"] is False
    finally:
        app.dependency_overrides.pop(get_ollama_client, None)

