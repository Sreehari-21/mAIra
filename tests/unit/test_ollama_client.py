from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import OllamaConnectionError, OllamaModelNotFoundError
from app.providers.ollama.client import OllamaClient


def test_ollama_health_success(monkeypatch):
    settings = Settings(OLLAMA_BASE_URL="http://mock-ollama:11434")
    client = OllamaClient(settings=settings)

    def mock_get(url, *args, **kwargs):
        return httpx.Response(200, json={"version": "0.1.32"})

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    result = client.check_health()
    assert result["status"] == "connected"
    assert result["version"] == "0.1.32"


def test_ollama_connection_failure(monkeypatch):
    settings = Settings(OLLAMA_BASE_URL="http://unreachable:11434")
    client = OllamaClient(settings=settings)

    def mock_get(url, *args, **kwargs):
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    with pytest.raises(OllamaConnectionError) as exc_info:
        client.check_health()
    assert exc_info.value.status_code == 503


def test_list_local_models(monkeypatch):
    settings = Settings(OLLAMA_BASE_URL="http://mock-ollama:11434")
    client = OllamaClient(settings=settings)

    def mock_get(url, *args, **kwargs):
        return httpx.Response(
            200,
            json={"models": [{"name": "qwen2.5:7b"}, {"name": "nomic-embed-text:latest"}]},
        )

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    models = client.list_local_models()
    assert "qwen2.5:7b" in models
    assert "nomic-embed-text:latest" in models
    assert client.verify_model_available("qwen2.5:7b") is True


def test_verify_model_missing(monkeypatch):
    settings = Settings(OLLAMA_BASE_URL="http://mock-ollama:11434")
    client = OllamaClient(settings=settings)

    def mock_get(url, *args, **kwargs):
        return httpx.Response(200, json={"models": [{"name": "llama3:8b"}]})

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    with pytest.raises(OllamaModelNotFoundError):
        client.verify_model_available("qwen2.5:7b")

