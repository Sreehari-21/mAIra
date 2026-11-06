from __future__ import annotations

import os
import shutil
import tempfile
from typing import Generator
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import app
from app.vectorstore.chroma_store import ChromaStore


@pytest.fixture(scope="session")
def test_temp_dir() -> Generator[str, None, None]:
    temp_dir = tempfile.mkdtemp(prefix="genai_test_chroma_")
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_settings(test_temp_dir: str) -> Settings:
    return Settings(
        APP_ENV="testing",
        DEBUG=True,
        OLLAMA_BASE_URL="http://localhost:11434",
        OLLAMA_GENERATION_MODEL="qwen2.5:7b",
        OLLAMA_EMBEDDING_MODEL="nomic-embed-text",
        CHROMA_PERSIST_DIRECTORY=test_temp_dir,
        CHROMA_COLLECTION_NAME="test_collection",
        RAG_TOP_K=2,
        RAG_SIMILARITY_THRESHOLD=0.2,
    )


@pytest.fixture
def test_chroma_store(test_settings: Settings) -> ChromaStore:
    return ChromaStore(settings=test_settings)


@pytest.fixture
def client(test_settings: Settings) -> Generator[TestClient, None, None]:
    # Override settings dependency
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

