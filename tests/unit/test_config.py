from __future__ import annotations

import os
from app.core.config import Settings, get_settings


def test_default_settings():
    settings = Settings()
    assert settings.APP_NAME == "GenAI Conversational Platform"
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert settings.OLLAMA_GENERATION_MODEL == "qwen2.5:7b"
    assert settings.OLLAMA_EMBEDDING_MODEL == "nomic-embed-text"
    assert settings.CHROMA_PERSIST_DIRECTORY == "./data/chroma"
    assert settings.RAG_TOP_K == 4
    assert settings.CHUNK_SIZE == 500
    assert settings.CHUNK_OVERLAP == 50


def test_settings_custom_override():
    custom = Settings(
        APP_ENV="testing",
        OLLAMA_GENERATION_MODEL="custom-model:latest",
        CHUNK_SIZE=300,
    )
    assert custom.APP_ENV == "testing"
    assert custom.OLLAMA_GENERATION_MODEL == "custom-model:latest"
    assert custom.CHUNK_SIZE == 300

