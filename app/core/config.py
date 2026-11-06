from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "GenAI Conversational Platform"
    APP_ENV: Literal["development", "testing", "production"] = "development"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Ollama Runtime Settings (Strict Local Execution)
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama instance",
    )
    OLLAMA_GENERATION_MODEL: str = Field(
        default="qwen2.5:7b",
        description="Local generation model hosted by Ollama",
    )
    OLLAMA_EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text",
        description="Local embedding model hosted by Ollama",
    )
    OLLAMA_TIMEOUT_SECONDS: float = Field(
        default=60.0,
        description="Timeout for Ollama API requests in seconds",
    )
    OLLAMA_MAX_RETRIES: int = Field(
        default=2,
        description="Number of retries for transient Ollama network errors",
    )

    # ChromaDB Vector Store
    CHROMA_PERSIST_DIRECTORY: str = Field(
        default="./data/chroma",
        description="Filesystem path for persistent ChromaDB storage",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="genai_knowledge_base",
        description="Default Chroma collection name",
    )
    CHROMA_DISTANCE_METRIC: Literal["cosine", "l2", "ip"] = "cosine"

    # RAG Pipeline Configuration
    RAG_TOP_K: int = Field(
        default=4,
        description="Number of top chunks to retrieve for RAG",
    )
    RAG_SIMILARITY_THRESHOLD: float = Field(
        default=0.35,
        description="Minimum similarity score threshold (higher is more relevant in cosine space)",
    )
    RAG_MAX_CONTEXT_TOKENS: int = Field(
        default=2048,
        description="Estimated token budget for context prompt",
    )
    CHUNK_SIZE: int = Field(
        default=500,
        description="Character count per chunk",
    )
    CHUNK_OVERLAP: int = Field(
        default=50,
        description="Character overlap between consecutive chunks",
    )

    # Storage Paths
    DATA_DOCUMENTS_DIR: str = "./data/documents"
    DATA_PROCESSED_DIR: str = "./data/processed"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def chroma_dir_path(self) -> Path:
        p = Path(self.CHROMA_PERSIST_DIRECTORY)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache()
def get_settings() -> Settings:
    """Returns cached application settings."""
    return Settings()

