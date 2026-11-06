from __future__ import annotations

from typing import List, Optional

from app.core.config import Settings, get_settings
from app.providers.ollama.embeddings import OllamaEmbedder


class EmbeddingService:
    """High-level abstraction for vector embeddings."""

    def __init__(
        self,
        embedder: Optional[OllamaEmbedder] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or OllamaEmbedder(settings=self.settings)

    def embed_text(self, text: str) -> List[float]:
        """Generates embedding for a single text."""
        return self.embedder.embed_text(text)

    async def aembed_text(self, text: str) -> List[float]:
        """Asynchronously generates embedding for a single text."""
        return await self.embedder.aembed_text(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a batch of texts."""
        return self.embedder.embed_batch(texts)

    async def aembed_batch(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously generates embeddings for a batch of texts."""
        return await self.embedder.aembed_batch(texts)

