from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    OllamaConnectionError,
    OllamaServiceError,
    OllamaTimeoutError,
)
from app.core.logging import logger
from app.providers.ollama.client import OllamaClient


class OllamaEmbedder:
    """Local text embedding provider using Ollama's /api/embeddings or /api/embed endpoints."""

    def __init__(self, client: Optional[OllamaClient] = None, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or OllamaClient(settings=self.settings)
        self.model = self.settings.OLLAMA_EMBEDDING_MODEL

    def embed_text(self, text: str, model: Optional[str] = None) -> List[float]:
        """Synchronously computes vector embedding for a single text string."""
        target_model = model or self.model
        payload = {
            "model": target_model,
            "prompt": text,
        }

        try:
            with self.client._get_client() as http_client:
                # Try standard /api/embeddings
                response = http_client.post("/api/embeddings", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding")
                    if isinstance(embedding, list):
                        return embedding

                # Try alternative /api/embed (Ollama v0.1.34+)
                embed_payload = {"model": target_model, "input": text}
                alt_response = http_client.post("/api/embed", json=embed_payload)
                if alt_response.status_code == 200:
                    data = alt_response.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and isinstance(embeddings[0], list):
                        return embeddings[0]

                raise OllamaServiceError(
                    f"Ollama embedding failed with status {response.status_code}: {response.text}",
                    status_code=502,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(base_url=self.client.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.client.timeout) from exc

    async def aembed_text(self, text: str, model: Optional[str] = None) -> List[float]:
        """Asynchronously computes vector embedding for a single text string."""
        target_model = model or self.model
        payload = {
            "model": target_model,
            "prompt": text,
        }

        try:
            async with self.client._get_async_client() as http_client:
                response = await http_client.post("/api/embeddings", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding")
                    if isinstance(embedding, list):
                        return embedding

                embed_payload = {"model": target_model, "input": text}
                alt_response = await http_client.post("/api/embed", json=embed_payload)
                if alt_response.status_code == 200:
                    data = alt_response.json()
                    embeddings = data.get("embeddings", [])
                    if embeddings and isinstance(embeddings[0], list):
                        return embeddings[0]

                raise OllamaServiceError(
                    f"Ollama embedding failed with status {response.status_code}: {response.text}",
                    status_code=502,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(base_url=self.client.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.client.timeout) from exc

    def embed_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Computes embeddings for a batch of texts sequentially."""
        return [self.embed_text(t, model=model) for t in texts]

    async def aembed_batch(self, texts: List[str], model: Optional[str] = None) -> List[List[float]]:
        """Asynchronously computes embeddings for a batch of texts."""
        results: List[List[float]] = []
        for text in texts:
            emb = await self.aembed_text(text, model=model)
            results.append(emb)
        return results

