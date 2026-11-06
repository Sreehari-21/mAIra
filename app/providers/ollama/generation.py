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


class OllamaGenerator:
    """Local text generation provider using Ollama's /api/generate and /api/chat endpoints."""

    def __init__(self, client: Optional[OllamaClient] = None, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or OllamaClient(settings=self.settings)
        self.model = self.settings.OLLAMA_GENERATION_MODEL

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synchronously generates text response via local Ollama."""
        target_model = model or self.model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            with self.client._get_client() as http_client:
                response = http_client.post("/api/generate", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "text": data.get("response", "").strip(),
                        "model": target_model,
                        "total_duration": data.get("total_duration", 0),
                        "eval_count": data.get("eval_count", 0),
                    }
                raise OllamaServiceError(
                    f"Ollama generation failed with status {response.status_code}: {response.text}",
                    status_code=502,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(base_url=self.client.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.client.timeout) from exc

    async def agenerate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Asynchronously generates text response via local Ollama."""
        target_model = model or self.model
        payload: Dict[str, Any] = {
            "model": target_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        try:
            async with self.client._get_async_client() as http_client:
                response = await http_client.post("/api/generate", json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "text": data.get("response", "").strip(),
                        "model": target_model,
                        "total_duration": data.get("total_duration", 0),
                        "eval_count": data.get("eval_count", 0),
                    }
                raise OllamaServiceError(
                    f"Ollama generation failed with status {response.status_code}: {response.text}",
                    status_code=502,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(base_url=self.client.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.client.timeout) from exc

