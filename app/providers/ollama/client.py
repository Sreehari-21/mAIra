from __future__ import annotations

from typing import Any, Dict, List, Optional
import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaServiceError,
    OllamaTimeoutError,
)
from app.core.logging import logger


class OllamaClient:
    """Reusable HTTP client interface for local Ollama runtime.

    Strict local execution guarantee: All calls route exclusively to the local
    Ollama daemon. No remote or third-party cloud AI APIs are invoked.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL.rstrip("/")
        self.timeout = self.settings.OLLAMA_TIMEOUT_SECONDS

    def _get_client(self) -> httpx.Client:
        return httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _get_async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)

    def check_health(self) -> Dict[str, Any]:
        """Performs a synchronous ping to local Ollama server."""
        try:
            with self._get_client() as client:
                response = client.get("/api/version")
                if response.status_code == 200:
                    data = response.json()
                    return {"status": "connected", "version": data.get("version", "unknown")}
                raise OllamaServiceError(
                    f"Ollama returned unexpected HTTP {response.status_code}",
                    status_code=503,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            logger.warning(f"Ollama connection check failed: {exc}")
            raise OllamaConnectionError(base_url=self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.timeout) from exc
        except Exception as exc:
            if isinstance(exc, (OllamaConnectionError, OllamaTimeoutError, OllamaServiceError)):
                raise
            raise OllamaServiceError(f"Unexpected error communicating with Ollama: {exc}") from exc

    async def acheck_health(self) -> Dict[str, Any]:
        """Asynchronously checks local Ollama server health."""
        try:
            async with self._get_async_client() as client:
                response = await client.get("/api/version")
                if response.status_code == 200:
                    data = response.json()
                    return {"status": "connected", "version": data.get("version", "unknown")}
                raise OllamaServiceError(
                    f"Ollama returned unexpected HTTP {response.status_code}",
                    status_code=503,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(base_url=self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.timeout) from exc
        except Exception as exc:
            if isinstance(exc, (OllamaConnectionError, OllamaTimeoutError, OllamaServiceError)):
                raise
            raise OllamaServiceError(f"Unexpected error communicating with Ollama: {exc}") from exc

    def list_local_models(self) -> List[str]:
        """Lists all models downloaded in the local Ollama store."""
        try:
            with self._get_client() as client:
                response = client.get("/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [m.get("name", "") for m in models if "name" in m]
                return []
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(base_url=self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.timeout) from exc

    async def alist_local_models(self) -> List[str]:
        """Asynchronously lists all models downloaded in local Ollama store."""
        try:
            async with self._get_async_client() as client:
                response = await client.get("/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [m.get("name", "") for m in models if "name" in m]
                return []
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise OllamaConnectionError(base_url=self.base_url) from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(timeout=self.timeout) from exc

    def verify_model_available(self, model_name: str) -> bool:
        """Verifies if model is locally present; raises OllamaModelNotFoundError if absent."""
        available = self.list_local_models()
        # Handle tag variants (e.g. 'qwen2.5:7b' vs 'qwen2.5:7b-instruct')
        if not any(model_name == m or model_name in m.split(":")[0] for m in available):
            raise OllamaModelNotFoundError(model_name=model_name)
        return True

