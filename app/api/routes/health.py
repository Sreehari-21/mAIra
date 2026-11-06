from __future__ import annotations

import datetime
from fastapi import APIRouter, Depends, status

from app.api.schemas.common import HealthResponse, OllamaHealthResponse
from app.core.config import Settings, get_settings
from app.core.exceptions import OllamaServiceError
from app.providers.ollama.client import OllamaClient

router = APIRouter(tags=["Health"])


def get_ollama_client(settings: Settings = Depends(get_settings)) -> OllamaClient:
    return OllamaClient(settings=settings)


@router.get("/health", response_model=HealthResponse, summary="Platform Health Check")
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Returns the operational status of the GenAI Platform API."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=settings.APP_ENV,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


@router.get("/health/ollama", response_model=OllamaHealthResponse, summary="Ollama LLM Runtime Health Check")
def ollama_health_check(
    settings: Settings = Depends(get_settings),
    client: OllamaClient = Depends(get_ollama_client),
) -> OllamaHealthResponse:
    """Inspects connectivity to the local Ollama daemon and checks required model availability."""
    try:
        client.check_health()
        models = client.list_local_models()

        gen_model = settings.OLLAMA_GENERATION_MODEL
        emb_model = settings.OLLAMA_EMBEDDING_MODEL

        gen_present = any(gen_model == m or gen_model in m for m in models)
        emb_present = any(emb_model == m or emb_model in m for m in models)

        if gen_present and emb_present:
            status_msg = "Ollama is online and all required models are present."
            is_healthy = True
        else:
            missing = []
            if not gen_present:
                missing.append(f"generation model '{gen_model}'")
            if not emb_present:
                missing.append(f"embedding model '{emb_model}'")
            status_msg = f"Ollama is connected but missing required model(s): {', '.join(missing)}. Run setup script."
            is_healthy = False

        return OllamaHealthResponse(
            status="connected" if is_healthy else "degraded",
            base_url=settings.OLLAMA_BASE_URL,
            generation_model=gen_model,
            embedding_model=emb_model,
            is_available=is_healthy,
            models_available=models,
            message=status_msg,
        )

    except OllamaServiceError as err:
        return OllamaHealthResponse(
            status="offline",
            base_url=settings.OLLAMA_BASE_URL,
            generation_model=settings.OLLAMA_GENERATION_MODEL,
            embedding_model=settings.OLLAMA_EMBEDDING_MODEL,
            is_available=False,
            models_available=[],
            message=f"Local Ollama daemon unreachable: {err.message}",
        )

