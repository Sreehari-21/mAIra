from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.chat import router as chat_router
from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.core.config import get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    GenAIPlatformException,
    IngestionError,
    OllamaConnectionError,
    OllamaServiceError,
    OllamaTimeoutError,
    QueryValidationError,
    RAGError,
    UnsupportedDocumentTypeError,
    VectorStoreError,
)
from app.core.logging import logger

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for application startup and graceful shutdown."""
    logger.info(f"Initializing {settings.APP_NAME} in '{settings.APP_ENV}' mode")
    logger.info(f"Target Ollama runtime: {settings.OLLAMA_BASE_URL}")
    logger.info(f"Configured models: Generation='{settings.OLLAMA_GENERATION_MODEL}', Embedding='{settings.OLLAMA_EMBEDDING_MODEL}'")
    logger.info(f"Persistent vector storage directory: {settings.CHROMA_PERSIST_DIRECTORY}")
    # Note: Models are NOT downloaded automatically at startup per project guidelines.
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description="Conversational AI Platform Foundation with local Ollama runtime and persistent ChromaDB.",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS configuration for local development integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Global Exception Handlers
# ------------------------------------------------------------------------------
@app.exception_handler(GenAIPlatformException)
async def genai_platform_exception_handler(request: Request, exc: GenAIPlatformException) -> JSONResponse:
    logger.error(f"GenAI platform error [{exc.code}]: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


@app.exception_handler(QueryValidationError)
async def query_validation_exception_handler(request: Request, exc: QueryValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_exception_handler(request: Request, exc: DocumentNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


@app.exception_handler(UnsupportedDocumentTypeError)
async def unsupported_doc_type_handler(request: Request, exc: UnsupportedDocumentTypeError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


@app.exception_handler(OllamaConnectionError)
async def ollama_connection_exception_handler(request: Request, exc: OllamaConnectionError) -> JSONResponse:
    logger.error(f"Ollama connection error: {exc.message}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


@app.exception_handler(OllamaTimeoutError)
async def ollama_timeout_exception_handler(request: Request, exc: OllamaTimeoutError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            },
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(f"Unhandled system error processing request: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred. Please consult server logs.",
                "details": {"exception_type": type(exc).__name__},
            },
        },
    )


# ------------------------------------------------------------------------------
# Router Mounts
# ------------------------------------------------------------------------------
app.include_router(health_router, prefix=settings.API_V1_STR)
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(knowledge_router, prefix=settings.API_V1_STR)


@app.get("/", include_in_schema=False)
def root_redirect() -> JSONResponse:
    return JSONResponse(
        content={
            "message": f"Welcome to {settings.APP_NAME}",
            "docs_url": "/docs",
            "health_url": f"{settings.API_V1_STR}/health",
        }
    )

