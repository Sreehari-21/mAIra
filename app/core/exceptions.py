from __future__ import annotations

from typing import Any, Optional


class GenAIPlatformException(Exception):
    """Base exception for all GenAI platform errors."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


# ------------------------------------------------------------------------------
# Ollama Runtime Exceptions
# ------------------------------------------------------------------------------
class OllamaServiceError(GenAIPlatformException):
    """Base exception for Ollama runtime failures."""

    def __init__(
        self,
        message: str,
        code: str = "OLLAMA_SERVICE_ERROR",
        status_code: int = 503,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code=code, status_code=status_code, details=details)


class OllamaConnectionError(OllamaServiceError):
    """Raised when the Ollama daemon cannot be reached."""

    def __init__(
        self,
        message: str = "Cannot connect to local Ollama runtime. Ensure 'ollama serve' is running.",
        base_url: str = "http://localhost:11434",
    ) -> None:
        super().__init__(
            message=f"{message} (Endpoint: {base_url})",
            code="OLLAMA_CONNECTION_FAILED",
            status_code=503,
            details={"base_url": base_url, "hint": "Run `ollama serve` and verify models are installed."},
        )


class OllamaTimeoutError(OllamaServiceError):
    """Raised when Ollama request exceeds timeout limit."""

    def __init__(self, message: str = "Ollama operation timed out.", timeout: float = 60.0) -> None:
        super().__init__(
            message=message,
            code="OLLAMA_TIMEOUT",
            status_code=504,
            details={"timeout_seconds": timeout},
        )


class OllamaModelNotFoundError(OllamaServiceError):
    """Raised when requested model is not downloaded in Ollama."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            message=f"Model '{model_name}' is not installed in local Ollama instance.",
            code="OLLAMA_MODEL_NOT_FOUND",
            status_code=400,
            details={
                "model_name": model_name,
                "hint": f"Run `ollama pull {model_name}` to install it locally.",
            },
        )


# ------------------------------------------------------------------------------
# Vector Store & Knowledge Base Exceptions
# ------------------------------------------------------------------------------
class VectorStoreError(GenAIPlatformException):
    """Base exception for vector database operations."""

    def __init__(
        self,
        message: str,
        code: str = "VECTOR_STORE_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code=code, status_code=status_code, details=details)


class DocumentNotFoundError(VectorStoreError):
    """Raised when document does not exist in vector store."""

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document with ID '{document_id}' not found in knowledge base.",
            code="DOCUMENT_NOT_FOUND",
            status_code=404,
            details={"document_id": document_id},
        )


# ------------------------------------------------------------------------------
# Ingestion Exceptions
# ------------------------------------------------------------------------------
class IngestionError(GenAIPlatformException):
    """Base exception for document ingestion failures."""

    def __init__(
        self,
        message: str,
        code: str = "INGESTION_ERROR",
        status_code: int = 422,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code=code, status_code=status_code, details=details)


class UnsupportedDocumentTypeError(IngestionError):
    """Raised when file extension is not supported by loaders."""

    def __init__(self, extension: str, supported: list[str]) -> None:
        super().__init__(
            message=f"Unsupported file format '{extension}'. Supported formats: {', '.join(supported)}",
            code="UNSUPPORTED_DOCUMENT_TYPE",
            status_code=415,
            details={"extension": extension, "supported": supported},
        )


# ------------------------------------------------------------------------------
# RAG & Retrieval Exceptions
# ------------------------------------------------------------------------------
class RAGError(GenAIPlatformException):
    """Base exception for RAG processing failures."""

    def __init__(
        self,
        message: str,
        code: str = "RAG_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code=code, status_code=status_code, details=details)


class QueryValidationError(GenAIPlatformException):
    """Raised when a user query fails domain validation."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            code="QUERY_VALIDATION_ERROR",
            status_code=400,
            details=details,
        )

