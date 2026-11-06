from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ErrorDetails(BaseModel):
    code: str = Field(..., description="Machine-readable error identifier")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[dict[str, Any]] = Field(default=None, description="Optional diagnostic details")


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetails


class StandardResponse(BaseModel, Generic[DataT]):
    success: bool = True
    data: DataT
    message: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    version: str = Field(..., examples=["0.1.0"])
    environment: str = Field(..., examples=["development"])
    timestamp: str


class OllamaHealthResponse(BaseModel):
    status: str = Field(..., examples=["connected"])
    base_url: str = Field(..., examples=["http://localhost:11434"])
    generation_model: str = Field(..., examples=["qwen2.5:7b"])
    embedding_model: str = Field(..., examples=["nomic-embed-text"])
    is_available: bool
    models_available: list[str] = Field(default_factory=list)
    message: str

