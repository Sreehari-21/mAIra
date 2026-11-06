from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Document title or identifier")
    content: str = Field(..., min_length=10, description="Raw text content to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional custom metadata tags")


class IngestionResponse(BaseModel):
    document_id: str = Field(..., description="Unique generated document identifier")
    filename: str = Field(..., description="Document filename or title")
    chunk_count: int = Field(..., description="Number of vectorized chunks produced")
    character_count: int = Field(..., description="Total characters processed")
    status: str = Field(..., examples=["indexed"])
    message: str = Field(..., examples=["Document successfully processed and indexed."])


class DocumentItem(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    source_type: str
    ingested_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentListResponse(BaseModel):
    documents: List[DocumentItem]
    total_count: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    chunks_removed: int
    message: str

