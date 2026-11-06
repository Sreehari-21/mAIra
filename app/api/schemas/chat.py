from __future__ import annotations

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="Message author role")
    content: str = Field(..., description="Text content of message")


class SourceReference(BaseModel):
    document_id: str = Field(..., description="Unique document ID in vector store")
    filename: str = Field(..., description="Origin filename")
    chunk_id: str = Field(..., description="Chunk ID within the document")
    page_number: Optional[int] = Field(None, description="Page number if applicable")
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    snippet: str = Field(..., description="Context excerpt text")


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User question or query",
        examples=["What is the architecture of the GenAI platform?"],
    )
    conversation_id: Optional[str] = Field(
        default=None,
        description="Optional conversation/session ID to preserve context across turns",
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Optional prior chat turn history",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Number of knowledge chunks to retrieve. Defaults to server setting.",
    )
    temperature: Optional[float] = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for LLM generation. Lower values ensure higher grounding.",
    )


class ChatResponse(BaseModel):
    answer: str = Field(..., description="Synthesized grounded answer from local LLM")
    conversation_id: str = Field(..., description="Conversation session ID")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="List of cited source chunks used to generate the response",
    )
    is_grounded: bool = Field(
        ...,
        description="Whether the answer is grounded in retrieved context (false for fallback/clarification)",
    )
    has_sufficient_context: bool = Field(
        ...,
        description="Whether sufficient context was retrieved from vector store",
    )
    latency_seconds: float = Field(..., description="Total processing latency in seconds")
    model_used: str = Field(..., description="LLM model used for synthesis")

