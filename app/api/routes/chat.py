from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.schemas.chat import ChatRequest, ChatResponse
from app.core.config import Settings, get_settings
from app.services.rag_service import RAGService

router = APIRouter(tags=["Chat & Conversation"])


def get_rag_service(settings: Settings = Depends(get_settings)) -> RAGService:
    return RAGService(settings=settings)


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate Grounded RAG Chat Response",
)
def chat_endpoint(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    """Processes a user question through the local RAG pipeline:

    1. Validates query formatting and safety limits.
    2. Resolves conversation session context.
    3. Retrieves top relevant chunks from local ChromaDB.
    4. Evaluates context sufficiency (triggers fallback if confidence is low).
    5. Synthesizes a factual, grounded answer via local Ollama LLM.
    6. Returns answer along with explicit source chunk citations and grounding flag.
    """
    return rag_service.answer_query(request)

