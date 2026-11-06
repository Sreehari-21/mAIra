from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.api.schemas.chat import ChatMessage, ChatRequest, ChatResponse, SourceReference
from app.core.config import Settings, get_settings
from app.core.exceptions import QueryValidationError, RAGError
from app.core.logging import logger
from app.services.conversation_service import ConversationService
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalResult, RetrievalService
from app.utils.validators import validate_query


class RAGService:
    """End-to-end RAG orchestrator: Validation -> Context -> Retrieval -> Prompt -> LLM -> Grounding."""

    def __init__(
        self,
        retrieval_service: Optional[RetrievalService] = None,
        llm_service: Optional[LLMService] = None,
        conversation_service: Optional[ConversationService] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retrieval_service = retrieval_service or RetrievalService(settings=self.settings)
        self.llm_service = llm_service or LLMService(settings=self.settings)
        self.conversation_service = conversation_service or ConversationService()

        # Load prompt templates
        self.rag_template = self._load_prompt("rag.txt")
        self.fallback_template = self._load_prompt("fallback.txt")
        self.clarification_template = self._load_prompt("clarification.txt")

    def _load_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return ""

    def _is_query_ambiguous(self, query: str) -> bool:
        """Heuristic check for vague, single-word or ambiguous queries lacking context."""
        cleaned = query.strip().lower()
        ambiguous_tokens = {"what", "tell me", "explain", "help", "info", "details", "how"}
        words = cleaned.split()
        return len(words) <= 1 and cleaned in ambiguous_tokens

    def _construct_context_string(self, chunks: List[Any]) -> str:
        """Constructs structured, source-labeled context block for LLM prompt."""
        if not chunks:
            return "No relevant context found in knowledge base."

        formatted_chunks: List[str] = []
        for i, chunk in enumerate(chunks, 1):
            page_info = f", Page: {chunk.page_number}" if chunk.page_number else ""
            header = f"--- [Document: {chunk.filename}{page_info} | Chunk ID: {chunk.chunk_id}] ---"
            formatted_chunks.append(f"{header}\n{chunk.content}")

        return "\n\n".join(formatted_chunks)

    def answer_query(self, request: ChatRequest) -> ChatResponse:
        """Orchestrates grounded question-answering with Ollama and ChromaDB."""
        start_time = time.perf_counter()

        # 1. Input Validation
        valid_query = validate_query(request.query)

        # 2. Conversation & Session Context
        conversation_id = self.conversation_service.get_or_create_session(request.conversation_id)
        history = request.history if request.history is not None else self.conversation_service.get_history(conversation_id)
        formatted_history = self.conversation_service.format_history_for_prompt(history)

        # 3. Ambiguity Check
        if self._is_query_ambiguous(valid_query):
            clarification_prompt = self.clarification_template.format(
                query=valid_query,
                context="No specific topic identified. Query consists of generic keywords.",
            )
            gen_result = self.llm_service.generate_response(
                prompt=clarification_prompt,
                temperature=0.2,
            )
            clarification_answer = gen_result.get("text", "") or "Could you please clarify your question with more specific details?"
            self.conversation_service.add_turn(conversation_id, valid_query, clarification_answer)
            latency = time.perf_counter() - start_time

            return ChatResponse(
                answer=clarification_answer,
                conversation_id=conversation_id,
                sources=[],
                is_grounded=False,
                has_sufficient_context=False,
                latency_seconds=round(latency, 3),
                model_used=self.settings.OLLAMA_GENERATION_MODEL,
            )

        # 4. Retrieval from ChromaDB
        retrieval_result = self.retrieval_service.retrieve(
            query=valid_query,
            top_k=request.top_k,
        )

        # 5. Insufficient Information Fallback Check
        if not retrieval_result.has_sufficient_context or not retrieval_result.chunks:
            fallback_text = self.fallback_template.format(query=valid_query)
            self.conversation_service.add_turn(conversation_id, valid_query, fallback_text)
            latency = time.perf_counter() - start_time

            return ChatResponse(
                answer=fallback_text,
                conversation_id=conversation_id,
                sources=[],
                is_grounded=False,
                has_sufficient_context=False,
                latency_seconds=round(latency, 3),
                model_used=self.settings.OLLAMA_GENERATION_MODEL,
            )

        # 6. Context Construction
        context_str = self._construct_context_string(retrieval_result.chunks)
        prompt = self.rag_template.format(
            context=context_str,
            conversation_history=formatted_history,
            query=valid_query,
        )

        # 7. Local LLM Generation via Ollama
        temp = request.temperature if request.temperature is not None else 0.1
        gen_result = self.llm_service.generate_response(
            prompt=prompt,
            temperature=temp,
        )
        answer = gen_result.get("text", "").strip()

        # 8. Grounding Verification
        # If model explicitly reported lack of info, reflect that in grounded status
        insufficient_phrases = [
            "do not have sufficient information",
            "not enough information",
            "cannot answer based on the provided context",
        ]
        is_grounded = not any(phrase in answer.lower() for phrase in insufficient_phrases)

        # 9. Build Source References
        sources: List[SourceReference] = []
        for c in retrieval_result.chunks:
            snippet = c.content[:200] + "..." if len(c.content) > 200 else c.content
            sources.append(
                SourceReference(
                    document_id=c.document_id,
                    filename=c.filename,
                    chunk_id=c.chunk_id,
                    page_number=c.page_number,
                    similarity_score=c.similarity_score,
                    snippet=snippet,
                )
            )

        # 10. Update Conversation Session History
        self.conversation_service.add_turn(conversation_id, valid_query, answer)

        latency = time.perf_counter() - start_time

        return ChatResponse(
            answer=answer,
            conversation_id=conversation_id,
            sources=sources,
            is_grounded=is_grounded,
            has_sufficient_context=True,
            latency_seconds=round(latency, 3),
            model_used=self.settings.OLLAMA_GENERATION_MODEL,
        )

