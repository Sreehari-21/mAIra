from __future__ import annotations

import uuid
from typing import Dict, List, Optional

from app.api.schemas.chat import ChatMessage


class ConversationService:
    """Manages multi-turn conversation sessions and context windows."""

    def __init__(self, max_history_turns: int = 6) -> None:
        self.max_history_turns = max_history_turns
        # In-memory session store: conversation_id -> List[ChatMessage]
        self._sessions: Dict[str, List[ChatMessage]] = {}

    def get_or_create_session(self, conversation_id: Optional[str] = None) -> str:
        """Returns existing session or creates a new UUID conversation ID."""
        if conversation_id and conversation_id.strip():
            cid = conversation_id.strip()
            if cid not in self._sessions:
                self._sessions[cid] = []
            return cid
        new_cid = f"conv-{uuid.uuid4().hex[:12]}"
        self._sessions[new_cid] = []
        return new_cid

    def get_history(self, conversation_id: str) -> List[ChatMessage]:
        """Returns the conversation history for a given session."""
        return self._sessions.get(conversation_id, [])

    def add_turn(self, conversation_id: str, user_message: str, assistant_message: str) -> None:
        """Appends a user and assistant turn, trimming old turns beyond max_history_turns."""
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = []

        history = self._sessions[conversation_id]
        history.append(ChatMessage(role="user", content=user_message))
        history.append(ChatMessage(role="assistant", content=assistant_message))

        # Enforce sliding window (each turn is 2 messages: user + assistant)
        max_messages = self.max_history_turns * 2
        if len(history) > max_messages:
            self._sessions[conversation_id] = history[-max_messages:]

    def format_history_for_prompt(self, history: List[ChatMessage]) -> str:
        """Formats conversation history into structured prompt text."""
        if not history:
            return "No previous conversation history."

        lines = []
        for msg in history:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")
        return "\n".join(lines)

