from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import OllamaServiceError
from app.providers.ollama.generation import OllamaGenerator


class LLMService:
    """High-level abstraction for local LLM text generation."""

    def __init__(
        self,
        generator: Optional[OllamaGenerator] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.generator = generator or OllamaGenerator(settings=self.settings)
        self.default_system_prompt = self._load_prompt("system.txt")

    def _load_prompt(self, filename: str) -> str:
        prompt_path = Path(__file__).resolve().parent.parent / "prompts" / filename
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return ""

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generates text completion using the local Ollama LLM."""
        effective_system = system_prompt if system_prompt is not None else self.default_system_prompt
        return self.generator.generate(
            prompt=prompt,
            system_prompt=effective_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def agenerate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Asynchronously generates text completion using local Ollama LLM."""
        effective_system = system_prompt if system_prompt is not None else self.default_system_prompt
        return await self.generator.agenerate(
            prompt=prompt,
            system_prompt=effective_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

