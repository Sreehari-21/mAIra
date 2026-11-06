from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentPage:
    page_number: Optional[int]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawDocument:
    source_path: str
    filename: str
    pages: List[DocumentPage]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.content for page in self.pages if page.content.strip())


class BaseLoader(ABC):
    """Abstract interface for document loaders."""

    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Determines whether this loader supports the given file format."""
        pass

    @abstractmethod
    def load(self, file_path: Path) -> RawDocument:
        """Extracts text content and metadata from the document file."""
        pass

