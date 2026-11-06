from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class VectorSearchResult:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float
    similarity_score: float  # Normalized 0.0 to 1.0 (higher is more similar)


class BaseVectorStore(ABC):
    """Abstract interface for local vector database backends."""

    @abstractmethod
    def add_documents(
        self,
        documents: List[VectorDocument],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Vectorizes and persists a list of documents with their precomputed embeddings."""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Searches vector space for closest matches to query embedding."""
        pass

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> int:
        """Deletes all chunks belonging to the given parent document_id. Returns chunk count deleted."""
        pass

    @abstractmethod
    def get_document_chunks(self, document_id: str) -> List[VectorDocument]:
        """Retrieves all stored chunks for a specific parent document_id."""
        pass

    @abstractmethod
    def list_documents(self) -> List[Dict[str, Any]]:
        """Returns summary metadata of all ingested parent documents in store."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Returns storage metrics and collection status."""
        pass

