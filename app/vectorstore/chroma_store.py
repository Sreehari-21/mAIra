from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentNotFoundError, VectorStoreError
from app.core.logging import logger
from app.vectorstore.base import BaseVectorStore, VectorDocument, VectorSearchResult


class ChromaStore(BaseVectorStore):
    """Production-grade persistent ChromaDB adapter implementing BaseVectorStore."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.persist_directory = str(self.settings.chroma_dir_path)
        self.collection_name = self.settings.CHROMA_COLLECTION_NAME
        self.distance_metric = self.settings.CHROMA_DISTANCE_METRIC

        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False, is_persistent=True),
            )
            # "hnsw:space" can be "cosine", "l2", or "ip"
            metadata = {"hnsw:space": self.distance_metric}
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata=metadata,
            )
            logger.info(
                f"Initialized ChromaDB at '{self.persist_directory}', collection='{self.collection_name}'"
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize ChromaDB store: {exc}") from exc

    def add_documents(
        self,
        documents: List[VectorDocument],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Adds documents with their corresponding embeddings into ChromaDB."""
        if not documents:
            return []

        if len(documents) != len(embeddings):
            raise VectorStoreError("Documents count and embeddings count must match.")

        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            return ids
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert documents into ChromaDB: {exc}") from exc

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[VectorSearchResult]:
        """Performs nearest-neighbor search using cosine similarity."""
        try:
            total_count = self._collection.count()
            if total_count == 0:
                return []

            n_results = min(top_k, total_count)
            query_args: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if filter_metadata:
                query_args["where"] = filter_metadata

            results = self._collection.query(**query_args)

            matched_docs = results.get("documents", [[]])[0]
            matched_ids = results.get("ids", [[]])[0]
            matched_metas = results.get("metadatas", [[]])[0]
            matched_dists = results.get("distances", [[]])[0]

            output: List[VectorSearchResult] = []
            for doc_id, text, meta, dist in zip(matched_ids, matched_docs, matched_metas, matched_dists):
                # For cosine distance: distance is 1 - cosine_similarity.
                # So similarity = 1 - distance.
                if self.distance_metric == "cosine":
                    sim = max(0.0, min(1.0, 1.0 - float(dist)))
                else:
                    # Inverted normalization for L2:
                    sim = 1.0 / (1.0 + float(dist))

                output.append(
                    VectorSearchResult(
                        id=doc_id,
                        text=text,
                        metadata=meta or {},
                        distance=float(dist),
                        similarity_score=round(sim, 4),
                    )
                )

            # Sort descending by similarity score
            output.sort(key=lambda x: x.similarity_score, reverse=True)
            return output
        except Exception as exc:
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

    def delete_by_document_id(self, document_id: str) -> int:
        """Deletes all chunks belonging to a document_id."""
        try:
            # Query existing chunk ids for this document
            existing = self._collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            ids_to_delete = existing.get("ids", [])
            if not ids_to_delete:
                raise DocumentNotFoundError(document_id=document_id)

            self._collection.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except DocumentNotFoundError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document '{document_id}': {exc}") from exc

    def get_document_chunks(self, document_id: str) -> List[VectorDocument]:
        """Retrieves all chunks matching the given parent document_id."""
        try:
            records = self._collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas"],
            )
            ids = records.get("ids", [])
            texts = records.get("documents", [])
            metas = records.get("metadatas", [])

            return [
                VectorDocument(id=cid, text=ctext, metadata=cmeta or {})
                for cid, ctext, cmeta in zip(ids, texts, metas)
            ]
        except Exception as exc:
            raise VectorStoreError(f"Failed to retrieve document chunks: {exc}") from exc

    def list_documents(self) -> List[Dict[str, Any]]:
        """Aggregates all unique parent documents currently indexed."""
        try:
            all_records = self._collection.get(include=["metadatas"])
            metas = all_records.get("metadatas", [])

            doc_map: Dict[str, Dict[str, Any]] = {}
            for m in metas:
                if not m or "document_id" not in m:
                    continue
                doc_id = m["document_id"]
                if doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "document_id": doc_id,
                        "filename": m.get("filename", "unknown"),
                        "source_type": m.get("source_type", "unknown"),
                        "ingested_at": m.get("ingested_at", "unknown"),
                        "chunk_count": 0,
                        "metadata": {
                            k: v
                            for k, v in m.items()
                            if k not in ("document_id", "filename", "source_type", "ingested_at", "chunk_id")
                        },
                    }
                doc_map[doc_id]["chunk_count"] += 1

            return list(doc_map.values())
        except Exception as exc:
            raise VectorStoreError(f"Failed to list documents: {exc}") from exc

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics for the active collection."""
        try:
            count = self._collection.count()
            docs = self.list_documents()
            return {
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "total_chunks": count,
                "total_documents": len(docs),
                "distance_metric": self.distance_metric,
            }
        except Exception as exc:
            raise VectorStoreError(f"Failed to get collection stats: {exc}") from exc

