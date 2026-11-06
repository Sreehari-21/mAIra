from __future__ import annotations

import datetime
from typing import Any, Dict, Optional


class MetadataEnricher:
    """Standardizes metadata assignment across ingested documents and chunks."""

    @staticmethod
    def create_document_metadata(
        document_id: str,
        filename: str,
        source_type: str,
        source_path: str,
        content_hash: str,
        total_chunks: int,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates unified root metadata for a document."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        base = {
            "document_id": document_id,
            "filename": filename,
            "source_type": source_type,
            "source_path": source_path,
            "content_hash": content_hash,
            "total_chunks": total_chunks,
            "ingested_at": now,
        }
        if custom_metadata:
            base.update(custom_metadata)
        return base

    @staticmethod
    def enrich_chunk_metadata(
        doc_metadata: Dict[str, Any],
        chunk_id: str,
        chunk_index: int,
        page_number: Optional[int],
        char_length: int,
    ) -> Dict[str, Any]:
        """Merges document-level attributes with chunk-specific indicators."""
        meta = dict(doc_metadata)
        meta["chunk_id"] = chunk_id
        meta["chunk_index"] = chunk_index
        meta["char_length"] = char_length
        if page_number is not None:
            meta["page_number"] = page_number
        return meta

