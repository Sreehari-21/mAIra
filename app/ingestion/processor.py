from __future__ import annotations

import hashlib
from typing import Any, Dict

from app.core.exceptions import IngestionError
from app.ingestion.loaders.base import RawDocument
from app.utils.text_cleaner import clean_text


class DocumentProcessor:
    """Performs validation, canonical normalization, and content hashing for deduplication."""

    @staticmethod
    def compute_content_hash(text: str) -> str:
        """Computes SHA-256 hash of normalized text for deduplication checks."""
        normalized = clean_text(text).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    @classmethod
    def process(cls, raw_doc: RawDocument) -> RawDocument:
        """Cleans and validates all pages within a raw document."""
        processed_pages = []
        for page in raw_doc.pages:
            cleaned = clean_text(page.content)
            if cleaned:
                page.content = cleaned
                processed_pages.append(page)

        if not processed_pages:
            raise IngestionError(f"Document '{raw_doc.filename}' has no valid text after cleaning.")

        raw_doc.pages = processed_pages
        raw_doc.metadata["content_hash"] = cls.compute_content_hash(raw_doc.full_text)
        raw_doc.metadata["total_characters"] = len(raw_doc.full_text)

        return raw_doc

