from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.ingestion.loaders.base import DocumentPage, RawDocument


@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    chunk_index: int
    page_number: Optional[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Splits document pages into semantically cohesive, overlapping chunks."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunk_size = chunk_size or self.settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.settings.CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Splits arbitrary text into segments with overlapping sliding window, respecting paragraph boundaries."""
        if not text:
            return []

        # If text is already smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]

        # Preferred split points in descending priority
        delimiters = ["\n\n", "\n", ". ", "; ", ", ", " "]
        
        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            if end < text_len:
                # Search backwards for a suitable delimiter to avoid cutting words
                best_split = -1
                for delim in delimiters:
                    split_candidate = text.rfind(delim, start + (self.chunk_size // 2), end)
                    if split_candidate != -1:
                        best_split = split_candidate + len(delim)
                        break

                if best_split != -1:
                    end = best_split

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Advance window by chunk_size - chunk_overlap
            advance = max(1, (end - start) - self.chunk_overlap)
            start += advance

        return chunks

    def chunk_document(self, document_id: str, raw_doc: RawDocument) -> List[DocumentChunk]:
        """Generates indexed chunks from all pages in a document, preserving page-level traceability."""
        chunks: List[DocumentChunk] = []
        global_chunk_idx = 0

        for page in raw_doc.pages:
            page_text = page.content.strip()
            if not page_text:
                continue

            raw_chunks = self._split_text_into_chunks(page_text)
            for c_text in raw_chunks:
                chunk_id = f"{document_id}#c{global_chunk_idx:04d}"
                chunk_meta = {
                    "document_id": document_id,
                    "filename": raw_doc.filename,
                    "chunk_id": chunk_id,
                    "chunk_index": global_chunk_idx,
                    "page_number": page.page_number,
                    "char_length": len(c_text),
                }
                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        text=c_text,
                        chunk_index=global_chunk_idx,
                        page_number=page.page_number,
                        metadata=chunk_meta,
                    )
                )
                global_chunk_idx += 1

        return chunks

