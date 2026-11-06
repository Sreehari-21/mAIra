from __future__ import annotations

import pytest

from app.ingestion.chunker import DocumentChunker
from app.ingestion.loaders.base import DocumentPage, RawDocument


def test_chunker_small_text():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    pages = [DocumentPage(page_number=1, content="Short text that fits in one chunk.")]
    raw_doc = RawDocument(source_path="test.txt", filename="test.txt", pages=pages)

    chunks = chunker.chunk_document("doc-test", raw_doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_id == "doc-test#c0000"
    assert chunks[0].page_number == 1
    assert "Short text" in chunks[0].text


def test_chunker_sliding_window():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=15)
    long_text = "This is a longer paragraph. It contains multiple sentences to verify sliding window chunking."
    pages = [DocumentPage(page_number=1, content=long_text)]
    raw_doc = RawDocument(source_path="test.txt", filename="test.txt", pages=pages)

    chunks = chunker.chunk_document("doc-test", raw_doc)
    assert len(chunks) > 1
    # Verify sequential indexing
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i
        assert chunk.chunk_id == f"doc-test#c{i:04d}"


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=50, chunk_overlap=50)

