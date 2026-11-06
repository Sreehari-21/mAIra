from __future__ import annotations

import pytest

from app.vectorstore.base import VectorDocument
from app.vectorstore.chroma_store import ChromaStore


def test_chroma_store_crud(test_chroma_store: ChromaStore):
    store = test_chroma_store

    docs = [
        VectorDocument(
            id="doc-1#c0000",
            text="Ollama is a local LLM runtime.",
            metadata={"document_id": "doc-1", "filename": "ollama.txt", "page_number": 1},
        ),
        VectorDocument(
            id="doc-1#c0001",
            text="ChromaDB provides persistent local embeddings storage.",
            metadata={"document_id": "doc-1", "filename": "ollama.txt", "page_number": 1},
        ),
    ]
    # Simple synthetic 4-dim unit vectors for test
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]

    # Add
    stored_ids = store.add_documents(docs, embeddings)
    assert len(stored_ids) == 2

    # Search with identical vector to doc 0
    query_emb = [0.99, 0.01, 0.0, 0.0]
    results = store.similarity_search(query_emb, top_k=2)
    assert len(results) == 2
    assert results[0].id == "doc-1#c0000"
    assert results[0].similarity_score > 0.9

    # List
    listed = store.list_documents()
    assert len(listed) == 1
    assert listed[0]["document_id"] == "doc-1"
    assert listed[0]["chunk_count"] == 2

    # Delete
    deleted_count = store.delete_by_document_id("doc-1")
    assert deleted_count == 2
    assert len(store.list_documents()) == 0

