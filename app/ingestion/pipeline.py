from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    DocumentNotFoundError,
    IngestionError,
    UnsupportedDocumentTypeError,
)
from app.core.logging import logger
from app.ingestion.chunker import DocumentChunker
from app.ingestion.loaders.base import BaseLoader, DocumentPage, RawDocument
from app.ingestion.loaders.docx_loader import DocxLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.text_loader import TextLoader
from app.ingestion.metadata import MetadataEnricher
from app.ingestion.processor import DocumentProcessor
from app.providers.ollama.embeddings import OllamaEmbedder
from app.vectorstore.base import BaseVectorStore, VectorDocument
from app.vectorstore.chroma_store import ChromaStore


class IngestionPipeline:
    """End-to-end ingestion pipeline: Load -> Process -> Chunk -> Embed -> Store."""

    def __init__(
        self,
        vector_store: Optional[BaseVectorStore] = None,
        embedder: Optional[OllamaEmbedder] = None,
        chunker: Optional[DocumentChunker] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or ChromaStore(settings=self.settings)
        self.embedder = embedder or OllamaEmbedder(settings=self.settings)
        self.chunker = chunker or DocumentChunker(settings=self.settings)

        # Register standard loaders
        self.loaders: List[BaseLoader] = [
            TextLoader(),
            PDFLoader(),
            DocxLoader(),
        ]

    def _get_loader_for_path(self, path: Path) -> BaseLoader:
        for loader in self.loaders:
            if loader.can_handle(path):
                return loader
        raise UnsupportedDocumentTypeError(
            extension=path.suffix.lower(),
            supported=[ext for l in self.loaders if hasattr(l, "SUPPORTED_EXTENSIONS") for ext in l.SUPPORTED_EXTENSIONS] or [".txt", ".md", ".json", ".csv", ".pdf", ".docx"],
        )

    def _generate_document_id(self, filename: str, content_hash: str) -> str:
        """Generates a deterministic or recognizable document identifier."""
        short_hash = content_hash[:8]
        clean_name = Path(filename).stem[:24].lower().replace(" ", "-")
        return f"doc-{clean_name}-{short_hash}"

    def ingest_file(
        self,
        file_path: Path,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Loads and indexes a physical file from disk."""
        loader = self._get_loader_for_path(file_path)
        raw_doc = loader.load(file_path)
        return self._execute_pipeline(raw_doc, custom_metadata)

    def ingest_raw_text(
        self,
        title: str,
        content: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingests raw text directly (e.g. from an API request)."""
        pages = [DocumentPage(page_number=1, content=content)]
        raw_doc = RawDocument(
            source_path="api-ingestion",
            filename=title,
            pages=pages,
            metadata={"source_type": "api_payload"},
        )
        return self._execute_pipeline(raw_doc, custom_metadata)

    def _execute_pipeline(
        self,
        raw_doc: RawDocument,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Internal execution flow for document ingestion."""
        # 1. Processing and cleaning
        processed_doc = DocumentProcessor.process(raw_doc)
        content_hash = processed_doc.metadata.get("content_hash", "")
        document_id = self._generate_document_id(processed_doc.filename, content_hash)

        # 2. Check for existing document with same content to avoid duplicate ingestion
        try:
            existing_chunks = self.vector_store.get_document_chunks(document_id)
            if existing_chunks:
                logger.info(f"Document '{document_id}' already exists in store with {len(existing_chunks)} chunks. Updating.")
                self.vector_store.delete_by_document_id(document_id)
        except Exception:
            pass  # Document doesn't exist yet

        # 3. Chunking
        chunks = self.chunker.chunk_document(document_id, processed_doc)
        if not chunks:
            raise IngestionError(f"No chunks generated for document '{processed_doc.filename}'.")

        # 4. Metadata enrichment
        doc_meta = MetadataEnricher.create_document_metadata(
            document_id=document_id,
            filename=processed_doc.filename,
            source_type=processed_doc.metadata.get("extension", "text"),
            source_path=processed_doc.source_path,
            content_hash=content_hash,
            total_chunks=len(chunks),
            custom_metadata=custom_metadata,
        )

        vector_docs: List[VectorDocument] = []
        texts_to_embed: List[str] = []

        for chunk in chunks:
            chunk_meta = MetadataEnricher.enrich_chunk_metadata(
                doc_metadata=doc_meta,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                page_number=chunk.page_number,
                char_length=len(chunk.text),
            )
            vector_docs.append(
                VectorDocument(
                    id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk_meta,
                )
            )
            texts_to_embed.append(chunk.text)

        # 5. Embedding generation via local Ollama
        logger.info(f"Generating embeddings for {len(texts_to_embed)} chunks of '{document_id}'")
        embeddings = self.embedder.embed_batch(texts_to_embed)

        # 6. Persistent vector store index
        self.vector_store.add_documents(documents=vector_docs, embeddings=embeddings)
        logger.info(f"Successfully indexed document '{document_id}' ({len(chunks)} chunks)")

        return {
            "document_id": document_id,
            "filename": processed_doc.filename,
            "chunk_count": len(chunks),
            "character_count": processed_doc.metadata.get("total_characters", 0),
            "status": "indexed",
        }

    def delete_document(self, document_id: str) -> int:
        """Removes a document and all its chunks from the vector database."""
        return self.vector_store.delete_by_document_id(document_id)

