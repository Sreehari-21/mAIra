from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.schemas.knowledge import (
    DocumentDeleteResponse,
    DocumentItem,
    DocumentListResponse,
    IngestionResponse,
    IngestTextRequest,
)
from app.core.config import Settings, get_settings
from app.core.exceptions import DocumentNotFoundError, IngestionError
from app.core.logging import logger
from app.ingestion.pipeline import IngestionPipeline
from app.utils.text_cleaner import sanitize_filename
from app.utils.validators import validate_file_extension
from app.vectorstore.chroma_store import ChromaStore

router = APIRouter(tags=["Knowledge Base"])


def get_pipeline(settings: Settings = Depends(get_settings)) -> IngestionPipeline:
    return IngestionPipeline(settings=settings)


def get_chroma_store(settings: Settings = Depends(get_settings)) -> ChromaStore:
    return ChromaStore(settings=settings)


@router.post(
    "/knowledge/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Text Content into Knowledge Base",
)
def ingest_text_endpoint(
    payload: IngestTextRequest,
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> IngestionResponse:
    """Ingests raw text directly into the local ChromaDB knowledge store."""
    result = pipeline.ingest_raw_text(
        title=payload.title,
        content=payload.content,
        custom_metadata=payload.metadata,
    )
    return IngestionResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        chunk_count=result["chunk_count"],
        character_count=result["character_count"],
        status=result["status"],
        message="Text successfully processed, chunked, and indexed into local vector database.",
    )


@router.post(
    "/knowledge/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest Document File",
)
async def upload_document_endpoint(
    file: UploadFile = File(...),
    pipeline: IngestionPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> IngestionResponse:
    """Uploads and ingests a document file (.txt, .md, .pdf, .docx)."""
    clean_name = sanitize_filename(file.filename or "uploaded_file.txt")
    validate_file_extension(clean_name)

    upload_dir = Path(settings.DATA_DOCUMENTS_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_target = upload_dir / clean_name

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise IngestionError("Uploaded file is empty.")

        with open(temp_target, "wb") as f:
            f.write(content_bytes)

        result = pipeline.ingest_file(temp_target)

        return IngestionResponse(
            document_id=result["document_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            character_count=result["character_count"],
            status=result["status"],
            message=f"File '{clean_name}' successfully ingested into knowledge base.",
        )
    finally:
        # Keep copy in documents dir for auditability
        pass


@router.get(
    "/knowledge",
    response_model=DocumentListResponse,
    summary="List Ingested Documents in Knowledge Base",
)
def list_knowledge_documents(
    store: ChromaStore = Depends(get_chroma_store),
) -> DocumentListResponse:
    """Returns an inventory of all indexed documents in the local ChromaDB vector store."""
    docs = store.list_documents()
    items = [
        DocumentItem(
            document_id=d["document_id"],
            filename=d["filename"],
            chunk_count=d["chunk_count"],
            source_type=d["source_type"],
            ingested_at=d["ingested_at"],
            metadata=d["metadata"],
        )
        for d in docs
    ]
    return DocumentListResponse(documents=items, total_count=len(items))


@router.delete(
    "/knowledge/{document_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete Document from Knowledge Base",
)
def delete_knowledge_document(
    document_id: str,
    pipeline: IngestionPipeline = Depends(get_pipeline),
) -> DocumentDeleteResponse:
    """Removes a document and all associated vector embeddings from the ChromaDB store."""
    try:
        chunks_deleted = pipeline.delete_document(document_id)
        return DocumentDeleteResponse(
            document_id=document_id,
            deleted=True,
            chunks_removed=chunks_deleted,
            message=f"Document '{document_id}' successfully removed from knowledge base.",
        )
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' does not exist in knowledge base.",
        )

