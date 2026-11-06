from __future__ import annotations

from pathlib import Path
from typing import List

from app.core.exceptions import IngestionError
from app.ingestion.loaders.base import BaseLoader, DocumentPage, RawDocument
from app.utils.text_cleaner import clean_text


class PDFLoader(BaseLoader):
    """Loader for PDF documents using pypdf parser."""

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def load(self, file_path: Path) -> RawDocument:
        if not file_path.exists():
            raise IngestionError(f"PDF file not found: {file_path}")

        try:
            import pypdf
        except ImportError:
            raise IngestionError(
                "pypdf package is required for PDF parsing. Install with `pip install pypdf`."
            )

        pages: List[DocumentPage] = []
        try:
            reader = pypdf.PdfReader(str(file_path))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    raise IngestionError(f"PDF '{file_path.name}' is password encrypted.")

            for page_idx, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text() or ""
                cleaned = clean_text(extracted)
                if cleaned:
                    pages.append(DocumentPage(page_number=page_idx, content=cleaned))

            if not pages:
                raise IngestionError(f"PDF '{file_path.name}' contains no extractable text.")

        except Exception as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(f"Corrupt or unreadable PDF '{file_path.name}': {exc}") from exc

        return RawDocument(
            source_path=str(file_path.resolve()),
            filename=file_path.name,
            pages=pages,
            metadata={
                "total_pages": len(reader.pages),
                "file_size_bytes": file_path.stat().st_size,
                "extension": ".pdf",
            },
        )

