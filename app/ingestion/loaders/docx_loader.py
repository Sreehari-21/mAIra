from __future__ import annotations

from pathlib import Path
from typing import List

from app.core.exceptions import IngestionError
from app.ingestion.loaders.base import BaseLoader, DocumentPage, RawDocument
from app.utils.text_cleaner import clean_text


class DocxLoader(BaseLoader):
    """Loader for Microsoft Word (.docx) documents."""

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".docx"

    def load(self, file_path: Path) -> RawDocument:
        if not file_path.exists():
            raise IngestionError(f"Docx file not found: {file_path}")

        try:
            import docx
        except ImportError:
            raise IngestionError(
                "python-docx is required for DOCX parsing. Install with `pip install python-docx`."
            )

        try:
            doc = docx.Document(str(file_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            # Also extract tables if present
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)

            combined_text = "\n\n".join(paragraphs)
            cleaned = clean_text(combined_text)

            if not cleaned:
                raise IngestionError(f"DOCX '{file_path.name}' contains no readable text content.")

            pages = [DocumentPage(page_number=1, content=cleaned)]

            return RawDocument(
                source_path=str(file_path.resolve()),
                filename=file_path.name,
                pages=pages,
                metadata={
                    "total_paragraphs": len(paragraphs),
                    "file_size_bytes": file_path.stat().st_size,
                    "extension": ".docx",
                },
            )
        except Exception as exc:
            if isinstance(exc, IngestionError):
                raise
            raise IngestionError(f"Failed to parse DOCX '{file_path.name}': {exc}") from exc

