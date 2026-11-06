from __future__ import annotations

from pathlib import Path
from typing import Set

from app.core.exceptions import IngestionError
from app.ingestion.loaders.base import BaseLoader, DocumentPage, RawDocument
from app.utils.text_cleaner import clean_text


class TextLoader(BaseLoader):
    """Loader for plain text, Markdown, JSON, and CSV documents."""

    SUPPORTED_EXTENSIONS: Set[str] = {".txt", ".md", ".json", ".csv"}

    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load(self, file_path: Path) -> RawDocument:
        if not file_path.exists():
            raise IngestionError(f"File not found: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            raise IngestionError(f"Failed to read text file '{file_path.name}': {exc}") from exc

        cleaned = clean_text(content)
        pages = [DocumentPage(page_number=1, content=cleaned)]

        return RawDocument(
            source_path=str(file_path.resolve()),
            filename=file_path.name,
            pages=pages,
            metadata={"file_size_bytes": file_path.stat().st_size, "extension": file_path.suffix.lower()},
        )

