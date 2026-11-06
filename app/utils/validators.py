from __future__ import annotations

from typing import Set

from app.core.exceptions import QueryValidationError, UnsupportedDocumentTypeError

SUPPORTED_EXTENSIONS: Set[str] = {".txt", ".md", ".json", ".csv", ".pdf", ".docx"}


def validate_query(query: str, min_length: int = 2, max_length: int = 4000) -> str:
    """Validates user query input.

    Raises:
        QueryValidationError: If query is blank, too short, or exceeds max characters.
    """
    if not query or not query.strip():
        raise QueryValidationError("Query text cannot be empty or whitespace only.")

    stripped = query.strip()
    if len(stripped) < min_length:
        raise QueryValidationError(
            f"Query is too short. Minimum length is {min_length} characters.",
            details={"length": len(stripped), "min_length": min_length},
        )

    if len(stripped) > max_length:
        raise QueryValidationError(
            f"Query exceeds maximum character limit of {max_length}.",
            details={"length": len(stripped), "max_length": max_length},
        )

    return stripped


def validate_file_extension(filename: str, supported: Set[str] = SUPPORTED_EXTENSIONS) -> str:
    """Validates if file extension is supported by the ingestion engine."""
    import os

    _, ext = os.path.splitext(filename)
    lower_ext = ext.lower()

    if lower_ext not in supported:
        raise UnsupportedDocumentTypeError(extension=lower_ext, supported=sorted(list(supported)))

    return lower_ext

