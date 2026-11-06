from __future__ import annotations

import pytest
from app.core.exceptions import (
    DocumentNotFoundError,
    GenAIPlatformException,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    QueryValidationError,
    UnsupportedDocumentTypeError,
)


def test_exception_hierarchy():
    exc = OllamaConnectionError(base_url="http://test:11434")
    assert isinstance(exc, GenAIPlatformException)
    assert exc.status_code == 503
    assert exc.code == "OLLAMA_CONNECTION_FAILED"
    assert "http://test:11434" in exc.message


def test_document_not_found():
    exc = DocumentNotFoundError("doc-123")
    assert exc.status_code == 404
    assert exc.code == "DOCUMENT_NOT_FOUND"
    assert exc.details["document_id"] == "doc-123"


def test_unsupported_document_type():
    exc = UnsupportedDocumentTypeError(".xyz", [".txt", ".pdf"])
    assert exc.status_code == 415
    assert exc.code == "UNSUPPORTED_DOCUMENT_TYPE"
    assert exc.details["extension"] == ".xyz"

