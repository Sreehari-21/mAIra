from __future__ import annotations

from pathlib import Path
import pytest

from app.ingestion.loaders.text_loader import TextLoader
from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.docx_loader import DocxLoader


def test_text_loader():
    loader = TextLoader()
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample.txt"
    assert loader.can_handle(fixture_path) is True

    raw_doc = loader.load(fixture_path)
    assert raw_doc.filename == "sample.txt"
    assert len(raw_doc.pages) == 1
    assert "Conversational AI Platform" in raw_doc.full_text


def test_markdown_loader():
    loader = TextLoader()
    fixture_path = Path(__file__).resolve().parent.parent / "fixtures" / "sample.md"
    assert loader.can_handle(fixture_path) is True

    raw_doc = loader.load(fixture_path)
    assert raw_doc.filename == "sample.md"
    assert "System Architecture" in raw_doc.full_text


def test_pdf_loader_can_handle():
    loader = PDFLoader()
    assert loader.can_handle(Path("document.pdf")) is True
    assert loader.can_handle(Path("document.txt")) is False


def test_docx_loader_can_handle():
    loader = DocxLoader()
    assert loader.can_handle(Path("document.docx")) is True
    assert loader.can_handle(Path("document.doc")) is False

