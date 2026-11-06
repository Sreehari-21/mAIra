from __future__ import annotations

from app.ingestion.loaders.base import DocumentPage, RawDocument
from app.ingestion.processor import DocumentProcessor
from app.utils.text_cleaner import clean_text


def test_clean_text():
    dirty = "Hello  world!\r\n\r\n\r\nThis has \x00 null bytes and \t tabs."
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
    assert "\r" not in cleaned
    assert "Hello world!" in cleaned


def test_processor_deduplication_hash():
    text1 = "This is a document content."
    text2 = "This  is a  document content.\r\n"
    # Both normalize to the same text, so hash should match
    h1 = DocumentProcessor.compute_content_hash(text1)
    h2 = DocumentProcessor.compute_content_hash(text2)
    assert h1 == h2


def test_processor_pipeline():
    pages = [DocumentPage(page_number=1, content="Some raw content   with whitespace.")]
    raw_doc = RawDocument(source_path="test.txt", filename="test.txt", pages=pages)

    processed = DocumentProcessor.process(raw_doc)
    assert "content_hash" in processed.metadata
    assert processed.metadata["total_characters"] > 0

