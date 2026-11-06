#!/usr/bin/env python3
"""Batch document ingestion CLI script for the GenAI Conversational Platform."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so app modules can be loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.core.exceptions import GenAIPlatformException
from app.core.logging import logger
from app.ingestion.pipeline import IngestionPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest documents from directory into local ChromaDB knowledge base."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default="./data/documents",
        help="Directory containing document files to ingest (default: ./data/documents)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Optional target ChromaDB collection name",
    )

    args = parser.parse_args()
    settings = get_settings()

    source_path = Path(args.source_dir)
    if not source_path.exists() or not source_path.is_dir():
        logger.error(f"Source directory '{source_path}' does not exist.")
        return 1

    pipeline = IngestionPipeline(settings=settings)

    supported_suffixes = {".txt", ".md", ".json", ".csv", ".pdf", ".docx"}
    files_to_process = [p for p in source_path.iterdir() if p.is_file() and p.suffix.lower() in supported_suffixes]

    if not files_to_process:
        logger.info(f"No supported document files found in '{source_path}'.")
        return 0

    logger.info(f"Found {len(files_to_process)} document(s) to process in '{source_path}'.")

    success_count = 0
    failure_count = 0

    for file_path in files_to_process:
        try:
            logger.info(f"Processing: {file_path.name}...")
            result = pipeline.ingest_file(file_path)
            logger.info(
                f"[✓] Ingested '{result['filename']}' -> Doc ID: {result['document_id']} ({result['chunk_count']} chunks)"
            )
            success_count += 1
        except GenAIPlatformException as exc:
            logger.error(f"[X] Failed to ingest '{file_path.name}': {exc.message}")
            failure_count += 1
        except Exception as exc:
            logger.error(f"[X] Unexpected error ingesting '{file_path.name}': {exc}")
            failure_count += 1

    logger.info("==================================================")
    logger.info(f"Ingestion Run Complete: {success_count} succeeded, {failure_count} failed.")
    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

