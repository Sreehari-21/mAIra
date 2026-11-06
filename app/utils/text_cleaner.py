from __future__ import annotations

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalizes whitespace and removes control characters while preserving semantic text."""
    if not text:
        return ""

    # Normalize unicode (NFKC replaces compatibility characters with standard forms)
    text = unicodedata.normalize("NFKC", text)

    # Strip null bytes and non-printable control characters except standard whitespace
    text = "".join(ch for ch in text if ch in ("\n", "\r", "\t") or (unicodedata.category(ch)[0] != "C"))

    # Normalize Windows and Mac line breaks to Unix
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace 3 or more consecutive newlines with 2 newlines (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Replace multiple spaces/tabs within a line with a single space
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]

    return "\n".join(lines).strip()


def sanitize_filename(filename: str) -> str:
    """Sanitizes filename by removing path traversal and disallowed characters."""
    clean = re.sub(r'[\\/*?:"<>|]', "", filename)
    clean = re.sub(r"\.\.+", ".", clean)
    return clean.strip(" .")

