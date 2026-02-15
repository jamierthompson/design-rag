"""
Document loaders for PDF and Markdown files.

Each loader reads a file and returns a list of "document" dicts — one per
logical page or section. Every dict has:
    - content:  the extracted text
    - metadata: info we'll carry through the pipeline for citations later
"""

import re
from pathlib import Path

from pypdf import PdfReader


def _normalize_text(text: str) -> str:
    """Clean up whitespace artifacts from PDF text extraction.

    pypdf often puts individual words on separate lines with whitespace-only
    lines between them (e.g., "word\\n \\nword\\n \\nword"). This function
    collapses that pattern back into readable prose.

    The approach:
    1. Replace the pypdf word-boundary pattern (\\n<whitespace>\\n) with a
       single space — this rejoins words that were split across lines
    2. Collapse remaining excessive whitespace (multiple spaces, runs of
       newlines) into clean single spaces and paragraph breaks
    """
    # Replace the pypdf word-boundary pattern: \n followed by whitespace-only
    # followed by \n. This is NOT a real paragraph break — it's just how pypdf
    # separates words in some PDF layouts.
    text = re.sub(r"\n[ \t]*\n", " ", text)
    # Now collapse any remaining runs of whitespace (spaces, tabs, newlines)
    # into a single space. At this point real paragraph structure from the PDF
    # is already lost (pypdf flattened it), so we produce clean flowing text
    # and let the chunker find its own split points.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_pdf(file_path: str, original_filename: str | None = None) -> list[dict]:
    """Load a PDF and return one document dict per page.

    Uses pypdf to extract text page-by-page. Each page becomes its own
    document so we can track page numbers for citations.

    Args:
        file_path: path to the PDF file on disk
        original_filename: if provided, use this as source_file in metadata
                          (useful when loading from a temp file)
    """
    path = Path(file_path)
    reader = PdfReader(path)
    source_name = original_filename or path.name

    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        text = _normalize_text(raw_text)
        if text:  # skip blank pages
            documents.append(
                {
                    "content": text,
                    "metadata": {
                        "source_file": source_name,
                        "page_number": page_number,
                    },
                }
            )

    return documents


def load_markdown(file_path: str, original_filename: str | None = None) -> list[dict]:
    """Load a Markdown file as a single document.

    Markdown doesn't have pages, so the whole file is one document
    with page_number set to 1.

    Args:
        file_path: path to the Markdown file on disk
        original_filename: if provided, use this as source_file in metadata
    """
    path = Path(file_path)
    text = path.read_text(encoding="utf-8")
    source_name = original_filename or path.name

    if not text.strip():
        return []

    return [
        {
            "content": text,
            "metadata": {
                "source_file": source_name,
                "page_number": 1,
            },
        }
    ]


def load_document(file_path: str, original_filename: str | None = None) -> list[dict]:
    """Load a document, auto-detecting the format by file extension.

    Supported: .pdf, .md
    Raises ValueError for unsupported formats.

    Args:
        file_path: path to the file on disk
        original_filename: if provided, use this as source_file in metadata
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path, original_filename)
    elif suffix == ".md":
        return load_markdown(file_path, original_filename)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
