"""
Document loaders for PDF and Markdown files.

Each loader reads a file and returns a list of "document" dicts — one per
logical page or section. Every dict has:
    - content:  the extracted text
    - metadata: info we'll carry through the pipeline for citations later
"""

from pathlib import Path

from pypdf import PdfReader


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
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages
            documents.append({
                "content": text,
                "metadata": {
                    "source_file": source_name,
                    "page_number": page_number,
                },
            })

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
