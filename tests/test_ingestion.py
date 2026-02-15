"""
Tests for the ingestion pipeline — loader and chunker.

These tests run without an OpenAI API key because they only exercise
the file-reading and text-splitting stages of the pipeline.
"""

from pathlib import Path

import pytest

from design_rag.ingestion.chunker import chunk_documents
from design_rag.ingestion.loader import _normalize_text, load_document, load_markdown


class TestLoader:
    """Tests for the document loader functions."""

    def test_load_markdown_returns_single_document(self, tmp_path: Path) -> None:
        """A Markdown file should produce exactly one document dict."""
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is a test document.")

        docs = load_markdown(str(md_file))

        assert len(docs) == 1
        assert docs[0]["content"] == "# Hello\n\nThis is a test document."
        assert docs[0]["metadata"]["source_file"] == "test.md"
        assert docs[0]["metadata"]["page_number"] == 1

    def test_load_markdown_skips_empty_files(self, tmp_path: Path) -> None:
        """Empty or whitespace-only files should return an empty list."""
        md_file = tmp_path / "empty.md"
        md_file.write_text("   \n\n  ")

        docs = load_markdown(str(md_file))

        assert docs == []

    def test_load_document_raises_on_unsupported_format(self, tmp_path: Path) -> None:
        """Unsupported file types should raise a ValueError."""
        txt_file = tmp_path / "notes.txt"
        txt_file.write_text("plain text content")

        with pytest.raises(ValueError, match="Unsupported file type: .txt"):
            load_document(str(txt_file))


class TestNormalizeText:
    """Tests for PDF text normalization."""

    def test_collapses_excessive_spaces(self) -> None:
        """Multiple spaces between words should become a single space."""
        raw = "JAMIE  THOMPSON  STUDIO   Trade  Standards"
        assert _normalize_text(raw) == "JAMIE THOMPSON STUDIO Trade Standards"

    def test_collapses_word_per_line_artifacts(self) -> None:
        """pypdf word-per-line pattern should become spaces."""
        raw = "Anyone\n \nwho\n \nperforms\n \nwork"
        assert _normalize_text(raw) == "Anyone who performs work"

    def test_collapses_all_whitespace_to_flowing_text(self) -> None:
        """Excessive newlines and spaces should all collapse to single spaces."""
        raw = "Section A\n\n\n\n\nSection B"
        assert _normalize_text(raw) == "Section A Section B"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        """Leading/trailing whitespace should be removed."""
        raw = "  hello  world  "
        assert _normalize_text(raw) == "hello world"

    def test_handles_realistic_pypdf_output(self) -> None:
        """Realistic pypdf output with mixed patterns should become clean text."""
        raw = (
            "JAMIE  THOMPSON  STUDIO   Trade  Standards  \n"
            "Deﬁnitions  \n"
            "Trade:\n  \nAnyone\n \nwho\n \nperforms\n \nwork."
        )
        result = _normalize_text(raw)
        assert "  " not in result
        assert "Anyone who performs work." in result


class TestChunker:
    """Tests for the text chunker."""

    def test_chunk_documents_preserves_metadata(
        self, sample_documents: list[dict]
    ) -> None:
        """Each chunk should carry forward the original document metadata."""
        chunks = chunk_documents(sample_documents, chunk_size=500, chunk_overlap=0)

        for chunk in chunks:
            assert "source_file" in chunk["metadata"]
            assert "page_number" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]

    def test_chunk_documents_respects_chunk_size(self) -> None:
        """Chunks should not exceed the specified chunk_size (approximately).

        The splitter may slightly exceed the limit when it can't find a
        clean split point, but should stay reasonably close.
        """
        # Create a document with enough text to require splitting
        long_text = "This is a sentence about design. " * 100
        documents = [
            {
                "content": long_text,
                "metadata": {"source_file": "test.md", "page_number": 1},
            }
        ]

        chunks = chunk_documents(documents, chunk_size=200, chunk_overlap=0)

        assert len(chunks) > 1, "Long text should be split into multiple chunks"
        for chunk in chunks:
            # Allow some tolerance — the splitter tries to break at
            # natural boundaries so chunks may slightly exceed the limit.
            assert len(chunk["content"]) < 400

    def test_chunk_documents_handles_empty_input(self) -> None:
        """An empty document list should return an empty chunk list."""
        chunks = chunk_documents([])

        assert chunks == []
