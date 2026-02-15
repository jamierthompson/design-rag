"""
Tests for the bulk ingestion script.

These tests verify file discovery, idempotency logic, and the overall
ingestion flow. OpenAI and ChromaDB calls are mocked so no API key
or database is needed.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from design_rag.scripts.bulk_ingest import (
    SUPPORTED_EXTENSIONS,
    discover_files,
    get_existing_files,
)


class TestDiscoverFiles:
    """Tests for finding ingestible files in a directory."""

    def test_finds_pdf_and_md_files(self, tmp_path: Path) -> None:
        """Should return .pdf and .md files, sorted alphabetically."""
        (tmp_path / "alpha.md").write_text("content")
        (tmp_path / "bravo.pdf").write_bytes(b"fake pdf")
        (tmp_path / "charlie.txt").write_text("ignored")
        (tmp_path / "delta.md").write_text("content")

        files = discover_files(tmp_path)

        assert len(files) == 3
        assert [f.name for f in files] == ["alpha.md", "bravo.pdf", "delta.md"]

    def test_returns_empty_for_no_matches(self, tmp_path: Path) -> None:
        """An empty directory or one with no supported files returns []."""
        (tmp_path / "notes.txt").write_text("not a supported format")

        assert discover_files(tmp_path) == []

    def test_ignores_subdirectories(self, tmp_path: Path) -> None:
        """Only files in the top-level directory are returned."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.md").write_text("content")
        (tmp_path / "top.md").write_text("content")

        files = discover_files(tmp_path)

        assert len(files) == 1
        assert files[0].name == "top.md"

    def test_supported_extensions_are_complete(self) -> None:
        """Verify the set of supported extensions matches what the loader handles."""
        assert {".pdf", ".md"} == SUPPORTED_EXTENSIONS


class TestGetExistingFiles:
    """Tests for the idempotency check against ChromaDB."""

    @patch("design_rag.scripts.bulk_ingest.get_chroma_client")
    def test_returns_filenames_from_collection(self, mock_chroma_fn: MagicMock) -> None:
        """Should extract unique source_file values from collection metadata."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {
            "metadatas": [
                {"source_file": "trade-standards.pdf"},
                {"source_file": "trade-standards.pdf"},
                {"source_file": "pricing.md"},
            ]
        }
        mock_chroma = MagicMock()
        mock_chroma.get_collection.return_value = mock_collection
        mock_chroma_fn.return_value = mock_chroma

        result = get_existing_files("default")

        assert result == {"trade-standards.pdf", "pricing.md"}

    @patch("design_rag.scripts.bulk_ingest.get_chroma_client")
    def test_returns_empty_set_for_nonexistent_collection(
        self, mock_chroma_fn: MagicMock
    ) -> None:
        """A collection that doesn't exist yet should return an empty set."""
        mock_chroma = MagicMock()
        mock_chroma.get_collection.side_effect = Exception("not found")
        mock_chroma_fn.return_value = mock_chroma

        result = get_existing_files("nonexistent")

        assert result == set()
