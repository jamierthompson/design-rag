"""
Tests for the FastAPI application endpoints.

These tests use FastAPI's TestClient, which doesn't start a real server —
it runs requests in-process for fast, isolated testing. Most tests don't
require an OpenAI API key because they only exercise validation logic,
the health check, and ChromaDB operations (no embeddings needed).
"""

import chromadb
import pytest
from fastapi.testclient import TestClient

from design_rag.main import app

# raise_server_exceptions=False lets us test our custom error handler
# without the TestClient re-raising the underlying exception.
client = TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def _use_ephemeral_chroma(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the ChromaDB client to an in-memory (ephemeral) instance.

    This lets us test delete operations against a real ChromaDB without
    touching the on-disk database or needing an OpenAI API key. The
    monkeypatch ensures the swap is automatically undone after each test.
    """
    ephemeral = chromadb.Client()

    # Patch everywhere get_chroma_client is imported
    monkeypatch.setattr("design_rag.main.get_chroma_client", lambda: ephemeral)
    monkeypatch.setattr(
        "design_rag.ingestion.embedder.get_chroma_client", lambda: ephemeral
    )


class TestHealthCheck:
    """Tests for the GET /health endpoint."""

    def test_health_check_returns_healthy(self) -> None:
        """The health endpoint should return 200 with status 'healthy'."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestUploadValidation:
    """Tests for input validation on the POST /upload endpoint."""

    def test_upload_rejects_unsupported_file_type(self) -> None:
        """Uploading a .txt file should return a 400 error."""
        response = client.post(
            "/upload",
            files={"file": ("notes.txt", b"plain text content", "text/plain")},
        )

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_upload_rejects_missing_filename(self) -> None:
        """A file upload with no filename should be rejected.

        FastAPI/Starlette's multipart parser returns 422 when the
        filename is empty, before our handler logic even runs.
        """
        response = client.post(
            "/upload",
            files={"file": ("", b"some content", "application/octet-stream")},
        )

        assert response.status_code in (400, 422)


def _seed_collection(chroma_client: chromadb.ClientAPI, name: str = "default") -> int:
    """Helper to seed a ChromaDB collection with fake chunks for testing.

    Creates 3 chunks from two different source files so we can test
    both targeted (by source_file) and full collection deletion.

    Returns the total number of chunks inserted.
    """
    collection = chroma_client.get_or_create_collection(name=name)
    collection.add(
        ids=["chunk-1", "chunk-2", "chunk-3"],
        documents=[
            "Trade standards content",
            "More trade standards content",
            "Pricing philosophy content",
        ],
        metadatas=[
            {"source_file": "trade-standards.pdf", "page_number": "1"},
            {"source_file": "trade-standards.pdf", "page_number": "2"},
            {"source_file": "pricing.md", "page_number": "1"},
        ],
        # Fake embeddings — ChromaDB just needs the right dimensionality
        embeddings=[[0.1] * 10, [0.2] * 10, [0.3] * 10],
    )
    return 3


@pytest.mark.usefixtures("_use_ephemeral_chroma")
class TestDeleteDocuments:
    """Tests for the DELETE /documents endpoint."""

    def test_delete_by_source_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deleting by source_file should only remove chunks from that file."""
        # Get the ephemeral client that our fixture set up
        ephemeral = chromadb.Client()
        monkeypatch.setattr("design_rag.main.get_chroma_client", lambda: ephemeral)
        monkeypatch.setattr(
            "design_rag.ingestion.embedder.get_chroma_client", lambda: ephemeral
        )
        _seed_collection(ephemeral)

        response = client.delete(
            "/documents",
            params={"source_file": "trade-standards.pdf"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["source_file"] == "trade-standards.pdf"
        assert data["chunks_deleted"] == 2

        # Verify the pricing doc chunks survived
        collection = ephemeral.get_collection(name="default")
        assert collection.count() == 1

    def test_delete_entire_collection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Omitting source_file should delete the entire collection."""
        ephemeral = chromadb.Client()
        monkeypatch.setattr("design_rag.main.get_chroma_client", lambda: ephemeral)
        monkeypatch.setattr(
            "design_rag.ingestion.embedder.get_chroma_client", lambda: ephemeral
        )
        _seed_collection(ephemeral)

        response = client.delete("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["source_file"] is None
        assert data["chunks_deleted"] == 3

    def test_delete_nonexistent_collection_returns_zero(self) -> None:
        """Deleting from a collection that doesn't exist should return 0, not error."""
        response = client.delete(
            "/documents",
            params={"collection_name": "nonexistent"},
        )

        assert response.status_code == 200
        assert response.json()["chunks_deleted"] == 0

    def test_delete_nonexistent_source_file_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting a source_file that doesn't exist should return 0, not error."""
        ephemeral = chromadb.Client()
        monkeypatch.setattr("design_rag.main.get_chroma_client", lambda: ephemeral)
        monkeypatch.setattr(
            "design_rag.ingestion.embedder.get_chroma_client", lambda: ephemeral
        )
        _seed_collection(ephemeral)

        response = client.delete(
            "/documents",
            params={"source_file": "nonexistent.pdf"},
        )

        assert response.status_code == 200
        assert response.json()["chunks_deleted"] == 0

        # Verify nothing was actually deleted
        collection = ephemeral.get_collection(name="default")
        assert collection.count() == 3


class TestQueryValidation:
    """Tests for input validation on the POST /query endpoint."""

    def test_rejects_empty_question(self) -> None:
        """An empty question string should be rejected."""
        response = client.post(
            "/query",
            json={"question": ""},
        )
        assert response.status_code == 422

    def test_rejects_missing_question(self) -> None:
        """A request body with no question field should be rejected."""
        response = client.post(
            "/query",
            json={"collection_name": "default"},
        )
        assert response.status_code == 422

    def test_rejects_question_too_long(self) -> None:
        """A question exceeding max_length should be rejected."""
        long_question = "a" * 1001
        response = client.post(
            "/query",
            json={"question": long_question},
        )
        assert response.status_code == 422

    def test_rejects_invalid_n_results(self) -> None:
        """n_results outside the valid range (1-20) should be rejected."""
        response = client.post(
            "/query",
            json={"question": "valid question", "n_results": 0},
        )
        assert response.status_code == 422

        response = client.post(
            "/query",
            json={"question": "valid question", "n_results": 21},
        )
        assert response.status_code == 422


@pytest.mark.usefixtures("_use_ephemeral_chroma")
class TestListDocuments:
    """Tests for the GET /documents endpoint."""

    def test_empty_collection_returns_empty_list(self) -> None:
        """A nonexistent collection should return an empty document list."""
        response = client.get(
            "/documents",
            params={"collection_name": "nonexistent"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []
        assert data["total_chunks"] == 0

    def test_lists_documents_with_chunk_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return document names and their chunk counts."""
        ephemeral = chromadb.Client()
        monkeypatch.setattr("design_rag.main.get_chroma_client", lambda: ephemeral)
        _seed_collection(ephemeral)

        response = client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["total_chunks"] == 3
        assert len(data["documents"]) == 2

        # Documents should be sorted by filename
        filenames = [d["filename"] for d in data["documents"]]
        assert filenames == ["pricing.md", "trade-standards.pdf"]

        # Check chunk counts
        docs_by_name = {d["filename"]: d for d in data["documents"]}
        assert docs_by_name["trade-standards.pdf"]["chunk_count"] == 2
        assert docs_by_name["pricing.md"]["chunk_count"] == 1
