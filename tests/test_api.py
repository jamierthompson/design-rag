"""
Tests for the FastAPI application endpoints.

These tests use FastAPI's TestClient, which doesn't start a real server —
it runs requests in-process for fast, isolated testing. None of these tests
require an OpenAI API key because they only exercise validation logic and
the health check.
"""

from fastapi.testclient import TestClient

from design_rag.main import app

client = TestClient(app)


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
