"""
Tests for the LLM-based document classifier.

These tests mock the OpenAI client so they run without an API key.
We verify that the classifier correctly parses valid LLM responses,
handles invalid responses gracefully, and samples the right amount
of text.
"""

from unittest.mock import MagicMock, patch

from design_rag.ingestion.classifier import _SAMPLE_LENGTH, classify_document


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _sample_documents(text: str = "Trade standards content...") -> list[dict]:
    """Build a minimal document list matching the loader's output format."""
    return [{"content": text, "metadata": {"source_file": "test.md", "page_number": 1}}]


class TestClassifyDocument:
    """Tests for the classify_document function."""

    @patch("design_rag.ingestion.classifier.get_openai_client")
    @patch("design_rag.ingestion.classifier.get_settings")
    def test_returns_valid_classification(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """A well-formed LLM response should be parsed into our taxonomy."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '{"topic_area": "trade_standards", "document_type": "narrative"}'
        )

        result = classify_document(_sample_documents())

        assert result["topic_area"] == "trade_standards"
        assert result["document_type"] == "narrative"

    @patch("design_rag.ingestion.classifier.get_openai_client")
    @patch("design_rag.ingestion.classifier.get_settings")
    def test_falls_back_on_invalid_json(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """Invalid JSON from the LLM should trigger the fallback, not crash."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "This is not JSON at all"
        )

        result = classify_document(_sample_documents())

        # Fallback values from _fallback()
        assert result["topic_area"] == "operations"
        assert result["document_type"] == "narrative"

    @patch("design_rag.ingestion.classifier.get_openai_client")
    @patch("design_rag.ingestion.classifier.get_settings")
    def test_falls_back_on_unknown_topic_area(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """A topic_area not in our taxonomy should trigger the fallback."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '{"topic_area": "made_up_category", "document_type": "narrative"}'
        )

        result = classify_document(_sample_documents())

        assert result["topic_area"] == "operations"
        assert result["document_type"] == "narrative"

    @patch("design_rag.ingestion.classifier.get_openai_client")
    @patch("design_rag.ingestion.classifier.get_settings")
    def test_truncates_long_documents_to_sample_length(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """Only the first _SAMPLE_LENGTH chars should be sent to the LLM."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            '{"topic_area": "pricing", "document_type": "narrative"}'
        )

        # Create a document much longer than the sample length
        long_text = "x" * (_SAMPLE_LENGTH * 3)
        classify_document(_sample_documents(long_text))

        # Check what was actually sent to the LLM
        call_args = mock_client.chat.completions.create.call_args
        user_message = call_args.kwargs["messages"][1]["content"]
        assert len(user_message) == _SAMPLE_LENGTH
