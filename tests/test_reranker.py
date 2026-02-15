"""
Tests for the LLM-based reranker.

These tests mock the OpenAI client so they run without an API key.
We verify that the reranker correctly parses LLM scores, re-sorts
results, and handles error cases gracefully.
"""

from unittest.mock import MagicMock, patch

from design_rag.retrieval.reranker import rerank


def _make_mock_response(content: str) -> MagicMock:
    """Build a mock OpenAI chat completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _make_results() -> list[dict]:
    """Build sample search results for testing."""
    return [
        {
            "content": "Chunk about pricing",
            "metadata": {"source_file": "pricing.md", "page_number": "1"},
            "score": 0.9,
        },
        {
            "content": "Chunk about trades",
            "metadata": {"source_file": "trades.pdf", "page_number": "1"},
            "score": 0.8,
        },
        {
            "content": "Chunk about maintenance",
            "metadata": {"source_file": "maintenance.pdf", "page_number": "1"},
            "score": 0.7,
        },
    ]


class TestRerank:
    """Tests for the rerank function."""

    @patch("design_rag.retrieval.reranker.get_openai_client")
    @patch("design_rag.retrieval.reranker.get_settings")
    def test_resorts_by_llm_scores(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """Results should be re-sorted by the LLM's relevance scores."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        # LLM says chunk 3 is most relevant, chunk 1 least
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "[2, 5, 9]"
        )

        results = _make_results()
        reranked = rerank("What are the maintenance tips?", results)

        # Chunk 3 (score 9/10=0.9) should now be first
        assert reranked[0]["content"] == "Chunk about maintenance"
        assert reranked[0]["score"] == 0.9
        # Chunk 1 (score 2/10=0.2) should be last
        assert reranked[-1]["content"] == "Chunk about pricing"
        assert reranked[-1]["score"] == 0.2

    @patch("design_rag.retrieval.reranker.get_openai_client")
    @patch("design_rag.retrieval.reranker.get_settings")
    def test_preserves_original_similarity_score(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """The original vector similarity score should be preserved."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "[8, 6, 4]"
        )

        results = _make_results()
        reranked = rerank("question", results)

        # Original scores should be in similarity_score field
        for r in reranked:
            assert "similarity_score" in r

    @patch("design_rag.retrieval.reranker.get_openai_client")
    @patch("design_rag.retrieval.reranker.get_settings")
    def test_returns_original_on_invalid_json(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """Invalid JSON from LLM should return original results unchanged."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_mock_response(
            "not valid json"
        )

        results = _make_results()
        reranked = rerank("question", results)

        # Should fall back to original results
        assert reranked == results

    @patch("design_rag.retrieval.reranker.get_openai_client")
    @patch("design_rag.retrieval.reranker.get_settings")
    def test_returns_original_on_wrong_score_count(
        self, mock_settings: MagicMock, mock_client_fn: MagicMock
    ) -> None:
        """Wrong number of scores should return original results."""
        mock_settings.return_value.llm_model = "gpt-4o-mini"
        mock_client = MagicMock()
        mock_client_fn.return_value = mock_client
        # Return 2 scores for 3 chunks
        mock_client.chat.completions.create.return_value = _make_mock_response("[8, 6]")

        results = _make_results()
        reranked = rerank("question", results)

        assert reranked == results

    def test_empty_results_returns_empty(self) -> None:
        """An empty results list should return empty without calling LLM."""
        assert rerank("question", []) == []
