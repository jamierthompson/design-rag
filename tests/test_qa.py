"""
Tests for the Q&A module — token counting and context trimming.

These tests don't require an OpenAI API key because they only exercise
the local token counting and trimming logic (tiktoken runs locally).
"""

from design_rag.retrieval.qa import count_tokens, trim_to_token_budget


def _make_result(content: str, score: float = 0.9) -> dict:
    """Build a minimal search result dict for testing."""
    return {
        "content": content,
        "metadata": {"source_file": "test.md", "page_number": "1"},
        "score": score,
    }


class TestCountTokens:
    """Tests for the token counting function."""

    def test_counts_simple_text(self) -> None:
        """A known string should produce a consistent token count."""
        # "hello world" is 2 tokens in cl100k_base
        tokens = count_tokens("hello world")
        assert tokens == 2

    def test_empty_string_is_zero_tokens(self) -> None:
        """An empty string should be 0 tokens."""
        assert count_tokens("") == 0

    def test_falls_back_on_unknown_model(self) -> None:
        """An unknown model name should still return a count (using fallback)."""
        tokens = count_tokens("hello world", model="not-a-real-model")
        assert tokens > 0


class TestTrimToTokenBudget:
    """Tests for context window trimming."""

    def test_all_chunks_fit(self) -> None:
        """When total tokens are under budget, all chunks are returned."""
        results = [_make_result("hello"), _make_result("world")]
        trimmed = trim_to_token_budget(results, token_budget=100)

        assert len(trimmed) == 2

    def test_drops_chunks_that_exceed_budget(self) -> None:
        """Chunks beyond the budget should be excluded."""
        # Each "word " repeated 50 times is ~50 tokens. Two of those exceed
        # a budget of 60 tokens, so only the first should survive.
        long_text = "word " * 50
        results = [
            _make_result(long_text, score=0.9),
            _make_result(long_text, score=0.5),
        ]
        trimmed = trim_to_token_budget(results, token_budget=60)

        assert len(trimmed) == 1

    def test_truncates_oversized_single_chunk(self) -> None:
        """A single chunk bigger than the budget should be truncated, not skipped."""
        long_text = "word " * 200  # ~200 tokens
        results = [_make_result(long_text)]
        trimmed = trim_to_token_budget(results, token_budget=50)

        assert len(trimmed) == 1
        # The truncated content should be shorter than the original
        assert len(trimmed[0]["content"]) < len(long_text)
        # Token count of truncated content should be at or under budget
        assert count_tokens(trimmed[0]["content"]) <= 50

    def test_preserves_relevance_order(self) -> None:
        """Chunks should be kept in their original (relevance) order."""
        results = [
            _make_result("first", score=0.9),
            _make_result("second", score=0.7),
            _make_result("third", score=0.5),
        ]
        trimmed = trim_to_token_budget(results, token_budget=100)

        assert [r["score"] for r in trimmed] == [0.9, 0.7, 0.5]

    def test_empty_results_returns_empty(self) -> None:
        """An empty results list should return an empty list."""
        assert trim_to_token_budget([], token_budget=100) == []
