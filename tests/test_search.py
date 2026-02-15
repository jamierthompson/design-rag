"""
Tests for the retrieval search module — where clause building and
filtered search behavior.
"""

from design_rag.retrieval.search import _build_where_clause


class TestBuildWhereClause:
    """Tests for converting filter dicts to ChromaDB where clauses."""

    def test_empty_filters_returns_none(self) -> None:
        """No filters should produce no where clause."""
        assert _build_where_clause({}) is None

    def test_single_filter_is_passed_directly(self) -> None:
        """A single filter becomes a simple dict (no $and wrapper)."""
        result = _build_where_clause({"topic_area": "pricing"})
        assert result == {"topic_area": "pricing"}

    def test_multiple_filters_use_and_wrapper(self) -> None:
        """Multiple filters get wrapped in $and for ChromaDB."""
        result = _build_where_clause(
            {
                "topic_area": "pricing",
                "document_type": "narrative",
            }
        )

        assert "$and" in result
        conditions = result["$and"]
        assert {"topic_area": "pricing"} in conditions
        assert {"document_type": "narrative"} in conditions
