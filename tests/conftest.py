"""
Shared pytest fixtures for the DesignRAG test suite.

Fixtures defined here are automatically available to all test files
without needing to import them.
"""

import pytest


@pytest.fixture
def sample_documents() -> list[dict]:
    """A small set of document dicts matching the loader output format.

    Useful for testing the chunker and other pipeline stages that
    consume loader output without needing to read actual files.
    """
    return [
        {
            "content": (
                "Color theory is fundamental to interior design. "
                "Warm colors like red, orange, and yellow create energy "
                "and excitement. Cool colors like blue, green, and purple "
                "promote calm and relaxation."
            ),
            "metadata": {
                "source_file": "color-theory.md",
                "page_number": 1,
            },
        },
        {
            "content": (
                "The 60-30-10 rule is a classic decorating guideline. "
                "60% of the room should be a dominant color, 30% a "
                "secondary color, and 10% an accent color. This creates "
                "a visually balanced and cohesive space."
            ),
            "metadata": {
                "source_file": "color-theory.md",
                "page_number": 2,
            },
        },
    ]
