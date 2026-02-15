"""
Demo queries — curated questions that showcase DesignRAG's capabilities.

Run this script to exercise the full RAG pipeline with questions designed
to highlight different features.

Usage:
    uv run python -m demo.demo_queries

Each query demonstrates a specific capability:
1. Basic Q&A with citations — grounded answer from a single source
2. Metadata filtering — narrowing results to a specific topic area
3. Cross-document synthesis — answer pulling from multiple sources
4. Scoping boundary — showing what RAG correctly *doesn't* know
5. Specific detail retrieval — exact product names, formulas, figures
"""

import json
import time

from design_rag.retrieval.qa import ask

# Each demo query has a title, the question, optional filters, and
# a brief explanation of what it demonstrates for the audience.
DEMO_QUERIES = [
    {
        "title": "1. Grounded Guidance with Citations",
        "description": (
            "A straightforward question about trade standards. "
            "The answer should cite trade-standards.pdf and include "
            "specific details about Trade Agreements."
        ),
        "question": (
            "When is a Trade Agreement required to be signed, and who receives it?"
        ),
        "filters": None,
    },
    {
        "title": "2. Metadata Filtering",
        "description": (
            "Same system, but narrowed to only search within the "
            "'pricing' topic area. This shows how metadata tagging "
            "at ingestion time enables precise retrieval."
        ),
        "question": (
            "How do I calculate the Furnishings Investment Estimate per square foot?"
        ),
        "filters": {"topic_area": "pricing"},
    },
    {
        "title": "3. Specific Product Recommendations",
        "description": (
            "A question about home maintenance that should return "
            "exact product names and brands. Tests whether the "
            "system preserves specific details through chunking."
        ),
        "question": (
            "What products should I use to clean marble surfaces and hardwood floors?"
        ),
        "filters": None,
    },
    {
        "title": "4. Process Knowledge",
        "description": (
            "A question about the revision policy — tests whether "
            "the system can explain a nuanced business process "
            "with specific rules and examples."
        ),
        "question": "What is the studio's revision policy for design meetings?",
        "filters": None,
    },
    {
        "title": "5. Scoping Boundary",
        "description": (
            "A question about something that lives in the "
            "operational database, NOT in the RAG knowledge base. "
            "The system should correctly say it doesn't have this "
            "information — demonstrating that it doesn't hallucinate."
        ),
        "question": (
            "What is the current project timeline for the Smith residence renovation?"
        ),
        "filters": None,
    },
]


def run_demo() -> None:
    """Run all demo queries and display formatted results."""
    print("=" * 70)
    print("  DESIGNRAG DEMO")
    print("=" * 70)
    print()

    for i, query in enumerate(DEMO_QUERIES):
        print(f"{'─' * 70}")
        print(f"  {query['title']}")
        print(f"{'─' * 70}")
        print(f"  What this shows: {query['description']}")
        print()
        print(f"  Question: {query['question']}")
        if query["filters"]:
            print(f"  Filters:  {json.dumps(query['filters'])}")
        print()

        # Time the query for latency discussion
        start = time.time()
        result = ask(
            question=query["question"],
            filters=query["filters"],
        )
        elapsed = time.time() - start

        # Display the answer
        print(f"  Answer ({elapsed:.1f}s, {result['tokens_used']} tokens):")
        print()

        # Word-wrap the answer for terminal readability
        answer_lines = result["answer"].split("\n")
        for line in answer_lines:
            # Indent each line for visual clarity
            print(f"    {line}")
        print()

        # Display sources
        if result["sources"]:
            print("  Sources:")
            for source in result["sources"]:
                print(
                    f"    - {source['file']} "
                    f"(page {source['page']}, "
                    f"relevance: {source['relevance_score']:.2f})"
                )
        else:
            print("  Sources: None")
        print()

        # Pause between queries for live demo pacing
        if i < len(DEMO_QUERIES) - 1:
            input("  Press Enter for next query...")
            print()

    print("=" * 70)
    print("  Demo complete.")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
