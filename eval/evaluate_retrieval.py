"""
Retrieval evaluation — measures whether the search pipeline finds the right documents.

Runs each test question through vector search and checks whether the expected
source documents appear in the top-k results. This is a retrieval-only evaluation
(no LLM generation step), so it can run quickly and cheaply.

Metrics:
- Precision@k: Of the k retrieved chunks, what fraction came from expected sources?
- Recall@k: Of the expected source documents, what fraction appeared in the top-k?
- MRR (Mean Reciprocal Rank): How high does the first relevant chunk appear?

Usage:
    uv run python -m eval.evaluate_retrieval [--collection NAME] [--k 5]
"""

import argparse
import json
import sys
from pathlib import Path

from design_rag.retrieval.search import search

# eval/ is at the project root, not inside src/
EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict]:
    """Load the hand-labeled Q&A test set."""
    with open(path) as f:
        data = json.load(f)
    return data["test_pairs"]


def evaluate_single_query(
    test_pair: dict,
    collection_name: str,
    k: int,
) -> dict:
    """Run a single query through retrieval and score the results.

    Returns a dict with the raw results and computed metrics for this query.
    """
    question = test_pair["question"]
    expected_sources = set(test_pair["expected_sources"])

    # Run the search — this calls ChromaDB with the embedded query
    results = search(
        query=question,
        collection_name=collection_name,
        n_results=k,
    )

    # Extract source files from the retrieved chunks
    retrieved_sources = [r["metadata"].get("source_file", "unknown") for r in results]

    # Precision@k: what fraction of retrieved chunks are from expected sources?
    # A chunk is "relevant" if its source file is in the expected sources list.
    relevant_count = sum(1 for s in retrieved_sources if s in expected_sources)
    precision_at_k = relevant_count / k if k > 0 else 0.0

    # Recall@k: what fraction of expected sources appear anywhere in top-k?
    # For our test set, most questions map to one source doc, so recall is
    # typically 0 or 1 — but this generalizes to multi-source questions.
    retrieved_source_set = set(retrieved_sources)
    sources_found = expected_sources & retrieved_source_set
    if expected_sources:
        recall_at_k = len(sources_found) / len(expected_sources)
    else:
        recall_at_k = 0.0

    # Reciprocal Rank: 1/position of the first relevant chunk (0 if none found).
    # This rewards retrieving relevant content at the very top of results.
    reciprocal_rank = 0.0
    for i, source in enumerate(retrieved_sources):
        if source in expected_sources:
            reciprocal_rank = 1.0 / (i + 1)
            break

    return {
        "id": test_pair["id"],
        "question": question,
        "topic_area": test_pair["topic_area"],
        "difficulty": test_pair["difficulty"],
        "expected_sources": sorted(expected_sources),
        "retrieved_sources": retrieved_sources,
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "reciprocal_rank": round(reciprocal_rank, 4),
    }


def compute_summary(results: list[dict], k: int) -> dict:
    """Aggregate individual query metrics into summary statistics.

    Computes overall averages and per-topic-area breakdowns so we can
    spot which areas the retrieval pipeline handles well vs. poorly.
    """
    # Overall averages
    avg_precision = sum(r["precision_at_k"] for r in results) / len(results)
    avg_recall = sum(r["recall_at_k"] for r in results) / len(results)
    avg_mrr = sum(r["reciprocal_rank"] for r in results) / len(results)

    # Per-topic breakdowns
    topics: dict[str, list[dict]] = {}
    for r in results:
        topic = r["topic_area"]
        topics.setdefault(topic, []).append(r)

    topic_summaries = {}
    for topic, topic_results in sorted(topics.items()):
        topic_summaries[topic] = {
            "count": len(topic_results),
            "avg_precision_at_k": round(
                sum(r["precision_at_k"] for r in topic_results) / len(topic_results), 4
            ),
            "avg_recall_at_k": round(
                sum(r["recall_at_k"] for r in topic_results) / len(topic_results), 4
            ),
            "avg_mrr": round(
                sum(r["reciprocal_rank"] for r in topic_results) / len(topic_results),
                4,
            ),
        }

    # Find worst-performing queries (recall < 1 means expected source wasn't found)
    failures = [r for r in results if r["recall_at_k"] < 1.0]

    return {
        "k": k,
        "total_queries": len(results),
        "avg_precision_at_k": round(avg_precision, 4),
        "avg_recall_at_k": round(avg_recall, 4),
        "avg_mrr": round(avg_mrr, 4),
        "by_topic": topic_summaries,
        "failed_queries": [
            {"id": f["id"], "question": f["question"], "recall": f["recall_at_k"]}
            for f in failures
        ],
    }


def print_report(summary: dict) -> None:
    """Print a human-readable evaluation report to stdout."""
    print("=" * 60)
    print("RETRIEVAL EVALUATION REPORT")
    print("=" * 60)
    print(f"Queries evaluated: {summary['total_queries']}")
    print(f"k (top results):   {summary['k']}")
    print()
    print("--- Overall Metrics ---")
    print(f"  Precision@{summary['k']}:  {summary['avg_precision_at_k']:.4f}")
    print(f"  Recall@{summary['k']}:     {summary['avg_recall_at_k']:.4f}")
    print(f"  MRR:           {summary['avg_mrr']:.4f}")
    print()
    print("--- By Topic Area ---")
    for topic, stats in summary["by_topic"].items():
        print(f"  {topic} ({stats['count']} queries)")
        print(f"    Precision@{summary['k']}: {stats['avg_precision_at_k']:.4f}")
        print(f"    Recall@{summary['k']}:    {stats['avg_recall_at_k']:.4f}")
        print(f"    MRR:          {stats['avg_mrr']:.4f}")

    if summary["failed_queries"]:
        print()
        print("--- Failed Queries (recall < 1.0) ---")
        for f in summary["failed_queries"]:
            print(f"  [{f['id']}] recall={f['recall']:.2f}: {f['question'][:70]}...")
    else:
        print()
        print("All queries retrieved their expected source documents.")

    print()
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against the hand-labeled test set."
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="ChromaDB collection to search (default: 'default')",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional, prints report to stdout regardless)",
    )
    args = parser.parse_args()

    # Load the test set
    test_pairs = load_test_set()
    print(f"Loaded {len(test_pairs)} test pairs from {TEST_SET_PATH.name}")
    print(f"Evaluating retrieval with k={args.k} on collection '{args.collection}'...")
    print()

    # Run each query through retrieval
    results = []
    for i, pair in enumerate(test_pairs, start=1):
        result = evaluate_single_query(pair, args.collection, args.k)
        status = "PASS" if result["recall_at_k"] >= 1.0 else "FAIL"
        label = f"{pair['id']}: {pair['question'][:50]}..."
        print(f"  [{i}/{len(test_pairs)}] {status} {label}")
        results.append(result)

    print()

    # Compute and display summary
    summary = compute_summary(results, args.k)
    print_report(summary)

    # Optionally save full results to JSON
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "summary": summary,
            "detailed_results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Full results saved to {output_path}")

    # Exit with non-zero status if any queries failed — useful for CI
    if summary["failed_queries"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
