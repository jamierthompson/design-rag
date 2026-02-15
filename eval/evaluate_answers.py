"""
Answer quality evaluation — measures whether DesignRAG generates good answers.

Runs each test question through the full RAG pipeline (retrieve → generate)
and uses an LLM-as-judge to score the answers on three dimensions:

1. Accuracy: Does the answer match the expected answer?
2. Groundedness: Does the answer cite the correct source documents?
3. Faithfulness: Does the answer stay true to source content (no hallucination)?

Each metric is scored 1-5 by the judge LLM. This costs real API calls
(one per test question for generation + one per question for judging).

Usage:
    uv run python -m eval.evaluate_answers [--collection NAME] [--k 5]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from design_rag.config import get_settings
from design_rag.ingestion.embedder import get_openai_client
from design_rag.retrieval.qa import ask

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"

# The judge prompt asks the LLM to evaluate the generated answer against
# the expected answer and source documents. We ask for structured JSON
# output so we can parse scores reliably.
JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a RAG system.
You will be given a question, the expected answer, and the system's
generated answer. Score the generated answer on three dimensions.

Return ONLY a JSON object with this exact structure:
{
  "accuracy": <1-5>,
  "accuracy_reason": "<brief explanation>",
  "groundedness": <1-5>,
  "groundedness_reason": "<brief explanation>",
  "faithfulness": <1-5>,
  "faithfulness_reason": "<brief explanation>"
}

Scoring rubric:
- accuracy (does the answer match the expected answer?):
  5 = Covers all key points from expected answer
  4 = Covers most key points, minor omissions
  3 = Covers some key points, notable gaps
  2 = Partially relevant but misses most key points
  1 = Incorrect or completely irrelevant

- groundedness (does the answer cite sources?):
  5 = Cites correct source documents explicitly
  4 = References sources but not by exact name
  3 = Mentions sources vaguely
  2 = No source citations but content is from sources
  1 = No citations and content appears fabricated

- faithfulness (does the answer avoid hallucination?):
  5 = Every claim is supported by the retrieved context
  4 = Nearly all claims supported, minor extrapolation
  3 = Some unsupported claims mixed with supported ones
  2 = Significant unsupported claims
  1 = Mostly hallucinated content"""

JUDGE_USER_TEMPLATE = """Question: {question}

Expected Answer: {expected_answer}

Generated Answer: {generated_answer}

Source Documents Used: {sources}

Score the generated answer on accuracy, groundedness, and faithfulness."""


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict]:
    """Load the hand-labeled Q&A test set."""
    with open(path) as f:
        data = json.load(f)
    return data["test_pairs"]


def judge_answer(
    question: str,
    expected_answer: str,
    generated_answer: str,
    sources: list[dict],
    model: str,
) -> dict:
    """Use an LLM to score the generated answer.

    Returns a dict with accuracy, groundedness, faithfulness scores (1-5)
    and brief explanations for each. Falls back to zeros on parse failure.
    """
    client = get_openai_client()

    # Format sources for the judge to review
    source_summary = ", ".join(
        f"{s['file']} (page {s['page']}, score {s['relevance_score']:.2f})"
        for s in sources
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": JUDGE_USER_TEMPLATE.format(
                    question=question,
                    expected_answer=expected_answer,
                    generated_answer=generated_answer,
                    sources=source_summary or "None",
                ),
            },
        ],
        temperature=0.0,  # deterministic judging
    )

    raw = response.choices[0].message.content or ""

    try:
        scores = json.loads(raw)
        return {
            "accuracy": int(scores["accuracy"]),
            "accuracy_reason": scores.get("accuracy_reason", ""),
            "groundedness": int(scores["groundedness"]),
            "groundedness_reason": scores.get("groundedness_reason", ""),
            "faithfulness": int(scores["faithfulness"]),
            "faithfulness_reason": scores.get("faithfulness_reason", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        logger.warning("Failed to parse judge response: %s", raw[:200])
        return {
            "accuracy": 0,
            "accuracy_reason": f"Parse error: {raw[:100]}",
            "groundedness": 0,
            "groundedness_reason": "Parse error",
            "faithfulness": 0,
            "faithfulness_reason": "Parse error",
        }


def evaluate_single(
    test_pair: dict,
    collection_name: str,
    k: int,
    judge_model: str,
) -> dict:
    """Run a single question through the full pipeline and judge it.

    Returns a dict with the question, generated answer, scores, and metadata.
    """
    question = test_pair["question"]
    expected_answer = test_pair["expected_answer"]

    # Run the full RAG pipeline
    rag_result = ask(
        question=question,
        collection_name=collection_name,
        n_results=k,
    )

    # Judge the generated answer
    scores = judge_answer(
        question=question,
        expected_answer=expected_answer,
        generated_answer=rag_result["answer"],
        sources=rag_result["sources"],
        model=judge_model,
    )

    # Check if expected sources were used
    retrieved_files = {s["file"] for s in rag_result["sources"]}
    expected_files = set(test_pair["expected_sources"])
    source_match = bool(expected_files & retrieved_files)

    return {
        "id": test_pair["id"],
        "question": question,
        "topic_area": test_pair["topic_area"],
        "difficulty": test_pair["difficulty"],
        "expected_answer": expected_answer,
        "generated_answer": rag_result["answer"],
        "expected_sources": sorted(expected_files),
        "retrieved_sources": [s["file"] for s in rag_result["sources"]],
        "source_match": source_match,
        "tokens_used": rag_result["tokens_used"],
        **scores,
    }


def compute_summary(results: list[dict]) -> dict:
    """Aggregate per-query scores into summary statistics."""
    # Filter out parse errors (score of 0) for cleaner averages
    valid = [r for r in results if r["accuracy"] > 0]

    if not valid:
        return {
            "total_queries": len(results),
            "valid_results": 0,
            "parse_errors": len(results),
            "avg_accuracy": 0,
            "avg_groundedness": 0,
            "avg_faithfulness": 0,
            "source_match_rate": 0,
        }

    avg_accuracy = sum(r["accuracy"] for r in valid) / len(valid)
    avg_groundedness = sum(r["groundedness"] for r in valid) / len(valid)
    avg_faithfulness = sum(r["faithfulness"] for r in valid) / len(valid)
    source_matches = sum(1 for r in valid if r["source_match"])

    # Per-topic breakdown
    topics: dict[str, list[dict]] = {}
    for r in valid:
        topics.setdefault(r["topic_area"], []).append(r)

    topic_summaries = {}
    for topic, topic_results in sorted(topics.items()):
        n = len(topic_results)
        topic_summaries[topic] = {
            "count": n,
            "avg_accuracy": round(sum(r["accuracy"] for r in topic_results) / n, 2),
            "avg_groundedness": round(
                sum(r["groundedness"] for r in topic_results) / n, 2
            ),
            "avg_faithfulness": round(
                sum(r["faithfulness"] for r in topic_results) / n, 2
            ),
        }

    # Worst performers (accuracy < 4)
    weak = [r for r in valid if r["accuracy"] < 4]

    total_tokens = sum(r["tokens_used"] for r in results)

    return {
        "total_queries": len(results),
        "valid_results": len(valid),
        "parse_errors": len(results) - len(valid),
        "avg_accuracy": round(avg_accuracy, 2),
        "avg_groundedness": round(avg_groundedness, 2),
        "avg_faithfulness": round(avg_faithfulness, 2),
        "source_match_rate": round(source_matches / len(valid), 4),
        "total_tokens_used": total_tokens,
        "by_topic": topic_summaries,
        "weak_queries": [
            {
                "id": w["id"],
                "question": w["question"][:70],
                "accuracy": w["accuracy"],
                "reason": w["accuracy_reason"],
            }
            for w in weak
        ],
    }


def print_report(summary: dict) -> None:
    """Print a human-readable answer quality report to stdout."""
    print("=" * 60)
    print("ANSWER QUALITY EVALUATION REPORT")
    print("=" * 60)
    print(f"Queries evaluated: {summary['total_queries']}")
    print(f"Valid results:     {summary['valid_results']}")
    if summary["parse_errors"]:
        print(f"Parse errors:      {summary['parse_errors']}")
    print()
    print("--- Overall Scores (1-5 scale) ---")
    print(f"  Accuracy:     {summary['avg_accuracy']:.2f}")
    print(f"  Groundedness: {summary['avg_groundedness']:.2f}")
    print(f"  Faithfulness: {summary['avg_faithfulness']:.2f}")
    print(f"  Source Match:  {summary['source_match_rate']:.0%}")
    print(f"  Total Tokens:  {summary['total_tokens_used']:,}")
    print()
    print("--- By Topic Area ---")
    for topic, stats in summary["by_topic"].items():
        print(f"  {topic} ({stats['count']} queries)")
        print(f"    Accuracy:     {stats['avg_accuracy']:.2f}")
        print(f"    Groundedness: {stats['avg_groundedness']:.2f}")
        print(f"    Faithfulness: {stats['avg_faithfulness']:.2f}")

    if summary["weak_queries"]:
        print()
        print("--- Weak Queries (accuracy < 4) ---")
        for w in summary["weak_queries"]:
            print(f"  [{w['id']}] accuracy={w['accuracy']}: {w['question'][:60]}...")
            print(f"    Reason: {w['reason']}")
    else:
        print()
        print("All queries scored 4+ on accuracy.")

    print()
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer quality using LLM-as-judge "
            "against the hand-labeled test set."
        ),
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
        help="Number of chunks to retrieve per query (default: 5)",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=("Model for the judge LLM (default: same as generation model)"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save JSON results (optional)",
    )
    args = parser.parse_args()

    # Use the configured LLM model for judging by default
    judge_model = args.judge_model or get_settings().llm_model

    test_pairs = load_test_set()
    print(f"Loaded {len(test_pairs)} test pairs")
    print("Running full RAG pipeline + LLM judge...")
    print(f"Judge model: {judge_model}")
    print()

    results = []
    for i, pair in enumerate(test_pairs, start=1):
        result = evaluate_single(pair, args.collection, args.k, judge_model)
        score = result["accuracy"]
        icon = "PASS" if score >= 4 else "WEAK"
        label = f"{pair['id']}: accuracy={score}"
        print(f"  [{i}/{len(test_pairs)}] {icon} {label}")
        results.append(result)

    print()

    summary = compute_summary(results)
    print_report(summary)

    if args.output:
        output_path = Path(args.output)
        output_data = {
            "summary": summary,
            "detailed_results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Full results saved to {output_path}")

    # Exit non-zero if average accuracy is below 3.5
    if summary["avg_accuracy"] < 3.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
