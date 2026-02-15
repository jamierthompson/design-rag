"""
LLM-based reranker — re-scores search results using gpt-4o-mini.

Vector similarity search is good at finding chunks that are *about* the
same topic as the question, but it can't judge whether a chunk actually
*answers* the question. The reranker fixes this by asking the LLM to
score each chunk's relevance, then re-sorting by those scores.

This runs in a single LLM call (all chunks scored at once) to keep
latency and cost low. It's configurable and off by default.
"""

import json
import logging

from design_rag.config import get_settings
from design_rag.ingestion.embedder import get_openai_client

logger = logging.getLogger(__name__)

_RERANK_SYSTEM_PROMPT = """\
You are a relevance scorer. Given a question and a list of text chunks, \
score each chunk's relevance to the question on a scale of 0 to 10.

- 10 = directly answers the question
- 7-9 = highly relevant, contains key information
- 4-6 = somewhat relevant, tangentially related
- 1-3 = barely relevant
- 0 = completely irrelevant

Respond with ONLY a JSON array of scores in the same order as the chunks.
Example for 3 chunks: [8, 3, 6]"""

_RERANK_USER_TEMPLATE = """\
Question: {question}

{chunks}"""


def rerank(
    question: str,
    results: list[dict],
) -> list[dict]:
    """Re-score and re-sort search results using an LLM.

    Sends all chunks to the LLM in a single call. Each chunk gets a
    relevance score (0-10), and results are re-sorted by that score
    (highest first). The original similarity score is preserved in the
    metadata; the top-level score is replaced with the reranker's score.

    Args:
        question: the user's original question
        results: search results from the vector similarity step

    Returns:
        the same results re-sorted by LLM relevance scores (0-1 scale)
    """
    if not results:
        return results

    settings = get_settings()
    client = get_openai_client()

    # Format chunks for the LLM prompt
    chunk_text = "\n\n".join(
        f"[Chunk {i}]\n{r['content']}" for i, r in enumerate(results, start=1)
    )

    user_message = _RERANK_USER_TEMPLATE.format(question=question, chunks=chunk_text)

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": _RERANK_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    raw = response.choices[0].message.content or "[]"

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Reranker returned invalid JSON: %s", raw)
        return results

    # Validate we got the right number of scores
    if not isinstance(scores, list) or len(scores) != len(results):
        logger.warning(
            "Reranker returned %d scores for %d chunks — skipping rerank",
            len(scores) if isinstance(scores, list) else 0,
            len(results),
        )
        return results

    # Attach reranker scores and re-sort (highest score first).
    # Normalize 0-10 to 0-1 to match the existing score format.
    reranked = []
    for result, score in zip(results, scores, strict=True):
        reranked.append(
            {
                **result,
                "similarity_score": result["score"],  # preserve original
                "score": round(float(score) / 10, 4),  # normalized reranker score
            }
        )

    reranked.sort(key=lambda r: r["score"], reverse=True)

    logger.info(
        "Reranked %d chunks — scores: %s",
        len(reranked),
        [r["score"] for r in reranked],
    )

    return reranked
