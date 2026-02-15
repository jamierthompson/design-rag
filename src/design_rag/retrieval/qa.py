"""
RAG Q&A — the final step that ties retrieval and generation together.

The full pipeline:
1. Retrieve relevant chunks via vector similarity search
2. Rerank chunks with an LLM (optional, configurable)
3. Trim context to fit within the token budget
4. Augment the user's question with that context
5. Generate an answer using an LLM, grounded in the retrieved context

By telling the LLM to ONLY use the provided context, we reduce
hallucination and can cite exactly where each answer came from.
"""

import logging

import tiktoken

from design_rag.config import get_settings
from design_rag.ingestion.embedder import get_openai_client
from design_rag.retrieval.reranker import rerank
from design_rag.retrieval.search import search

logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say so.
Always cite which source document(s) your answer comes from."""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}"""


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count the number of tokens in a string using tiktoken.

    tiktoken is OpenAI's fast BPE tokenizer — it gives exact token counts
    for any OpenAI model, so we know precisely how much context fits.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base (used by gpt-4, gpt-4o, gpt-4o-mini)
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def trim_to_token_budget(
    results: list[dict],
    token_budget: int,
    model: str = "gpt-4o-mini",
) -> list[dict]:
    """Select as many chunks as fit within the token budget.

    Chunks are already sorted by relevance (from the search step), so we
    greedily add the most relevant chunks until we'd exceed the budget.
    If a single chunk exceeds the entire budget, we truncate its content
    to fit — better to give partial context than none.

    Args:
        results: search results sorted by relevance (best first)
        token_budget: max tokens allowed for context
        model: the LLM model name (for accurate tokenization)

    Returns:
        the subset of results that fit within the budget
    """
    trimmed = []
    tokens_used = 0

    for result in results:
        chunk_tokens = count_tokens(result["content"], model=model)

        if tokens_used + chunk_tokens <= token_budget:
            # This chunk fits — include it in full
            trimmed.append(result)
            tokens_used += chunk_tokens
        elif tokens_used == 0:
            # First chunk exceeds entire budget — truncate rather than skip.
            # Decode back from tokens to get a clean truncation at a token
            # boundary (no partial characters).
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            tokens = encoding.encode(result["content"])
            truncated_text = encoding.decode(tokens[:token_budget])

            truncated_result = {
                **result,
                "content": truncated_text,
            }
            trimmed.append(truncated_result)
            logger.warning(
                "Truncated oversized chunk from %d to %d tokens",
                chunk_tokens,
                token_budget,
            )
            break
        else:
            # Budget exhausted — stop adding chunks
            break

    if len(trimmed) < len(results):
        logger.info(
            "Trimmed context from %d to %d chunks to fit %d token budget",
            len(results),
            len(trimmed),
            token_budget,
        )

    return trimmed


def build_context(results: list[dict]) -> str:
    """Format search results into a context string for the LLM.

    Each chunk is labeled with its source so the LLM can cite it.
    """
    pieces = []
    for i, result in enumerate(results, start=1):
        source = result["metadata"].get("source_file", "unknown")
        page = result["metadata"].get("page_number", "?")
        pieces.append(f"[Source {i}: {source}, page {page}]\n{result['content']}")
    return "\n\n".join(pieces)


def ask(
    question: str,
    collection_name: str = "default",
    n_results: int = 5,
    filters: dict[str, str] | None = None,
) -> dict:
    """Ask a question and get an answer grounded in the stored documents.

    Args:
        question: the user's question
        collection_name: which collection to search
        n_results: how many chunks to retrieve for context
        filters: optional metadata filters (e.g., {"topic_area": "pricing"})

    Returns:
        dict with: answer, sources, model, tokens_used
    """
    settings = get_settings()
    client = get_openai_client()

    # Step 1: Retrieve relevant chunks (with optional metadata filtering).
    # When reranking is enabled, fetch 2x candidates so the reranker has
    # a larger pool to re-score — the top N survive after reranking.
    fetch_count = n_results * 2 if settings.reranking_enabled else n_results
    results = search(
        question,
        collection_name=collection_name,
        n_results=fetch_count,
        filters=filters,
    )

    if not results:
        return {
            "answer": (
                "No relevant documents found. Please upload some documents first."
            ),
            "sources": [],
            "model": settings.llm_model,
            "tokens_used": 0,
        }

    # Step 2: Rerank (if enabled) — LLM re-scores chunks by relevance,
    # then we take the top n_results after re-sorting
    if settings.reranking_enabled:
        results = rerank(question, results)
        results = results[:n_results]

    # Step 3: Trim context to fit within the token budget
    results = trim_to_token_budget(
        results,
        token_budget=settings.context_token_budget,
        model=settings.llm_model,
    )

    # Step 4: Build the prompt with retrieved context
    context = build_context(results)
    user_message = RAG_USER_TEMPLATE.format(context=context, question=question)

    # Step 5: Call the LLM
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # low temperature for factual, grounded answers
    )

    answer = response.choices[0].message.content
    tokens_used = response.usage.total_tokens if response.usage else 0

    # Step 6: Package the response with sources for citations
    sources = [
        {
            "content": r["content"],
            "file": r["metadata"].get("source_file", "unknown"),
            "page": int(r["metadata"].get("page_number", 0)),
            "relevance_score": r["score"],
        }
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "model": settings.llm_model,
        "tokens_used": tokens_used,
    }
