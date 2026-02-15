"""
RAG Q&A — the final step that ties retrieval and generation together.

The "RAG" pattern:
1. Retrieve relevant context from the vector store
2. Augment the user's question with that context
3. Generate an answer using an LLM, grounded in the retrieved context

By telling the LLM to ONLY use the provided context, we reduce
hallucination and can cite exactly where each answer came from.
"""

from design_rag.config import get_settings
from design_rag.ingestion.embedder import get_openai_client
from design_rag.retrieval.search import search

RAG_SYSTEM_PROMPT = """Answer the question based ONLY on the following context.
If the context doesn't contain enough information, say so.
Always cite which source document(s) your answer comes from."""

RAG_USER_TEMPLATE = """Context:
{context}

Question: {question}"""


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

    # Step 1: Retrieve relevant chunks (with optional metadata filtering)
    results = search(
        question,
        collection_name=collection_name,
        n_results=n_results,
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

    # Step 2: Build the prompt with retrieved context
    context = build_context(results)
    user_message = RAG_USER_TEMPLATE.format(context=context, question=question)

    # Step 3: Call the LLM
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

    # Step 4: Package the response with sources for citations
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
