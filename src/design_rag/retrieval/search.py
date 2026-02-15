"""
Vector similarity search — finds the most relevant chunks for a query.

How it works:
1. The user's question gets embedded into a vector (same model as ingestion)
2. ChromaDB compares that vector against all stored chunk vectors
3. It returns the closest matches — "closest" meaning most semantically similar

Hybrid search (keyword + semantic) was evaluated and intentionally skipped.
With a 27-chunk corpus, vector similarity already surfaces the correct
documents for both natural language and exact-term queries (tested with
domain terms like "FIE", "Fiberseal", "change order"). Adding a keyword
index would add complexity for marginal benefit at this corpus size.
Revisit if evaluation scores reveal keyword-specific retrieval failures.
"""

from design_rag.ingestion.embedder import embed_texts, get_chroma_client


def _build_where_clause(filters: dict[str, str]) -> dict | None:
    """Convert a flat filter dict into a ChromaDB `where` clause.

    Single filter:  {"topic_area": "pricing"}
        → {"topic_area": "pricing"}

    Multiple filters:  {"topic_area": "pricing", "document_type": "narrative"}
        → {"$and": [{"topic_area": "pricing"}, {"document_type": "narrative"}]}

    ChromaDB requires the $and wrapper when filtering on multiple fields.
    """
    if not filters:
        return None

    conditions = [{k: v} for k, v in filters.items()]

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


def search(
    query: str,
    collection_name: str = "default",
    n_results: int = 5,
    filters: dict[str, str] | None = None,
) -> list[dict]:
    """Search for chunks most relevant to the query.

    Args:
        query: the user's question as a string
        collection_name: which ChromaDB collection to search
        n_results: how many results to return
        filters: optional metadata filters (e.g., {"topic_area": "pricing"}).
                 Only chunks matching ALL filters are returned.

    Returns:
        list of result dicts, each with:
          - content: the chunk text
          - metadata: source_file, page_number, chunk_index, topic_area, etc.
          - score: relevance score (0-1, higher = more relevant)
    """
    chroma = get_chroma_client()
    collection = chroma.get_or_create_collection(name=collection_name)

    # Embed the query using the same model we used for the documents
    query_embedding = embed_texts([query])[0]

    # Build the ChromaDB where clause from our filter dict
    where = _build_where_clause(filters or {})

    # ChromaDB returns results sorted by distance (ascending)
    query_kwargs: dict = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if where:
        query_kwargs["where"] = where

    raw = collection.query(**query_kwargs)

    # ChromaDB returns parallel lists for each query. We only have one query,
    # so we grab index [0]. The `or [[]]` handles the case where fields are
    # None (they're Optional in ChromaDB's types, but always present with
    # default include settings).
    documents = raw["documents"] or [[]]
    metadatas = raw["metadatas"] or [[]]
    distances = raw["distances"] or [[]]

    results = []
    for doc, metadata, distance in zip(
        documents[0],
        metadatas[0],
        distances[0],
        strict=True,
    ):
        results.append(
            {
                "content": doc,
                "metadata": metadata,
                "score": round(1 - distance, 4),  # convert distance to similarity
            }
        )

    return results
