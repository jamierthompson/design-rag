"""
Vector similarity search — finds the most relevant chunks for a query.

How it works:
1. The user's question gets embedded into a vector (same model as ingestion)
2. ChromaDB compares that vector against all stored chunk vectors
3. It returns the closest matches — "closest" meaning most semantically similar
"""

from design_rag.ingestion.embedder import embed_texts, get_chroma_client


def search(
    query: str,
    collection_name: str = "default",
    n_results: int = 5,
) -> list[dict]:
    """Search for chunks most relevant to the query.

    Args:
        query: the user's question as a string
        collection_name: which ChromaDB collection to search
        n_results: how many results to return

    Returns:
        list of result dicts, each with:
          - content: the chunk text
          - metadata: source_file, page_number, chunk_index
          - score: relevance score (lower distance = more relevant)
    """
    chroma = get_chroma_client()
    collection = chroma.get_or_create_collection(name=collection_name)

    # Embed the query using the same model we used for the documents
    query_embedding = embed_texts([query])[0]

    # ChromaDB returns results sorted by distance (ascending)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

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
    ):
        results.append({
            "content": doc,
            "metadata": metadata,
            "score": round(1 - distance, 4),  # convert distance to similarity
        })

    return results
