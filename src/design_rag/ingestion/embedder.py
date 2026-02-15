"""
Embedder — turns text chunks into vectors and stores them in ChromaDB.

The embedding step converts human-readable text into a list of floats
(a "vector") that captures the *meaning* of the text. Similar meanings
land near each other in vector space, which is what makes semantic search
possible.
"""

import uuid
from typing import cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import Metadata, PyEmbedding
from openai import OpenAI

from design_rag.config import get_settings


def get_chroma_client() -> ClientAPI:
    """Create a ChromaDB client with persistent local storage."""
    settings = get_settings()
    return chromadb.PersistentClient(path=settings.chroma_db_path)


def get_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from settings."""
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Call the OpenAI embeddings API to convert texts into vectors.

    Args:
        texts: list of strings to embed

    Returns:
        list of embedding vectors (each is a list of floats)
    """
    settings = get_settings()
    client = get_openai_client()

    response = client.embeddings.create(
        input=texts,
        model=settings.embedding_model,
    )

    # The API returns embeddings in the same order as the input
    return [item.embedding for item in response.data]


def embed_and_store(
    chunks: list[dict],
    collection_name: str = "default",
) -> dict:
    """Embed chunks and upsert them into a ChromaDB collection.

    Each chunk gets:
      - A unique ID (UUID)
      - Its embedding vector
      - Its text stored as a ChromaDB "document"
      - Its metadata stored as ChromaDB metadata

    Args:
        chunks: list of {"content": str, "metadata": dict} from the chunker
        collection_name: name of the ChromaDB collection to use

    Returns:
        dict with summary: {"collection": str, "chunks_stored": int}
    """
    chroma = get_chroma_client()
    collection = chroma.get_or_create_collection(name=collection_name)

    # Extract the text content from each chunk
    texts = [chunk["content"] for chunk in chunks]

    # Get embeddings from OpenAI
    embeddings = embed_texts(texts)

    # Generate a unique ID for each chunk
    ids = [str(uuid.uuid4()) for _ in chunks]

    # ChromaDB metadata values must be str, int, float, or bool —
    # so we make sure everything is a simple type.
    metadatas: list[dict[str, str]] = [
        {str(k): str(v) for k, v in chunk["metadata"].items()} for chunk in chunks
    ]

    # Upsert into ChromaDB (insert or update if ID already exists)
    # cast() is needed because ChromaDB's type stubs use invariant List[]
    # instead of covariant Sequence[], so list[list[float]] doesn't satisfy
    # List[PyEmbedding] even though it's compatible at runtime.
    collection.upsert(
        ids=ids,
        embeddings=cast(list[PyEmbedding], embeddings),
        documents=texts,
        metadatas=cast(list[Metadata], metadatas),
    )

    return {
        "collection": collection_name,
        "chunks_stored": len(chunks),
    }


def delete_by_source(
    source_file: str,
    collection_name: str = "default",
) -> dict:
    """Delete all chunks from a specific source file in a collection.

    Uses ChromaDB's `where` filter to find chunks matching the source_file
    metadata, then deletes them by ID. This is useful for re-ingesting a
    document after edits — delete the old chunks first, then upload again.

    Args:
        source_file: the filename to match against chunk metadata
        collection_name: name of the ChromaDB collection

    Returns:
        dict with summary: {"collection": str, "chunks_deleted": int}
    """
    chroma = get_chroma_client()

    try:
        collection = chroma.get_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist — nothing to delete
        return {"collection": collection_name, "chunks_deleted": 0}

    # Find all chunk IDs that belong to this source file
    results = collection.get(
        where={"source_file": source_file},
        include=[],
    )
    ids_to_delete = results.get("ids") or []

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    return {
        "collection": collection_name,
        "chunks_deleted": len(ids_to_delete),
    }


def delete_collection(collection_name: str = "default") -> dict:
    """Delete an entire collection from ChromaDB.

    This removes all chunks, embeddings, and metadata for the collection.
    Useful for starting fresh during development or testing.

    Args:
        collection_name: name of the ChromaDB collection to delete

    Returns:
        dict with summary: {"collection": str, "chunks_deleted": int}
    """
    chroma = get_chroma_client()

    try:
        collection = chroma.get_collection(name=collection_name)
        chunk_count = collection.count()
        chroma.delete_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist — nothing to delete
        return {"collection": collection_name, "chunks_deleted": 0}

    return {
        "collection": collection_name,
        "chunks_deleted": chunk_count,
    }
