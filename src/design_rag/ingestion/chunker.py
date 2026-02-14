"""
Text chunker — splits documents into smaller overlapping pieces.

Why chunk? Embedding models and LLMs have token limits. Smaller chunks also
give more precise search results — a 200-word chunk about "auth" is a better
match than a 5,000-word page that mentions "auth" once.

The overlap ensures we don't cut a sentence in half and lose context at
chunk boundaries.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split documents into overlapping chunks, preserving metadata.

    Args:
        documents: list of {"content": str, "metadata": dict} from the loader
        chunk_size: target character count per chunk
        chunk_overlap: characters of overlap between consecutive chunks

    Returns:
        list of chunk dicts, each with the original metadata plus a
        chunk_index field.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # The splitter tries these separators in order — it prefers to
        # break on paragraph boundaries, then sentences, then words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        text_pieces = splitter.split_text(doc["content"])

        for i, piece in enumerate(text_pieces):
            chunks.append(
                {
                    "content": piece,
                    "metadata": {
                        **doc["metadata"],  # spread the original metadata
                        "chunk_index": i,
                    },
                }
            )

    return chunks
