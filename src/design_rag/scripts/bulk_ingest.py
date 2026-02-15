"""
Bulk ingestion script — process an entire directory of documents at once.

Usage:
    uv run python -m design_rag.scripts.bulk_ingest ./path/to/docs
    uv run python -m design_rag.scripts.bulk_ingest ./docs --collection my_collection
    uv run python -m design_rag.scripts.bulk_ingest ./docs --force  # re-ingest all

This scans a directory for .pdf and .md files, classifies each document
via the LLM classifier, chunks and embeds them, and stores them in
ChromaDB. It's idempotent by default — files already in the collection
are skipped unless --force is passed.
"""

import argparse
import logging
import sys
from pathlib import Path

from design_rag.ingestion.chunker import chunk_documents
from design_rag.ingestion.classifier import classify_document
from design_rag.ingestion.embedder import (
    delete_by_source,
    embed_and_store,
    get_chroma_client,
)
from design_rag.ingestion.loader import load_document

# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS = {".pdf", ".md"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_existing_files(collection_name: str) -> set[str]:
    """Query ChromaDB for source filenames already in a collection.

    Returns a set of filenames so we can skip files that are already
    ingested (idempotency check).
    """
    chroma = get_chroma_client()

    try:
        collection = chroma.get_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist yet — nothing is ingested
        return set()

    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas") or []

    return {str(meta.get("source_file", "")) for meta in metadatas if meta}


def discover_files(directory: Path) -> list[Path]:
    """Find all ingestible files in a directory (non-recursive).

    Returns files sorted alphabetically for predictable ordering.
    """
    files = [
        f
        for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda f: f.name)


def ingest_file(
    file_path: Path,
    collection_name: str,
) -> dict:
    """Run the full ingestion pipeline on a single file.

    Pipeline: load → classify → enrich metadata → chunk → embed → store

    Returns a summary dict with filename, classification, and chunk count.
    """
    # Step 1: Load
    documents = load_document(str(file_path), original_filename=file_path.name)

    if not documents:
        logger.warning("  No content extracted from %s — skipping", file_path.name)
        return {"filename": file_path.name, "chunks_stored": 0, "skipped": True}

    # Step 2: Classify
    classification = classify_document(documents)
    logger.info(
        "  Classified as topic_area=%s, document_type=%s",
        classification["topic_area"],
        classification["document_type"],
    )

    # Step 3: Enrich metadata on every document dict
    for doc in documents:
        doc["metadata"]["topic_area"] = classification["topic_area"]
        doc["metadata"]["document_type"] = classification["document_type"]

    # Step 4: Chunk
    chunks = chunk_documents(documents)

    # Step 5: Embed and store
    result = embed_and_store(chunks, collection_name=collection_name)

    return {
        "filename": file_path.name,
        "chunks_stored": result["chunks_stored"],
        "topic_area": classification["topic_area"],
        "document_type": classification["document_type"],
        "skipped": False,
    }


def bulk_ingest(
    directory: Path,
    collection_name: str = "default",
    force: bool = False,
) -> list[dict]:
    """Ingest all supported documents from a directory into ChromaDB.

    Args:
        directory: path to the directory containing documents
        collection_name: ChromaDB collection to store chunks in
        force: if True, re-ingest files even if already present
               (deletes old chunks first)

    Returns:
        list of summary dicts, one per file processed
    """
    files = discover_files(directory)
    if not files:
        logger.warning("No .pdf or .md files found in %s", directory)
        return []

    logger.info(
        "Found %d file(s) in %s: %s",
        len(files),
        directory,
        ", ".join(f.name for f in files),
    )

    # Check what's already ingested for idempotency
    existing = get_existing_files(collection_name)
    if existing:
        logger.info(
            "Collection '%s' already contains: %s",
            collection_name,
            ", ".join(sorted(existing)),
        )

    results = []
    ingested_count = 0
    skipped_count = 0
    error_count = 0
    total_chunks = 0

    for file_path in files:
        # Idempotency: skip files already in the collection
        if file_path.name in existing and not force:
            logger.info("Skipping %s — already ingested", file_path.name)
            results.append(
                {
                    "filename": file_path.name,
                    "chunks_stored": 0,
                    "skipped": True,
                }
            )
            skipped_count += 1
            continue

        # If forcing re-ingestion, delete old chunks first
        if file_path.name in existing and force:
            logger.info("Re-ingesting %s — deleting old chunks first", file_path.name)
            delete_result = delete_by_source(
                file_path.name, collection_name=collection_name
            )
            logger.info("  Deleted %d old chunks", delete_result["chunks_deleted"])

        logger.info("Ingesting %s ...", file_path.name)

        try:
            result = ingest_file(file_path, collection_name)
            results.append(result)

            if not result.get("skipped"):
                ingested_count += 1
                total_chunks += result["chunks_stored"]
                logger.info("  Stored %d chunks", result["chunks_stored"])
        except Exception:
            error_count += 1
            logger.exception("  Failed to ingest %s", file_path.name)
            results.append(
                {
                    "filename": file_path.name,
                    "chunks_stored": 0,
                    "skipped": False,
                    "error": True,
                }
            )

    # Print summary
    logger.info("=" * 50)
    logger.info("Bulk ingestion complete:")
    logger.info("  Files ingested: %d", ingested_count)
    logger.info("  Files skipped:  %d", skipped_count)
    logger.info("  Errors:         %d", error_count)
    logger.info("  Total chunks:   %d", total_chunks)
    logger.info("  Collection:     %s", collection_name)

    return results


def main() -> None:
    """CLI entry point for bulk ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest a directory of documents into DesignRAG",
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to directory containing .pdf and .md files",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="ChromaDB collection name (default: 'default')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest files even if already present (deletes old chunks first)",
    )

    args = parser.parse_args()

    if not args.directory.is_dir():
        logger.error("Not a directory: %s", args.directory)
        sys.exit(1)

    bulk_ingest(
        directory=args.directory,
        collection_name=args.collection,
        force=args.force,
    )


if __name__ == "__main__":
    main()
