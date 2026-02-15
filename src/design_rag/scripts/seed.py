"""
Seed/reset script — clear and re-ingest the corpus in one command.

Usage:
    uv run python -m design_rag.scripts.seed
    uv run python -m design_rag.scripts.seed --docs-dir ./docs
    uv run python -m design_rag.scripts.seed --collection my_collection

This is a development convenience script that:
1. Deletes the entire collection (fresh start)
2. Re-ingests all documents from the docs directory

Useful when you've changed chunking parameters, updated the classifier,
or want to reset the vector store to a known state.
"""

import argparse
import logging
import sys
from pathlib import Path

from design_rag.ingestion.embedder import delete_collection
from design_rag.scripts.bulk_ingest import bulk_ingest

# Default docs directory: ../../docs relative to the project root.
# Path: seed.py → scripts/ → design_rag/ → src/ → design-rag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT.parent / "docs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def seed(
    docs_dir: Path,
    collection_name: str = "default",
) -> None:
    """Delete the collection and re-ingest all documents.

    Args:
        docs_dir: path to the directory containing .pdf and .md files
        collection_name: ChromaDB collection to reset and populate
    """
    # Step 1: Delete the entire collection
    logger.info("Deleting collection '%s'...", collection_name)
    result = delete_collection(collection_name=collection_name)
    logger.info(
        "  Deleted %d chunks from '%s'",
        result["chunks_deleted"],
        collection_name,
    )

    # Step 2: Re-ingest all documents (force=True is redundant after
    # deletion, but bulk_ingest handles it cleanly either way)
    logger.info("Re-ingesting documents from %s...", docs_dir)
    bulk_ingest(
        directory=docs_dir,
        collection_name=collection_name,
        force=False,  # Collection is already empty
    )

    logger.info("Seed complete.")


def main() -> None:
    """CLI entry point for seed/reset."""
    parser = argparse.ArgumentParser(
        description=(
            "Reset and re-ingest the document corpus. "
            "Deletes the collection and re-ingests all documents."
        ),
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DEFAULT_DOCS_DIR,
        help=f"Directory containing documents (default: {DEFAULT_DOCS_DIR})",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="ChromaDB collection name (default: 'default')",
    )

    args = parser.parse_args()

    if not args.docs_dir.is_dir():
        logger.error("Not a directory: %s", args.docs_dir)
        sys.exit(1)

    seed(docs_dir=args.docs_dir, collection_name=args.collection)


if __name__ == "__main__":
    main()
