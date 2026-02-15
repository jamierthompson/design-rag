"""
Metadata schema and taxonomy for the DesignRAG corpus.

This module defines the controlled vocabulary used to tag document chunks
during ingestion. Consistent metadata enables filtered retrieval — an agent
can ask "What are our trade standards?" and scope the search to only
trade_standards documents, improving precision.

Why enums instead of free-text strings:
- Prevents typos and inconsistencies across documents
- Makes the valid values discoverable via code / API docs
- Enables reliable filtering (exact match, not fuzzy)
"""

from enum import StrEnum


class TopicArea(StrEnum):
    """The subject domain a document covers.

    Each document maps to exactly one topic area. This is the primary
    filter agents will use to scope their queries.
    """

    TRADE_STANDARDS = "trade_standards"
    """Trade agreements, bidding philosophy, PM fees, change orders."""

    PRICING = "pricing"
    """FIE calculations, flat design fees, case studies, rate guidance."""

    MEETING_PROCEDURES = "meeting_procedures"
    """Note-taking rules, agenda templates, revision policy."""

    PROJECT_PROCESS = "project_process"
    """Project timelines, design phases, client handoff procedures."""

    CLIENT_RELATIONS = "client_relations"
    """Home maintenance guidance, client-facing materials."""

    OPERATIONS = "operations"
    """Hiring, vendor interviews, internal operational processes."""


class DocumentType(StrEnum):
    """The structural format of a document.

    This describes *how* the content is organized, not *what* it's about.
    Different document types may benefit from different chunking strategies
    (e.g., checklists shouldn't be split mid-item).
    """

    NARRATIVE = "narrative"
    """Prose-heavy documents: explanations, philosophy, guidance."""

    CHECKLIST = "checklist"
    """Structured lists of rules, steps, or questions."""

    TEMPLATE = "template"
    """Fill-in-the-blank formats with placeholder fields."""


# ============================================================
# Metadata schema definition
# ============================================================

# These are the metadata fields attached to every chunk in ChromaDB.
# ChromaDB metadata values must be str, int, float, or bool — so we
# store enum values as their string representations.
CHUNK_METADATA_FIELDS = {
    "source_file": "Original filename (e.g., 'trade-standards.pdf')",
    "page_number": "Page number in the source document (1-indexed)",
    "chunk_index": "Position of this chunk within the document (0-indexed)",
    "topic_area": f"Subject domain — one of: {', '.join(TopicArea)}",
    "document_type": f"Structural format — one of: {', '.join(DocumentType)}",
}
