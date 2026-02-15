"""
Document classifier — auto-detects topic area and document type using an LLM.

Called once per document during ingestion (not per chunk, not per query),
so the cost is negligible. The LLM sees a text sample and maps it to our
controlled taxonomy defined in metadata.py.

Why LLM-based instead of keyword heuristics:
- More robust as the corpus grows — no need to maintain keyword lists
- Handles ambiguous documents that span multiple keywords
- The corpus is small enough that one cheap LLM call per doc is fine
"""

import json
import logging

from design_rag.config import get_settings
from design_rag.ingestion.embedder import get_openai_client
from design_rag.metadata import DocumentType, TopicArea

logger = logging.getLogger(__name__)

# How much text to send to the LLM for classification.
# 1500 chars is typically enough to capture a document's title, intro,
# and first section — more than enough for accurate classification.
_SAMPLE_LENGTH = 1500

_SYSTEM_PROMPT = f"""\
You are a document classifier for an interior design firm's knowledge base.

Given a text sample from a document, classify it into exactly one topic_area \
and one document_type from the options below.

## Topic Areas
{chr(10).join(f"- {t.value}: {t.__doc__ or ''}" for t in TopicArea)}

## Document Types
{chr(10).join(f"- {t.value}: {t.__doc__ or ''}" for t in DocumentType)}

Respond with ONLY a JSON object, no markdown fences, no explanation:
{{"topic_area": "<value>", "document_type": "<value>"}}"""


def classify_document(
    documents: list[dict],
) -> dict[str, str]:
    """Classify a loaded document by its content.

    Takes the list of document dicts from the loader (one per page/section),
    concatenates a representative text sample, and asks the LLM to classify it
    into our taxonomy.

    Args:
        documents: list of {"content": str, "metadata": dict} from the loader

    Returns:
        dict with "topic_area" and "document_type" as string values
        matching our enum values
    """
    # Build a text sample from the first N characters of the full document.
    # For multi-page PDFs, concatenate pages to get a representative sample.
    full_text = "\n\n".join(doc["content"] for doc in documents)
    sample = full_text[:_SAMPLE_LENGTH]

    settings = get_settings()
    client = get_openai_client()

    response = client.chat.completions.create(
        model=settings.llm_model,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": sample},
        ],
    )

    raw = response.choices[0].message.content or "{}"

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Classifier returned invalid JSON: %s", raw)
        return _fallback()

    # Validate that the returned values are actually in our taxonomy
    topic_area = result.get("topic_area", "")
    document_type = result.get("document_type", "")

    if topic_area not in {t.value for t in TopicArea}:
        logger.warning("Classifier returned unknown topic_area: %s", topic_area)
        return _fallback()

    if document_type not in {t.value for t in DocumentType}:
        logger.warning("Classifier returned unknown document_type: %s", document_type)
        return _fallback()

    logger.info(
        "Classified document as topic_area=%s, document_type=%s",
        topic_area,
        document_type,
    )

    return {"topic_area": topic_area, "document_type": document_type}


def _fallback() -> dict[str, str]:
    """Return safe defaults when classification fails.

    Using explicit 'unknown' values instead of silently guessing ensures
    we can easily find and fix misclassified documents later.
    """
    logger.warning("Using fallback classification — review this document manually")
    return {"topic_area": "operations", "document_type": "narrative"}
