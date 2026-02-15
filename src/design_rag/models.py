"""
Pydantic models — define the shape of API requests and responses.

Pydantic does two things:
1. Validation: rejects bad data with helpful error messages
2. Serialization: converts Python objects to/from JSON automatically

FastAPI uses these models to generate OpenAPI docs (the Swagger UI).
"""

from pydantic import BaseModel, Field

# ============================================================
# /query endpoint models
# ============================================================


class QueryRequest(BaseModel):
    """What the client sends to ask a question."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The question to ask",
    )
    collection_name: str = Field(
        default="default",
        description="Which document collection to search",
    )
    n_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve for context",
    )
    filter: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional metadata filters to narrow results. "
            'Example: {"topic_area": "pricing"} or '
            '{"topic_area": "trade_standards", '
            '"document_type": "narrative"}'
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "How do I calculate the FIE?",
                    "collection_name": "default",
                    "n_results": 5,
                },
                {
                    "question": "What are the trade standards?",
                    "filter": {"topic_area": "trade_standards"},
                },
            ]
        }
    }


class Source(BaseModel):
    """A single source citation from the retrieved documents."""

    content: str = Field(..., description="The chunk text that was retrieved")
    file: str = Field(..., description="Source filename")
    page: int = Field(..., description="Page number in the source document")
    relevance_score: float = Field(..., description="How relevant this chunk is (0-1)")


class QueryResponse(BaseModel):
    """What we return after answering a question."""

    answer: str = Field(..., description="The LLM-generated answer")
    sources: list[Source] = Field(
        ..., description="Source chunks used to generate the answer"
    )
    model: str = Field(..., description="The LLM model used for generation")
    tokens_used: int = Field(..., description="Total tokens consumed by the LLM call")


# ============================================================
# /upload endpoint models
# ============================================================


class UploadResponse(BaseModel):
    """What we return after processing an uploaded document."""

    filename: str = Field(..., description="Name of the uploaded file")
    collection: str = Field(..., description="Collection the chunks were stored in")
    chunks_stored: int = Field(..., description="Number of chunks created")


# ============================================================
# /documents endpoint models
# ============================================================


class DocumentInfo(BaseModel):
    """Info about a single document in a collection."""

    filename: str
    chunk_count: int


class DocumentsResponse(BaseModel):
    """List of documents in a collection."""

    collection: str
    documents: list[DocumentInfo]
    total_chunks: int


# ============================================================
# DELETE /documents endpoint models
# ============================================================


class DeleteResponse(BaseModel):
    """What we return after deleting documents from a collection."""

    collection: str = Field(..., description="Collection the chunks were deleted from")
    source_file: str | None = Field(
        None,
        description="Source filename that was deleted (None if entire collection)",
    )
    chunks_deleted: int = Field(..., description="Number of chunks removed")
