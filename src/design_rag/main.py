"""
FastAPI application — the HTTP layer that exposes our RAG pipeline.

FastAPI gives us:
- Automatic request validation (via Pydantic models)
- Automatic OpenAPI docs at /docs
- Async support (though we're using sync for simplicity)
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile

from design_rag.ingestion.chunker import chunk_documents
from design_rag.ingestion.embedder import embed_and_store, get_chroma_client
from design_rag.ingestion.loader import load_document
from design_rag.models import (
    DocumentInfo,
    DocumentsResponse,
    QueryRequest,
    QueryResponse,
    Source,
    UploadResponse,
)
from design_rag.retrieval.qa import ask

# Create the FastAPI app instance
app = FastAPI(
    title="DesignRAG",
    description=(
        "RAG system over institutional interior design knowledge — "
        "upload documents and ask questions with cited answers"
    ),
    version="0.1.0",
)


# ============================================================
# Health check — useful for deployment monitoring
# ============================================================


@app.get("/health")
def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy"}


# ============================================================
# POST /upload — ingest a document into the vector store
# ============================================================


@app.post("/upload", response_model=UploadResponse)
def upload_document(
    file: UploadFile,
    collection_name: str = "default",
) -> UploadResponse:
    """Upload a PDF or Markdown file and process it into the vector store.

    The pipeline: load → chunk → embed → store in ChromaDB
    """
    # Validate file type
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".md"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use .pdf or .md",
        )

    # Save the uploaded file to a temp location so our loader can read it
    # (UploadFile is a stream; our loader expects a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Run the ingestion pipeline
        documents = load_document(tmp_path, original_filename=file.filename)
        chunks = chunk_documents(documents)
        result = embed_and_store(chunks, collection_name=collection_name)

        return UploadResponse(
            filename=file.filename,
            collection=result["collection"],
            chunks_stored=result["chunks_stored"],
        )
    finally:
        # Clean up the temp file
        Path(tmp_path).unlink(missing_ok=True)


# ============================================================
# POST /query — ask a question and get an answer with citations
# ============================================================


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    """Ask a question and get an answer grounded in the uploaded documents."""
    result = ask(
        question=request.question,
        collection_name=request.collection_name,
        n_results=request.n_results,
    )

    # Convert the raw dicts from ask() into our Pydantic Source model
    sources = [
        Source(
            content=s["content"],
            file=s["file"],
            page=s["page"],
            relevance_score=s["relevance_score"],
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        answer=result["answer"],
        sources=sources,
        model=result["model"],
        tokens_used=result["tokens_used"],
    )


# ============================================================
# GET /documents — list what's in a collection
# ============================================================


@app.get("/documents", response_model=DocumentsResponse)
def list_documents(collection_name: str = "default") -> DocumentsResponse:
    """List all documents stored in a collection."""
    chroma = get_chroma_client()

    try:
        collection = chroma.get_collection(name=collection_name)
    except Exception:
        # Collection doesn't exist yet
        return DocumentsResponse(
            collection=collection_name,
            documents=[],
            total_chunks=0,
        )

    # Get all metadata from the collection
    all_data = collection.get(include=["metadatas"])
    metadatas = all_data.get("metadatas") or []

    # Count chunks per source file
    # We cast to str because we know source_file is always a string,
    # but ChromaDB's types allow various metadata value types.
    file_counts: dict[str, int] = {}
    for meta in metadatas:
        if meta:
            filename = str(meta.get("source_file", "unknown"))
            file_counts[filename] = file_counts.get(filename, 0) + 1

    documents = [
        DocumentInfo(filename=name, chunk_count=count)
        for name, count in sorted(file_counts.items())
    ]

    return DocumentsResponse(
        collection=collection_name,
        documents=documents,
        total_chunks=len(metadatas),
    )
