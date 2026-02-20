# DesignRAG

A **Retrieval-Augmented Generation (RAG)** system built for interior design knowledge. Upload design documents (PDFs, Markdown), then ask questions and get cited answers grounded in your source material.

## How It Works

```
Document  ──→  Load  ──→  Classify  ──→  Chunk  ──→  Embed  ──→  Store (ChromaDB)
                                                                       │
Question  ──→  Embed  ──→  Search (similarity)  ──→  [Rerank]  ──→  Retrieve top chunks
                                                                       │
                                                          Trim to budget  ──→  LLM  ──→  Cited answer
```

1. **Ingestion** — Documents are loaded, auto-classified by topic area and document type via an LLM, split into overlapping chunks, embedded via OpenAI, and stored in ChromaDB with rich metadata.
2. **Retrieval** — Questions are embedded with the same model, then matched against stored chunks using vector similarity search. Optional LLM-based reranking improves precision. A token budget ensures context fits within the LLM's window.
3. **Generation** — Retrieved chunks are injected into an LLM prompt that's constrained to answer only from the provided context, with source citations.

## Architecture Decisions

**Why RAG-only (no structured data)?** — The broader AI-Native Design Studio separates *institutional knowledge* (how to price, what trade standards to follow, how to take meeting notes) from *operational data* (client records, project timelines, invoices). DesignRAG handles the knowledge layer — unstructured expertise that benefits from semantic search. Structured operational data lives in Postgres, accessed by other parts of the system.

**Why not hybrid search?** — With a 27-chunk corpus, vector similarity already surfaces correct documents for both natural language and exact-term queries (tested with domain terms like "FIE", "Fiberseal", "change order"). Adding a keyword index would add complexity for marginal benefit. This will be revisited if evaluation scores reveal keyword-specific retrieval failures.

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Vector Store | ChromaDB (persistent local storage) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| PDF Parsing | pypdf |
| Token Counting | tiktoken |
| Validation | Pydantic v2 + pydantic-settings |
| Testing | pytest (50 tests) |
| Linting/Formatting | Ruff |
| Package Management | uv |

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Local Development

```bash
# Clone the repository
git clone https://github.com/jamierthompson/design-rag.git
cd design-rag

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Install dependencies
uv sync

# Start the server
uv run uvicorn design_rag.main:app --reload

# Open the interactive API docs
open http://localhost:8000/docs
```

### Ingest Documents

```bash
# Bulk ingest a directory of PDFs and Markdown files
uv run python -m design_rag.scripts.bulk_ingest ../docs

# Reset and re-ingest everything (useful after config changes)
uv run python -m design_rag.scripts.seed
```

### Docker

```bash
# Build and run with Docker Compose
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

docker compose up --build

# API available at http://localhost:8000
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PDF or Markdown file for ingestion |
| `POST` | `/query` | Ask a question and get a cited answer |
| `GET` | `/documents` | List documents in a collection |
| `DELETE` | `/documents` | Remove documents by source file or entire collection |
| `GET` | `/docs` | Interactive Swagger UI (auto-generated) |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I calculate the FIE per square foot?"}'
```

```json
{
  "answer": "The FIE per square foot is calculated by dividing the Total Spent on Furnishings by the Total Square Feet...",
  "sources": [
    {
      "file": "how-to-price-your-services.md",
      "page": 1,
      "relevance_score": 0.82
    }
  ],
  "model": "gpt-4o-mini",
  "tokens_used": 847
}
```

### Example: Filter by Topic Area

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What products are recommended?", "filter": {"topic_area": "client_relations"}}'
```

## Corpus Overview

The knowledge base covers institutional interior design expertise across 5 topic areas:

| Topic Area | Documents | Description |
|---|---|---|
| `trade_standards` | Trade standards guide | Trade definitions, agreements, bidding process, change orders |
| `pricing` | Pricing guide | FIE formula, flat design fees, case study benchmarks by market/experience |
| `meeting_procedures` | Note-taking checklist, Meeting agenda template | Meeting note formatting, action items, revision policy, naming conventions |
| `client_relations` | Home maintenance template | Product recommendations for marble, glass, wood, carpet, upholstery care |
| `operations` | Receiver interview questions | Vetting checklist for delivery/receiving vendors |

**Metadata taxonomy** — Each chunk is tagged with a `topic_area` (6 values) and `document_type` (narrative, checklist, or template), auto-detected by an LLM classifier at ingestion time.

## Chunking Strategy

Documents are split using LangChain's `RecursiveCharacterTextSplitter`:

- **Chunk size:** 1,000 characters (configurable via `CHUNK_SIZE`)
- **Overlap:** 200 characters between chunks (configurable via `CHUNK_OVERLAP`)
- **Separator hierarchy:** `\n\n` → `\n` → `. ` → ` ` → `""`

The recursive approach tries to split on paragraph boundaries first, then sentences, then words — preserving semantic coherence within each chunk. The 200-character overlap ensures that concepts spanning a split point appear in both chunks.

## Evaluation Results

Baseline scores from the evaluation harness (25 hand-labeled Q&A pairs, 5 per topic area):

### Retrieval Quality

| Metric | Score |
|---|---|
| Recall@5 | **1.00** — Every query finds its expected source document |
| MRR | **1.00** — Correct document always at rank #1 |
| Precision@5 | **0.76** — Some top-5 results include non-target docs (expected for small corpus) |

### Answer Quality (LLM-as-Judge, 1-5 scale)

| Metric | Score |
|---|---|
| Accuracy | **4.72** |
| Groundedness | **4.64** |
| Faithfulness | **4.68** |
| Source Match Rate | **100%** |

Run the evaluations yourself:

```bash
# Retrieval only (no LLM calls, fast)
uv run python -m eval.evaluate_retrieval

# Full answer quality (uses OpenAI API)
uv run python -m eval.evaluate_answers
```

## Project Structure

```
design-rag/
├── src/
│   └── design_rag/
│       ├── __init__.py
│       ├── config.py              # Settings via pydantic-settings + .env
│       ├── main.py                # FastAPI app and route handlers
│       ├── models.py              # Pydantic request/response models
│       ├── metadata.py            # TopicArea and DocumentType enums
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loader.py          # PDF and Markdown file loaders
│       │   ├── chunker.py         # Text splitting with overlap
│       │   ├── classifier.py      # LLM-based document classifier
│       │   └── embedder.py        # OpenAI embeddings + ChromaDB storage
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── search.py          # Vector similarity search
│       │   ├── reranker.py        # LLM-based result reranking
│       │   └── qa.py              # RAG prompt + context trimming + LLM call
│       └── scripts/
│           ├── __init__.py
│           ├── bulk_ingest.py     # Bulk document ingestion CLI
│           └── seed.py            # Reset and re-ingest corpus
├── eval/
│   ├── test_set.json              # 25 hand-labeled Q&A test pairs
│   ├── evaluate_retrieval.py      # Retrieval metrics (precision, recall, MRR)
│   ├── evaluate_answers.py        # Answer quality (LLM-as-judge)
│   └── BASELINE.md                # Baseline evaluation scores
├── tests/                         # 50 pytest tests (no API key required)
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## Running Tests

```bash
# Run the full test suite (no API key required)
uv run python -m pytest -v
```

## License

[MIT](LICENSE)
