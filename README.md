# DesignRAG

A **Retrieval-Augmented Generation (RAG)** system built for interior design knowledge. Upload design documents (PDFs, Markdown), then ask questions and get cited answers grounded in your source material.

> **Part of the [AI-Native Design Studio](https://github.com/jamierthompson/studio-os)** — a suite of AI tools for a solo interior design practice. DesignRAG is the knowledge layer that makes institutional design expertise searchable and citable.

## How It Works

```
Document  ──→  Load  ──→  Chunk  ──→  Embed  ──→  Store (ChromaDB)
                                                        │
Question  ──→  Embed  ──→  Search (similarity)  ──→  Retrieve top chunks
                                                        │
                                                  Augment prompt  ──→  LLM  ──→  Cited answer
```

1. **Ingestion** — Documents are loaded, split into overlapping chunks, embedded via OpenAI, and stored in ChromaDB.
2. **Retrieval** — Questions are embedded with the same model, then matched against stored chunks using vector similarity search.
3. **Generation** — Retrieved chunks are injected into an LLM prompt that's constrained to answer only from the provided context, with source citations.

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI |
| Vector Store | ChromaDB (persistent local storage) |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| PDF Parsing | pypdf |
| Validation | Pydantic v2 + pydantic-settings |
| Testing | pytest |
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
| `GET` | `/docs` | Interactive Swagger UI (auto-generated) |

## Project Structure

```
design-rag/
├── src/
│   └── design_rag/
│       ├── __init__.py
│       ├── config.py              # Settings via pydantic-settings + .env
│       ├── main.py                # FastAPI app and route handlers
│       ├── models.py              # Pydantic request/response models
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── loader.py          # PDF and Markdown file loaders
│       │   ├── chunker.py         # Text splitting with overlap
│       │   └── embedder.py        # OpenAI embeddings + ChromaDB storage
│       └── retrieval/
│           ├── __init__.py
│           ├── search.py          # Vector similarity search
│           └── qa.py              # RAG prompt construction + LLM call
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures
│   ├── test_ingestion.py          # Loader and chunker unit tests
│   └── test_api.py                # FastAPI endpoint tests
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
uv run pytest -v
```

## License

[MIT](LICENSE)
