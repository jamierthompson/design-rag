FROM python:3.12-slim

WORKDIR /app

# Install uv for fast, reproducible dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies first (cached layer) — this layer only rebuilds
# when pyproject.toml or uv.lock changes, not on every code edit.
COPY pyproject.toml uv.lock ./

# Create a minimal README so hatchling's build validation passes
RUN echo "# DesignRAG" > README.md

RUN uv sync --frozen --no-dev

# Copy application code
COPY src/ ./src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "design_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
