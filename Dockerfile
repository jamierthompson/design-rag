FROM python:3.12-slim

WORKDIR /app

# Install uv for fast, reproducible dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies first (cached layer) — this layer only rebuilds
# when pyproject.toml or uv.lock changes, not on every code edit.
COPY pyproject.toml uv.lock ./

# Install dependencies only (cached layer) — --no-install-project skips
# building the project package, which needs src/ that isn't copied yet.
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code and README (needed by hatchling build metadata)
COPY src/ ./src/
COPY README.md ./

# Now install the project package itself
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "design_rag.main:app", "--host", "0.0.0.0", "--port", "8000"]
