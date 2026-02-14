from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve the .env file relative to the project root (design-rag/).
# Path: config.py → design_rag/ → src/ → design-rag/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    openai_api_key: str
    chroma_db_path: str = str(PROJECT_ROOT / "chroma_db")
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
