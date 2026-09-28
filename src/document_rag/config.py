"""Runtime settings kept in one place so examples stay uncluttered."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration for ingestion, retrieval, and generation."""

    documents_dir: Path = Path("data/pdf")
    vector_store_dir: Path = Path("data/vector_store")
    collection_name: str = "documents"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    groq_model: str = "openai/gpt-oss-20b"
    chunk_size: int = 1_000
    chunk_overlap: int = 200
    top_k: int = 4

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        return cls(
            documents_dir=Path(os.getenv("DOCUMENTS_DIR", "data/pdf")),
            vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", "data/vector_store")),
            collection_name=os.getenv("CHROMA_COLLECTION", "documents"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("TOP_K", "4")),
        )
