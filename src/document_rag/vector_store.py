"""Chroma construction and indexing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Create the local embedding model shared by indexing and retrieval."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def open_vector_store(
    persist_directory: Path, collection_name: str, embedding_model: str
) -> Chroma:
    """Open (or create) a persistent Chroma collection."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=create_embeddings(embedding_model),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(document: Document) -> str:
    """Produce a stable ID, allowing Chroma to update instead of duplicate chunks."""
    identity = "|".join(
        [
            str(document.metadata.get("path", document.metadata.get("source", ""))),
            str(document.metadata.get("page", "")),
            str(document.metadata.get("start_index", "")),
            document.page_content,
        ]
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def index_documents(store: Chroma, documents: list[Document]) -> int:
    """Rebuild the collection from the current chunks and return the count indexed."""
    if not documents:
        return 0
    # A full rebuild keeps the local index in sync when a PDF is edited or removed.
    store.reset_collection()
    store.add_documents(documents=documents, ids=[_chunk_id(doc) for doc in documents])
    return len(documents)
