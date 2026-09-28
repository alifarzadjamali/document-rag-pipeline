from pathlib import Path

import pytest
from langchain_core.documents import Document

from document_rag.document_loader import find_pdfs, split_documents


def test_find_pdfs_is_recursive_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "z.pdf").touch()
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.pdf").touch()
    (nested / "ignore.txt").touch()

    assert find_pdfs(tmp_path) == [nested / "a.pdf", tmp_path / "z.pdf"]


def test_split_documents_preserves_metadata() -> None:
    source = Document(
        page_content="A short paragraph. " * 20,
        metadata={"source": "guide.pdf", "page": 2},
    )

    chunks = split_documents([source], chunk_size=80, chunk_overlap=10)

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "guide.pdf" for chunk in chunks)
    assert all(chunk.metadata["page"] == 2 for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_overlap_must_be_smaller_than_chunk() -> None:
    with pytest.raises(ValueError, match="chunk_overlap"):
        split_documents([], chunk_size=100, chunk_overlap=100)
