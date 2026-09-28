from langchain_core.documents import Document

from document_rag.vector_store import index_documents


class FakeStore:
    def __init__(self) -> None:
        self.was_reset = False
        self.ids: list[str] = []

    def reset_collection(self) -> None:
        self.was_reset = True

    def add_documents(self, documents: list[Document], ids: list[str]) -> None:
        self.ids = ids


def test_index_rebuilds_collection_with_stable_ids() -> None:
    document = Document(
        page_content="same chunk",
        metadata={"path": "guide.pdf", "page": 0, "start_index": 0},
    )
    first_store = FakeStore()
    second_store = FakeStore()

    assert index_documents(first_store, [document]) == 1
    index_documents(second_store, [document])

    assert first_store.was_reset
    assert first_store.ids == second_store.ids
