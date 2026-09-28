from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from document_rag.agentic_rag import AgenticRAGPipeline
from document_rag.rag import RAGPipeline


class RecordingRetriever:
    def __init__(self, responses: list[list[Document]]) -> None:
        self.responses = iter(responses)
        self.queries: list[str] = []

    def invoke(self, query: str) -> list[Document]:
        self.queries.append(query)
        return next(self.responses)


def test_rag_returns_answer_and_deduplicated_sources() -> None:
    docs = [
        Document(
            page_content="Attention weighs tokens.", metadata={"source": "paper.pdf", "page": 0}
        ),
        Document(
            page_content="It builds contextual representations.",
            metadata={"source": "paper.pdf", "page": 0},
        ),
    ]
    pipeline = RAGPipeline(RecordingRetriever([docs]), FakeListChatModel(responses=["Answer [1]."]))

    result = pipeline.ask("What is attention?")

    assert result.text == "Answer [1]."
    assert [source.label for source in result.sources] == ["paper.pdf, p. 1"]


def test_agentic_rag_rewrites_and_retries() -> None:
    weak = [Document(page_content="Unrelated", metadata={"source": "a.pdf", "page": 0})]
    useful = [Document(page_content="Relevant", metadata={"source": "b.pdf", "page": 1})]
    retriever = RecordingRetriever([weak, useful])
    llm = FakeListChatModel(responses=["NO", "better search terms", "Grounded answer [1]."])

    result = AgenticRAGPipeline(retriever, llm).ask("A vague question")

    assert retriever.queries == ["A vague question", "better search terms"]
    assert result.text == "Grounded answer [1]."
    assert result.sources[0].label == "b.pdf, p. 2"
