"""A compact agentic RAG loop with visible decisions."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from document_rag.rag import Answer, RAGPipeline, Retriever, format_context


class AgenticRAGPipeline(RAGPipeline):
    """Retrieve, inspect relevance, and retry once with an LLM-rewritten query.

    The loop is bounded. It lets the model choose the next action without turning this small
    teaching project into a large framework.
    """

    def __init__(self, retriever: Retriever, llm: BaseChatModel) -> None:
        super().__init__(retriever, llm)
        self.last_search_query: str | None = None

    def ask(self, question: str) -> Answer:
        documents = self.retriever.invoke(question)
        self.last_search_query = question

        if documents and self._context_is_relevant(question, format_context(documents)):
            return self.answer_from_documents(question, documents)

        rewritten = self._rewrite_query(question)
        self.last_search_query = rewritten
        retry_documents = self.retriever.invoke(rewritten)
        return self.answer_from_documents(question, retry_documents)

    def _context_is_relevant(self, question: str, context: str) -> bool:
        response = self.llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Does this context contain useful information for answering the question? "
                        "Reply with only YES or NO.\n\n"
                        f"Question: {question}\n\nContext:\n{context}"
                    )
                )
            ]
        )
        return str(response.content).strip().upper().startswith("YES")

    def _rewrite_query(self, question: str) -> str:
        response = self.llm.invoke(
            [
                HumanMessage(
                    content=(
                        "Rewrite this question as one concise semantic-search query. "
                        "Return only the query.\n\n"
                        f"Question: {question}"
                    )
                )
            ]
        )
        return str(response.content).strip()
