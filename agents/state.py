"""Shared state schema for RAG workflow."""

from dataclasses import dataclass, field
from typing import TypedDict, Callable, Any

from langchain_core.documents import Document


# Status callback type: (node_name: str, status: str, details: dict | None) -> None
StatusCallback = Callable[[str, str, dict | None], None]


class ChatMessage(TypedDict):
    """A single chat message for conversation history."""

    role: str  # "user" or "assistant"
    content: str


class RAGState(TypedDict, total=False):
    """State shared across all agents in the RAG workflow.

    Attributes:
        query: The user's original question (preserved)
        contextualized_query: Query rewritten with conversation context (used for retrieval)
        chat_history: Previous conversation messages for context
        retrieved_docs: Documents retrieved from vector store
        generated_answer: The LLM-generated answer
        is_valid: Whether the answer passed validation
        validation_feedback: Feedback from validator if answer failed
        retry_count: Number of generation retries attempted
        final_response: The final response to return to user
        error: Error message if something went wrong
    """

    query: str
    contextualized_query: str
    chat_history: list[ChatMessage]
    retrieved_docs: list[Document]
    generated_answer: str
    is_valid: bool
    validation_feedback: str
    retry_count: int
    final_response: str
    error: str | None
    status_callback: StatusCallback | None


@dataclass
class RetrievedChunk:
    """A retrieved chunk with metadata."""

    text: str
    page_number: int
    document_name: str
    distance: float
    document_id: str = ""

    def to_document(self) -> Document:
        """Convert to LangChain Document."""
        return Document(
            page_content=self.text,
            metadata={
                "page_number": self.page_number,
                "document_name": self.document_name,
                "distance": self.distance,
                "document_id": self.document_id,
            },
        )
