"""Chat session management for RAG system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator
from uuid import uuid4

from common import get_logger, get_settings
from agents.graph import create_rag_graph, get_langfuse_handler
from agents.state import RAGState

log = get_logger(__name__)


@dataclass
class Message:
    """A chat message."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sources: list[dict] = field(default_factory=list)


@dataclass
class ChatSession:
    """Manages a chat session with conversation history."""

    session_id: str = field(default_factory=lambda: uuid4().hex[:8])
    messages: list[Message] = field(default_factory=list)
    _graph: object = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the RAG graph."""
        self._graph = create_rag_graph()
        self._langfuse_handler = get_langfuse_handler()
        log.info("chat_session_created", session_id=self.session_id)

    def _build_config(self) -> dict:
        """Build invocation config with optional tracing."""
        config = {
            "metadata": {
                "session_id": self.session_id,
            }
        }

        if self._langfuse_handler:
            config["callbacks"] = [self._langfuse_handler]

        return config

    def _extract_sources(self, state: RAGState) -> list[dict]:
        """Extract source information from state."""
        sources = []
        seen = set()

        for doc in state.get("retrieved_docs", []):
            metadata = doc.metadata
            name = metadata.get("document_name", "Unknown")
            page = metadata.get("page_number", "N/A")

            source_key = f"{name}:{page}"
            if source_key not in seen:
                seen.add(source_key)
                sources.append({
                    "document": name,
                    "page": page,
                    "distance": metadata.get("distance"),
                })

        return sources

    def chat(self, user_message: str) -> Message:
        """Process a user message and return the assistant response.

        Args:
            user_message: The user's question

        Returns:
            Assistant Message with response and sources
        """
        log.info(
            "chat_message",
            session_id=self.session_id,
            message=user_message[:100],
        )

        # Add user message to history
        self.messages.append(
            Message(role="user", content=user_message)
        )

        # Prepare initial state
        initial_state: RAGState = {
            "query": user_message,
            "retrieved_docs": [],
            "generated_answer": "",
            "is_valid": False,
            "validation_feedback": "",
            "retry_count": 0,
            "final_response": "",
            "error": None,
        }

        # Invoke the graph
        config = self._build_config()
        result = self._graph.invoke(initial_state, config=config)

        # Extract response and sources
        response_text = result.get("final_response", "No response generated.")
        sources = self._extract_sources(result)

        # Create assistant message
        assistant_message = Message(
            role="assistant",
            content=response_text,
            sources=sources,
        )

        # Add to history
        self.messages.append(assistant_message)

        log.info(
            "chat_response",
            session_id=self.session_id,
            response_len=len(response_text),
            sources=len(sources),
        )

        return assistant_message

    def get_history(self) -> list[dict]:
        """Get conversation history as list of dicts.

        Returns:
            List of message dictionaries
        """
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "sources": msg.sources,
            }
            for msg in self.messages
        ]

    def clear_history(self):
        """Clear conversation history."""
        self.messages = []
        log.info("history_cleared", session_id=self.session_id)


# Convenience function for quick queries
def ask(question: str) -> str:
    """Ask a single question without maintaining session.

    Args:
        question: The question to ask

    Returns:
        The response string
    """
    session = ChatSession()
    response = session.chat(question)
    return response.content
