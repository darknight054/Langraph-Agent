"""Query rewriter agent node - contextualizes queries using chat history."""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from common import get_logger, get_settings
from agents.state import RAGState, ChatMessage

log = get_logger(__name__)


QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query rewriter that makes search queries self-contained.

Given a conversation history and a follow-up question, rewrite the question to be a standalone query
that captures the full context needed for document retrieval.

Rules:
- If the question references something from the conversation (e.g., "it", "that", "this topic", "the previous one"), replace those references with the actual subject
- If the question is already self-contained, return it as-is
- Keep the rewritten query concise and focused on what to search for
- Do NOT answer the question, just rewrite it for better search

Examples:
- History: "User: What is machine learning?" -> Follow-up: "Tell me more about it" -> Rewritten: "Tell me more about machine learning"
- History: "User: Explain neural networks" -> Follow-up: "How does that relate to deep learning?" -> Rewritten: "How do neural networks relate to deep learning?"
- Follow-up: "What is Python?" -> Rewritten: "What is Python?" (already standalone)

Conversation History:
{chat_history}

Follow-up Question: {question}

Rewrite the question to be standalone (or return as-is if already standalone):""",
    ),
])


class QueryRewriterAgent:
    """Rewrites queries to be self-contained using conversation context."""

    def __init__(self, model: str = "gpt-4.1-mini-2025-04-14", temperature: float = 0.0):
        """Initialize query rewriter agent.

        Args:
            model: OpenAI model to use
            temperature: Generation temperature
        """
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )
        self.chain = QUERY_REWRITER_PROMPT | self.llm
        log.info("query_rewriter_initialized", model=model)

    def _format_chat_history(self, chat_history: list[ChatMessage]) -> str:
        """Format chat history for the rewriter."""
        if not chat_history:
            return "No previous conversation."

        formatted = []
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            # Truncate long messages to keep context focused
            content = msg["content"][:500] + "..." if len(msg["content"]) > 500 else msg["content"]
            formatted.append(f"{role}: {content}")

        return "\n".join(formatted)

    def rewrite(self, query: str, chat_history: list[ChatMessage]) -> str:
        """Rewrite a query to be self-contained.

        Args:
            query: The current user query
            chat_history: Previous conversation messages

        Returns:
            Rewritten standalone query
        """
        # Skip rewriting if no history - query is already standalone
        if not chat_history:
            log.info("query_rewrite_skipped", reason="no_history")
            return query

        formatted_history = self._format_chat_history(chat_history)

        log.info("rewriting_query", original=query[:100])

        response = self.chain.invoke({
            "chat_history": formatted_history,
            "question": query,
        })

        rewritten = response.content.strip()

        # If rewriter returns empty, use original
        if not rewritten:
            rewritten = query

        log.info("query_rewritten", original=query[:50], rewritten=rewritten[:50])
        return rewritten


# Global query rewriter instance (lazy initialized)
_query_rewriter: QueryRewriterAgent | None = None


def get_query_rewriter() -> QueryRewriterAgent:
    """Get or create the global query rewriter instance."""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriterAgent()
    return _query_rewriter


def _call_status(state: RAGState, node: str, status: str, details: dict | None = None):
    """Helper to call status callback if present."""
    callback = state.get("status_callback")
    if callback:
        try:
            callback(node, status, details)
        except Exception:
            pass  # Don't fail on callback errors


def query_rewriter_node(state: RAGState) -> RAGState:
    """Query rewriter node for LangGraph.

    Rewrites the query to be self-contained using conversation history.
    This ensures the retriever can find relevant documents even when
    the user references previous conversation context.

    The original query is preserved in state["query"], while the
    contextualized version is stored in state["contextualized_query"].

    Args:
        state: Current workflow state

    Returns:
        Updated state with contextualized_query
    """
    query = state.get("query", "")
    chat_history = state.get("chat_history", [])

    if not query:
        log.warning("query_rewriter_no_query")
        return {
            **state,
            "contextualized_query": "",
        }

    # Skip rewriting if no chat history - query is already standalone
    if not chat_history:
        _call_status(state, "query_rewriter", "Query is standalone", None)
        return {
            **state,
            "contextualized_query": query,  # Use original as contextualized
        }

    try:
        _call_status(state, "query_rewriter", "Contextualizing query...", None)

        rewriter = get_query_rewriter()
        rewritten_query = rewriter.rewrite(query, chat_history)

        # Log if query was changed
        if rewritten_query != query:
            _call_status(
                state,
                "query_rewriter",
                "Query contextualized",
                {"original": query[:50], "rewritten": rewritten_query[:50]},
            )
            log.info(
                "query_rewriter_complete",
                original=query[:50],
                rewritten=rewritten_query[:50],
            )
        else:
            _call_status(state, "query_rewriter", "Query unchanged", None)

        return {
            **state,
            "contextualized_query": rewritten_query,
        }

    except Exception as e:
        log.error("query_rewriter_failed", error=str(e))
        _call_status(state, "query_rewriter", f"Error: {str(e)}", None)
        # On error, use original query as contextualized
        return {
            **state,
            "contextualized_query": query,
        }
