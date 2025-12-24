"""LangGraph workflow definition for RAG system."""

from typing import Literal

from langgraph.graph import StateGraph, END

from common import get_logger, get_settings
from agents.state import RAGState
from agents.nodes import (
    retriever_node,
    generator_node,
    validator_node,
    response_node,
)

log = get_logger(__name__)


def should_retry(state: RAGState) -> Literal["generator", "response"]:
    """Determine if we should retry generation or proceed to response.

    This is a conditional edge function for the LangGraph workflow.

    Args:
        state: Current workflow state

    Returns:
        "generator" to retry, "response" to proceed to final response
    """
    settings = get_settings()
    max_retries = settings.max_retry_count

    is_valid = state.get("is_valid", False)
    retry_count = state.get("retry_count", 0)

    if is_valid:
        log.info("validation_passed", retry_count=retry_count)
        return "response"

    if retry_count >= max_retries:
        log.warning(
            "max_retries_reached",
            retry_count=retry_count,
            max_retries=max_retries,
        )
        return "response"

    log.info(
        "retrying_generation",
        retry_count=retry_count,
        max_retries=max_retries,
    )
    return "generator"


def create_rag_graph() -> StateGraph:
    """Create the RAG workflow graph.

    The workflow follows this pattern:
        START -> retriever -> generator -> validator -> [should_retry?]
                                                            |
                                              [valid] -> response -> END
                                              [invalid & retries left] -> generator

    Returns:
        Compiled LangGraph StateGraph
    """
    log.info("creating_rag_graph")

    # Create the graph with our state type
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retriever", retriever_node)
    graph.add_node("generator", generator_node)
    graph.add_node("validator", validator_node)
    graph.add_node("response", response_node)

    # Define edges
    # Start with retrieval
    graph.set_entry_point("retriever")

    # Retriever -> Generator
    graph.add_edge("retriever", "generator")

    # Generator -> Validator
    graph.add_edge("generator", "validator")

    # Validator -> Conditional (retry or respond)
    graph.add_conditional_edges(
        "validator",
        should_retry,
        {
            "generator": "generator",
            "response": "response",
        },
    )

    # Response -> END
    graph.add_edge("response", END)

    log.info("rag_graph_created")

    return graph.compile()


def get_langfuse_handler():
    """Get Langfuse callback handler if configured.

    Returns:
        CallbackHandler or None if not configured
    """
    settings = get_settings()

    if not settings.langfuse_enabled:
        log.debug("langfuse_not_configured")
        return None

    try:
        from langfuse.callback import CallbackHandler

        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        log.info("langfuse_handler_created")
        return handler

    except ImportError:
        log.warning("langfuse_not_installed")
        return None
    except Exception as e:
        log.error("langfuse_init_failed", error=str(e))
        return None


def invoke_rag(query: str) -> str:
    """Convenience function to invoke the RAG graph with a query.

    Args:
        query: The user's question

    Returns:
        The final response string
    """
    graph = create_rag_graph()

    # Build config with optional Langfuse tracing
    config = {}
    langfuse_handler = get_langfuse_handler()
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    # Invoke the graph
    initial_state: RAGState = {
        "query": query,
        "retrieved_docs": [],
        "generated_answer": "",
        "is_valid": False,
        "validation_feedback": "",
        "retry_count": 0,
        "final_response": "",
        "error": None,
    }

    result = graph.invoke(initial_state, config=config)

    return result.get("final_response", "No response generated.")
