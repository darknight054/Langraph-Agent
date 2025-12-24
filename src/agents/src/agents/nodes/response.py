"""Response agent node - formats and returns the final response."""

from langchain_core.documents import Document

from common import get_logger
from agents.state import RAGState

log = get_logger(__name__)


def format_sources(documents: list[Document]) -> str:
    """Format source documents for display.

    Args:
        documents: Retrieved documents

    Returns:
        Formatted sources string
    """
    if not documents:
        return ""

    sources = []
    seen = set()

    for doc in documents:
        metadata = doc.metadata
        name = metadata.get("document_name", "Unknown")
        page = metadata.get("page_number", "N/A")

        source_key = f"{name}:{page}"
        if source_key not in seen:
            seen.add(source_key)
            sources.append(f"- {name}, Page {page}")

    if sources:
        return "\n\n**Sources:**\n" + "\n".join(sources)

    return ""


def response_node(state: RAGState) -> RAGState:
    """Response node for LangGraph.

    Formats the final response with sources and metadata.

    Args:
        state: Current workflow state

    Returns:
        Updated state with final response
    """
    answer = state.get("generated_answer", "")
    documents = state.get("retrieved_docs", [])
    is_valid = state.get("is_valid", False)
    retry_count = state.get("retry_count", 0)
    error = state.get("error")

    # Handle error case
    if error:
        log.warning("response_with_error", error=error)
        return {
            **state,
            "final_response": f"I encountered an error: {error}",
        }

    # Handle validation failure after max retries
    if not is_valid:
        log.warning(
            "response_validation_failed",
            retry_count=retry_count,
        )
        feedback = state.get("validation_feedback", "Unknown validation issue")
        return {
            **state,
            "final_response": (
                f"{answer}\n\n"
                f"*Note: This answer may not be fully accurate. "
                f"Validation feedback: {feedback}*"
            ),
        }

    # Format successful response
    sources = format_sources(documents)
    final_response = f"{answer}{sources}"

    log.info(
        "response_complete",
        answer_len=len(answer),
        sources_count=len(documents),
        retry_count=retry_count,
    )

    return {
        **state,
        "final_response": final_response,
    }
