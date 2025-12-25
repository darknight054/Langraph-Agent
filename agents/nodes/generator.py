"""Generator agent node - generates answers using LLM."""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from common import get_logger, get_settings
from agents.state import RAGState

log = get_logger(__name__)


GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
- Only use information from the provided context to answer the question
- If the context doesn't contain enough information, say so clearly
- Be concise but complete in your answers
- When citing information, mention the source page number if available
- Do not make up or hallucinate information

Context:
{context}""",
    ),
    ("human", "{question}"),
])


class GeneratorAgent:
    """Agent that generates answers using an LLM."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini-2025-04-14",
        temperature: float = 0.0,
    ):
        """Initialize generator agent.

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

        self.chain = GENERATOR_PROMPT | self.llm

        log.info("generator_initialized", model=model)

    def _format_context(self, documents: list[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source = metadata.get("document_name", "Unknown")
            page = metadata.get("page_number", "N/A")

            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def generate(
        self,
        question: str,
        documents: list[Document],
        validation_feedback: str | None = None,
    ) -> str:
        """Generate an answer based on retrieved documents.

        Args:
            question: The user's question (already contextualized by query_rewriter)
            documents: Retrieved context documents
            validation_feedback: Optional feedback from previous validation failure

        Returns:
            Generated answer string
        """
        context = self._format_context(documents)

        # If we have validation feedback, include it in the question
        if validation_feedback:
            question = (
                f"{question}\n\n"
                f"(Previous answer was rejected: {validation_feedback}. "
                "Please provide a more accurate answer.)"
            )

        log.info(
            "generating",
            question=question[:100],
            context_docs=len(documents),
        )

        response = self.chain.invoke({
            "context": context,
            "question": question,
        })

        answer = response.content
        log.info("generated", answer_len=len(answer))

        return answer


# Global generator instance (lazy initialized)
_generator: GeneratorAgent | None = None


def get_generator() -> GeneratorAgent:
    """Get or create the global generator instance."""
    global _generator
    if _generator is None:
        _generator = GeneratorAgent()
    return _generator


def _call_status(state: RAGState, node: str, status: str, details: dict | None = None):
    """Helper to call status callback if present."""
    callback = state.get("status_callback")
    if callback:
        try:
            callback(node, status, details)
        except Exception:
            pass  # Don't fail on callback errors


def generator_node(state: RAGState) -> RAGState:
    """Generator node for LangGraph.

    Generates an answer using the LLM based on retrieved documents.

    Args:
        state: Current workflow state

    Returns:
        Updated state with generated answer
    """
    # Use contextualized query for generation, fall back to original
    query = state.get("contextualized_query") or state.get("query", "")
    documents = state.get("retrieved_docs", [])
    validation_feedback = state.get("validation_feedback")
    retry_count = state.get("retry_count", 0)

    if not query:
        log.warning("generator_no_query")
        return {
            **state,
            "generated_answer": "",
            "error": "No query provided",
        }

    if not documents:
        log.warning("generator_no_docs")
        _call_status(state, "generator", "No documents found", None)
        return {
            **state,
            "generated_answer": "I couldn't find any relevant information to answer your question.",
            "is_valid": True,  # Skip validation for no-docs case
        }

    try:
        if retry_count > 0:
            _call_status(state, "generator", f"Regenerating (attempt {retry_count + 1})...", {"retry": retry_count})
        else:
            _call_status(state, "generator", "Generating answer...", {"docs": len(documents)})

        generator = get_generator()
        answer = generator.generate(
            question=query,
            documents=documents,
            validation_feedback=validation_feedback if retry_count > 0 else None,
        )

        _call_status(state, "generator", "Answer generated", {"length": len(answer)})

        log.info(
            "generator_complete",
            retry_count=retry_count,
            answer_len=len(answer),
        )

        return {
            **state,
            "generated_answer": answer,
            "retry_count": retry_count,
            "error": None,
        }

    except Exception as e:
        log.error("generator_failed", error=str(e))
        _call_status(state, "generator", f"Error: {str(e)}", None)
        return {
            **state,
            "generated_answer": "",
            "error": f"Generation failed: {str(e)}",
        }
