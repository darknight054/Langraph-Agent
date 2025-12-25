"""Validator agent node - checks answers for hallucinations and relevance."""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from common import get_logger, get_settings
from agents.state import RAGState

log = get_logger(__name__)


class ValidationResult(BaseModel):
    """Structured output for validation."""

    is_valid: bool = Field(description="Whether the answer is valid")
    confidence: float = Field(
        description="Confidence score from 0 to 1", ge=0.0, le=1.0
    )
    feedback: str = Field(
        description="Feedback explaining why the answer is valid or invalid"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of specific issues found",
    )


VALIDATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict validator that checks AI-generated answers for accuracy and relevance.

Your task is to verify that the answer:
1. Is supported by the provided context (no hallucinations)
2. Actually answers the question asked
3. Does not contain made-up facts or information not in the context
4. Is complete and addresses the key aspects of the question

Context provided to the AI:
{context}

Question asked:
{question}

Answer to validate:
{answer}

Analyze the answer carefully and determine if it should be accepted or rejected.""",
    ),
    (
        "human",
        "Validate this answer and provide your assessment.",
    ),
])


class ValidatorAgent:
    """Agent that validates generated answers for hallucinations."""

    def __init__(
        self,
        model: str = "gpt-4.1-mini-2025-04-14",
        temperature: float = 0.0,
        confidence_threshold: float = 0.7,
    ):
        """Initialize validator agent.

        Args:
            model: OpenAI model to use
            temperature: Generation temperature
            confidence_threshold: Minimum confidence to accept answer
        """
        self.confidence_threshold = confidence_threshold

        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=get_settings().openai_api_key,
        )

        # Use structured output
        self.chain = VALIDATOR_PROMPT | llm.with_structured_output(ValidationResult)

        log.info(
            "validator_initialized",
            model=model,
            threshold=confidence_threshold,
        )

    def _format_context(self, documents: list[Document]) -> str:
        """Format documents into context string."""
        if not documents:
            return "No context provided."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source = metadata.get("document_name", "Unknown")
            page = metadata.get("page_number", "N/A")

            context_parts.append(
                f"[Source {i}: {source}, Page {page}]\n{doc.page_content}"
            )

        return "\n\n---\n\n".join(context_parts)

    def validate(
        self,
        question: str,
        answer: str,
        documents: list[Document],
    ) -> ValidationResult:
        """Validate an answer against the source documents.

        Args:
            question: The original question
            answer: The generated answer to validate
            documents: The source documents used for generation

        Returns:
            ValidationResult with is_valid, confidence, and feedback
        """
        context = self._format_context(documents)

        log.info(
            "validating",
            question=question[:50],
            answer_len=len(answer),
        )

        result = self.chain.invoke({
            "context": context,
            "question": question,
            "answer": answer,
        })

        # Apply confidence threshold
        if result.confidence < self.confidence_threshold:
            result.is_valid = False
            result.issues.append(
                f"Confidence {result.confidence:.2f} below threshold {self.confidence_threshold}"
            )

        log.info(
            "validation_complete",
            is_valid=result.is_valid,
            confidence=result.confidence,
            issues=len(result.issues),
        )

        return result


# Global validator instance (lazy initialized)
_validator: ValidatorAgent | None = None


def get_validator() -> ValidatorAgent:
    """Get or create the global validator instance."""
    global _validator
    if _validator is None:
        _validator = ValidatorAgent()
    return _validator


def _call_status(state: RAGState, node: str, status: str, details: dict | None = None):
    """Helper to call status callback if present."""
    callback = state.get("status_callback")
    if callback:
        try:
            callback(node, status, details)
        except Exception:
            pass  # Don't fail on callback errors


def validator_node(state: RAGState) -> RAGState:
    """Validator node for LangGraph.

    Validates the generated answer for hallucinations and relevance.

    Args:
        state: Current workflow state

    Returns:
        Updated state with validation result
    """
    query = state.get("query", "")
    answer = state.get("generated_answer", "")
    documents = state.get("retrieved_docs", [])
    retry_count = state.get("retry_count", 0)

    # Skip validation if already marked valid (e.g., no-docs case)
    if state.get("is_valid"):
        _call_status(state, "validator", "Skipped (no docs case)", None)
        return state

    if not answer:
        log.warning("validator_no_answer")
        return {
            **state,
            "is_valid": False,
            "validation_feedback": "No answer to validate",
        }

    try:
        _call_status(state, "validator", "Checking answer quality...", None)

        validator = get_validator()
        result = validator.validate(
            question=query,
            answer=answer,
            documents=documents,
        )

        log.info(
            "validator_complete",
            is_valid=result.is_valid,
            confidence=result.confidence,
            retry_count=retry_count,
        )

        if result.is_valid:
            _call_status(state, "validator", f"Answer validated (confidence: {result.confidence:.2f})", {"confidence": result.confidence})
        else:
            _call_status(state, "validator", "Validation failed, retrying...", {"issues": result.issues})

        return {
            **state,
            "is_valid": result.is_valid,
            "validation_feedback": result.feedback,
            "retry_count": retry_count + (0 if result.is_valid else 1),
        }

    except Exception as e:
        log.error("validator_failed", error=str(e))
        _call_status(state, "validator", f"Validation error: {str(e)}", None)
        # On validation error, mark as invalid and let retry logic handle it
        # This surfaces the error rather than silently accepting potentially bad answers
        return {
            **state,
            "is_valid": False,
            "validation_feedback": f"Validation error: {str(e)}",
            "error": f"Validation failed: {str(e)}",
        }
