"""Tests for agent state."""

import pytest


def test_rag_state_import():
    """Test that RAGState can be imported."""
    from agents.state import RAGState

    state: RAGState = {
        "query": "test query",
        "retrieved_docs": [],
        "generated_answer": "",
        "is_valid": False,
        "validation_feedback": "",
        "retry_count": 0,
        "final_response": "",
        "error": None,
    }

    assert state["query"] == "test query"
    assert state["retry_count"] == 0


def test_retrieved_chunk():
    """Test RetrievedChunk dataclass."""
    from agents.state import RetrievedChunk

    chunk = RetrievedChunk(
        text="Sample text",
        page_number=1,
        document_name="test.pdf",
        distance=0.5,
    )

    assert chunk.text == "Sample text"
    assert chunk.page_number == 1

    doc = chunk.to_document()
    assert doc.page_content == "Sample text"
    assert doc.metadata["page_number"] == 1
