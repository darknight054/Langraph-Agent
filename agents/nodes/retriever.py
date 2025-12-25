"""Retriever agent node - fetches relevant documents from vector store."""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from common import get_logger, get_settings, EmbeddingProvider
from agents.state import RAGState
from ingestion.embeddings import get_embedding_model, EmbeddingModel

log = get_logger(__name__)

# Global embedding model for agents (lazy initialized, separate from ingestion)
_agent_embedder: EmbeddingModel | None = None


def get_agent_embedding_model() -> EmbeddingModel:
    """Get or create shared embedding model for agents.

    This singleton ensures the embedding model is loaded only once
    for all agent operations, separate from ingestion pipeline.
    """
    global _agent_embedder
    if _agent_embedder is None:
        log.info("loading_agent_embedding_model")
        _agent_embedder = get_embedding_model()
        log.info("agent_embedding_model_loaded")
    return _agent_embedder


class RetrieverAgent:
    """Agent that retrieves relevant documents from ChromaDB."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = "documents",
        embedding_provider: EmbeddingProvider | str | None = None,
        embedding_model: str | None = None,
        top_k: int = 5,
    ):
        """Initialize retriever agent.

        Args:
            persist_dir: ChromaDB persistence directory
            collection_name: Collection name
            embedding_provider: Embedding provider (openai or huggingface)
            embedding_model: Embedding model name
            top_k: Number of documents to retrieve
        """
        settings = get_settings()
        self.top_k = top_k

        # Use shared embedding model singleton (loaded only once)
        self.embedder: EmbeddingModel = get_agent_embedding_model()

        # Initialize ChromaDB
        persist_path = Path(persist_dir) if persist_dir else settings.chroma_persist_dir

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "retriever_initialized",
            persist_dir=str(persist_path),
            collection=collection_name,
            embedding_provider=settings.embedding_provider.value,
            embedding_model=settings.embedding_model,
            doc_count=self.collection.count(),
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for query text."""
        embeddings = self.embedder.get_embeddings([text])
        return embeddings[0] if embeddings else []

    def retrieve(self, query: str) -> list[Document]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query

        Returns:
            List of LangChain Documents
        """
        log.info("retrieving", query=query[:100], top_k=self.top_k)

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to Documents
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0.0

                documents.append(
                    Document(
                        page_content=doc_text,
                        metadata={
                            **metadata,
                            "distance": distance,
                        },
                    )
                )

        log.info("retrieved", count=len(documents))
        return documents


# Global retriever instance (lazy initialized)
_retriever: RetrieverAgent | None = None


def get_retriever() -> RetrieverAgent:
    """Get or create the global retriever instance."""
    global _retriever
    if _retriever is None:
        settings = get_settings()
        _retriever = RetrieverAgent(top_k=settings.retrieval_top_k)
    return _retriever


def _call_status(state: RAGState, node: str, status: str, details: dict | None = None):
    """Helper to call status callback if present."""
    callback = state.get("status_callback")
    if callback:
        try:
            callback(node, status, details)
        except Exception:
            pass  # Don't fail on callback errors


def retriever_node(state: RAGState) -> RAGState:
    """Retriever node for LangGraph.

    Fetches relevant documents from the vector store based on the contextualized query.
    Uses the contextualized_query (set by query_rewriter) for better search results
    when the user references previous conversation context.

    Args:
        state: Current workflow state

    Returns:
        Updated state with retrieved documents
    """
    # Use contextualized query for retrieval, fall back to original query
    search_query = state.get("contextualized_query") or state.get("query", "")
    original_query = state.get("query", "")

    if not search_query:
        log.warning("retriever_no_query")
        return {
            **state,
            "retrieved_docs": [],
            "error": "No query provided",
        }

    try:
        _call_status(state, "retriever", "Searching documents...", {"query": search_query[:50]})

        retriever = get_retriever()
        documents = retriever.retrieve(search_query)

        _call_status(state, "retriever", f"Found {len(documents)} documents", {"count": len(documents)})

        log.info(
            "retriever_complete",
            original_query=original_query[:50],
            search_query=search_query[:50],
            docs_found=len(documents),
        )

        return {
            **state,
            "retrieved_docs": documents,
            "error": None,
        }

    except Exception as e:
        log.error("retriever_failed", error=str(e))
        _call_status(state, "retriever", f"Error: {str(e)}", None)
        return {
            **state,
            "retrieved_docs": [],
            "error": f"Retrieval failed: {str(e)}",
        }
