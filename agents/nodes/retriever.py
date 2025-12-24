"""Retriever agent node - fetches relevant documents from vector store."""

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from openai import OpenAI

from common import get_logger, get_settings
from agents.state import RAGState

log = get_logger(__name__)


class RetrieverAgent:
    """Agent that retrieves relevant documents from ChromaDB."""

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 5,
    ):
        """Initialize retriever agent.

        Args:
            persist_dir: ChromaDB persistence directory
            collection_name: Collection name
            embedding_model: OpenAI embedding model
            top_k: Number of documents to retrieve
        """
        settings = get_settings()
        self.top_k = top_k
        self.embedding_model = embedding_model

        # Initialize OpenAI for embeddings
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

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
            doc_count=self.collection.count(),
        )

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for query text."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[text],
        )
        return response.data[0].embedding

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


def retriever_node(state: RAGState) -> RAGState:
    """Retriever node for LangGraph.

    Fetches relevant documents from the vector store based on the query.

    Args:
        state: Current workflow state

    Returns:
        Updated state with retrieved documents
    """
    query = state.get("query", "")

    if not query:
        log.warning("retriever_no_query")
        return {
            **state,
            "retrieved_docs": [],
            "error": "No query provided",
        }

    try:
        retriever = get_retriever()
        documents = retriever.retrieve(query)

        log.info(
            "retriever_complete",
            query=query[:50],
            docs_found=len(documents),
        )

        return {
            **state,
            "retrieved_docs": documents,
            "error": None,
        }

    except Exception as e:
        log.error("retriever_failed", error=str(e))
        return {
            **state,
            "retrieved_docs": [],
            "error": f"Retrieval failed: {str(e)}",
        }
