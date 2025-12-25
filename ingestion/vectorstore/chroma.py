"""ChromaDB vector store implementation."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from common import get_logger, get_settings, EmbeddingProvider
from ingestion.processor.chunker import Chunk
from ingestion.embeddings import get_embedding_model, EmbeddingModel

log = get_logger(__name__)


class ChromaVectorStore:
    """ChromaDB vector store with configurable embeddings."""

    def __init__(
        self,
        persist_dir: Path | str = "./chroma_db",
        collection_name: str = "documents",
        embedding_provider: EmbeddingProvider | str | None = None,
        embedding_model: str | None = None,
        openai_api_key: str | None = None,
    ):
        """Initialize ChromaDB vector store.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the collection
            embedding_provider: Embedding provider (openai or huggingface)
            embedding_model: Embedding model name
            openai_api_key: OpenAI API key (only used if provider is openai)
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name

        # Initialize embedding model using the abstraction
        self.embedder: EmbeddingModel = get_embedding_model(
            provider=embedding_provider,
            model_name=embedding_model,
            api_key=openai_api_key,
        )

        # Initialize ChromaDB with persistent storage
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
            ),
        )

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        settings = get_settings()
        log.info(
            "vectorstore_initialized",
            persist_dir=str(self.persist_dir),
            collection=collection_name,
            embedding_provider=settings.embedding_provider.value,
            embedding_model=settings.embedding_model,
            existing_docs=self.collection.count(),
        )

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return self.embedder.get_embeddings(texts)

    def add_chunks(
        self,
        chunks: list[Chunk],
        document_id: str,
        document_name: str,
    ) -> list[str]:
        """Add chunks to the vector store.

        Args:
            chunks: List of Chunk objects to add
            document_id: Unique identifier for the source document
            document_name: Human-readable document name

        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []

        log.info(
            "adding_chunks",
            count=len(chunks),
            document_id=document_id,
        )

        # Prepare data for ChromaDB
        texts = [chunk.contextualized_text for chunk in chunks]
        ids = [f"{document_id}_chunk_{chunk.chunk_index}" for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "document_name": document_name,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "has_context": bool(chunk.context),
                "original_text": chunk.text[:1000],  # Store first 1000 chars
            }
            for chunk in chunks
        ]

        # Get embeddings
        embeddings = self._get_embeddings(texts)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        log.info(
            "chunks_added",
            count=len(ids),
            document_id=document_id,
        )

        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector store.

        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_document_id: Optional document ID filter

        Returns:
            List of result dictionaries with text, metadata, and score
        """
        log.debug("querying", query=query_text[:100], n_results=n_results)

        # Get query embedding
        query_embedding = self._get_embeddings([query_text])[0]

        # Build where filter
        where_filter = None
        if filter_document_id:
            where_filter = {"document_id": filter_document_id}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append(
                    {
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None,
                        "id": results["ids"][0][i] if results["ids"] else None,
                    }
                )

        log.debug("query_results", count=len(formatted))
        return formatted

    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of chunks deleted
        """
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            log.info("document_deleted", document_id=document_id, chunks=len(results["ids"]))
            return len(results["ids"])

        return 0

    def list_documents(self) -> list[dict[str, Any]]:
        """List all ingested documents.

        Returns:
            List of document info dictionaries
        """
        # Get all metadata
        results = self.collection.get(include=["metadatas"])

        # Aggregate by document
        documents: dict[str, dict] = {}
        for metadata in results["metadatas"] or []:
            doc_id = metadata.get("document_id", "unknown")
            if doc_id not in documents:
                documents[doc_id] = {
                    "document_id": doc_id,
                    "document_name": metadata.get("document_name", "Unknown"),
                    "chunk_count": 0,
                    "pages": set(),
                }
            documents[doc_id]["chunk_count"] += 1
            documents[doc_id]["pages"].add(metadata.get("page_number", 0))

        # Convert to list
        result = []
        for doc in documents.values():
            doc["pages"] = sorted(doc["pages"])
            doc["page_count"] = len(doc["pages"])
            result.append(doc)

        return result

    def count(self) -> int:
        """Get total number of chunks in the collection."""
        return self.collection.count()
