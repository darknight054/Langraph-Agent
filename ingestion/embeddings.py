"""Embedding providers for vector store operations.

Supports both OpenAI and HuggingFace embedding models with a unified interface.
"""

from abc import ABC, abstractmethod
from typing import Protocol

from common import get_logger, get_settings, EmbeddingProvider

log = get_logger(__name__)


class EmbeddingModel(Protocol):
    """Protocol for embedding models."""

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts."""
        ...


class OpenAIEmbeddings:
    """OpenAI embedding model implementation."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None):
        """Initialize OpenAI embeddings.

        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (uses env if not provided)
        """
        from openai import OpenAI

        self.model = model_name
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        log.info("openai_embeddings_initialized", model=model_name)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts using OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [item.embedding for item in response.data]


class HuggingFaceEmbeddings:
    """HuggingFace embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        """Initialize HuggingFace embeddings.

        Args:
            model_name: HuggingFace model ID or local path
        """
        from sentence_transformers import SentenceTransformer

        log.info("loading_huggingface_model", model=model_name)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.model_name = model_name
        log.info("huggingface_embeddings_initialized", model=model_name)

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts using sentence-transformers.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (normalized)
        """
        if not texts:
            return []

        # Encode with normalization for cosine similarity
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()


def get_embedding_model(
    provider: EmbeddingProvider | str | None = None,
    model_name: str | None = None,
    api_key: str | None = None,
) -> EmbeddingModel:
    """Factory function to get the appropriate embedding model.

    Args:
        provider: Embedding provider (openai or huggingface). Uses config if not provided.
        model_name: Model name. Uses config if not provided.
        api_key: API key for OpenAI. Uses config/env if not provided.

    Returns:
        Configured embedding model instance
    """
    settings = get_settings()

    # Determine provider
    if provider is None:
        provider = settings.embedding_provider
    elif isinstance(provider, str):
        provider = EmbeddingProvider(provider)

    # Determine model name
    if model_name is None:
        model_name = settings.embedding_model

    # Create appropriate embedding model
    if provider == EmbeddingProvider.OPENAI:
        return OpenAIEmbeddings(
            model_name=model_name,
            api_key=api_key or settings.openai_api_key,
        )
    elif provider == EmbeddingProvider.HUGGINGFACE:
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")


# Global embedding model instance (lazy initialized)
_embedding_model: EmbeddingModel | None = None


def get_shared_embedding_model() -> EmbeddingModel:
    """Get or create a shared embedding model instance.

    This is useful for sharing the model across the application
    to avoid loading it multiple times.

    Returns:
        Shared embedding model instance
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model
