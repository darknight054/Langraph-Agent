"""Document ingestion pipeline with OCR and vector storage."""

from ingestion.pipeline import IngestionPipeline
from ingestion.embeddings import (
    get_embedding_model,
    get_shared_embedding_model,
    OpenAIEmbeddings,
    HuggingFaceEmbeddings,
)

__all__ = [
    "IngestionPipeline",
    "get_embedding_model",
    "get_shared_embedding_model",
    "OpenAIEmbeddings",
    "HuggingFaceEmbeddings",
]
