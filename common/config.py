"""Configuration management using Pydantic Settings."""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingStrategy(str, Enum):
    """Chunking strategy options."""

    SEMANTIC = "semantic"  # Base chunking with RecursiveCharacterTextSplitter
    CONTEXTUAL = "contextual"  # Semantic + LLM-generated context per chunk


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = Field(..., description="OpenAI API key")

    # DeepSeek OCR (vLLM endpoint)
    deepseek_ocr_url: str = Field(
        default="http://localhost:8000/v1",
        description="DeepSeek OCR vLLM server URL",
    )
    ocr_dpi: int = Field(
        default=150,
        description="PDF rendering DPI for OCR (150 is optimal balance of quality/speed)",
    )
    ocr_prompt: str = Field(
        default="<|grounding|>Convert the document to markdown.",
        description="OCR prompt for DeepSeek model",
    )
    ocr_temperature: float = Field(
        default=0.0,
        description="OCR temperature (0.0 for deterministic output)",
    )

    # Vector Store
    chroma_persist_dir: Path = Field(
        default=Path("./chroma_db"),
        description="ChromaDB persistence directory",
    )

    # Langfuse (optional)
    langfuse_public_key: str | None = Field(
        default=None,
        description="Langfuse public key for tracing",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        description="Langfuse secret key for tracing",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server host",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: Literal["json", "console"] = Field(
        default="console",
        description="Log output format",
    )

    # Chunking
    chunk_size: int = Field(default=512, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, description="Chunk overlap in tokens")
    chunking_strategy: ChunkingStrategy = Field(
        default=ChunkingStrategy.CONTEXTUAL,
        description="Chunking strategy: 'semantic' (fast) or 'contextual' (better retrieval)",
    )

    # Image Extraction
    extract_images: bool = Field(
        default=True,
        description="Extract images from OCR output using reference tags",
    )
    image_output_dir: Path | None = Field(
        default=None,
        description="Directory for extracted images (None = use document directory)",
    )

    # RAG
    retrieval_top_k: int = Field(default=5, description="Number of chunks to retrieve")
    max_retry_count: int = Field(default=3, description="Max validation retries")

    @property
    def langfuse_enabled(self) -> bool:
        """Check if Langfuse tracing is configured."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
