"""Text chunking with optional contextual enhancement.

Supports two strategies:
- SEMANTIC: Base chunking with RecursiveCharacterTextSplitter (fast, good baseline)
- CONTEXTUAL: Semantic + LLM-generated context per chunk (slower, 49% better retrieval)

Reference: https://www.anthropic.com/news/contextual-retrieval
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from common import get_logger

log = get_logger(__name__)


class ChunkingStrategy(str, Enum):
    """Chunking strategy options."""

    SEMANTIC = "semantic"
    CONTEXTUAL = "contextual"


@dataclass
class Chunk:
    """A text chunk with metadata."""

    text: str
    page_number: int
    chunk_index: int
    context: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def contextualized_text(self) -> str:
        """Get text with context prepended (for embedding)."""
        if self.context:
            return f"{self.context}\n\n{self.text}"
        return self.text


class TextChunker:
    """Text chunker with configurable strategy.

    Strategies:
    - SEMANTIC: Uses RecursiveCharacterTextSplitter for intelligent splitting
      at sentence/paragraph boundaries. Fast and good baseline.

    - CONTEXTUAL: Same as semantic, but adds LLM-generated context to each
      chunk before embedding. Based on Anthropic's research showing 49%
      improvement in retrieval accuracy.

    For handwritten notes: Both strategies work well. The semantic strategy
    preserves the original structure, while contextual adds understanding
    of how each piece fits the whole document.
    """

    CONTEXT_PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: ChunkingStrategy | Literal["semantic", "contextual"] = ChunkingStrategy.SEMANTIC,
        context_model: str = "gpt-4o-mini",
        openai_api_key: str | None = None,
    ):
        """Initialize text chunker.

        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            strategy: "semantic" (fast) or "contextual" (better retrieval)
            context_model: Model for generating context (only used if contextual)
            openai_api_key: OpenAI API key (uses env if not provided)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Normalize strategy to enum
        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy)
        self.strategy = strategy

        self.context_model = context_model

        # Initialize tokenizer for accurate token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize OpenAI client only if contextual strategy
        if self.strategy == ChunkingStrategy.CONTEXTUAL:
            self.client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
        else:
            self.client = None

        # Use langchain's splitter with separators optimized for documents
        # Tries to split at natural boundaries: paragraphs > sentences > words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,  # ~4 chars per token approximation
            chunk_overlap=chunk_overlap * 4,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n",    # Line breaks
                ". ",    # Sentence ends
                "? ",    # Questions
                "! ",    # Exclamations
                "; ",    # Semicolons
                ", ",    # Commas
                " ",     # Words
                "",      # Characters (last resort)
            ],
            length_function=self._token_count,
        )

        log.info(
            "chunker_initialized",
            strategy=self.strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _token_count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def chunk_document(
        self,
        document: str,
        page_texts: list[tuple[int, str]] | None = None,
    ) -> list[Chunk]:
        """Chunk a document using the configured strategy.

        Args:
            document: Full document text
            page_texts: Optional list of (page_number, text) for page-aware chunking

        Returns:
            List of Chunk objects
        """
        if not document.strip():
            return []

        # Step 1: Split into base chunks (semantic splitting)
        if page_texts:
            chunks = self._chunk_by_pages(page_texts)
        else:
            chunks = self._chunk_full_document(document)

        log.info("base_chunks_created", count=len(chunks), strategy=self.strategy.value)

        # Step 2: Add contextual information if using contextual strategy
        if self.strategy == ChunkingStrategy.CONTEXTUAL and self.client:
            chunks = self._add_context(document, chunks)

        return chunks

    def _chunk_full_document(self, document: str) -> list[Chunk]:
        """Chunk entire document without page awareness."""
        texts = self.splitter.split_text(document)

        return [
            Chunk(
                text=text,
                page_number=1,
                chunk_index=i,
            )
            for i, text in enumerate(texts)
        ]

    def _chunk_by_pages(self, page_texts: list[tuple[int, str]]) -> list[Chunk]:
        """Chunk document with page awareness."""
        chunks = []
        chunk_index = 0

        for page_num, page_text in page_texts:
            if not page_text.strip():
                continue

            page_chunks = self.splitter.split_text(page_text)

            for text in page_chunks:
                chunks.append(
                    Chunk(
                        text=text,
                        page_number=page_num,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

        return chunks

    def _add_context(self, document: str, chunks: list[Chunk]) -> list[Chunk]:
        """Add contextual information to each chunk using LLM."""
        log.info("adding_context", chunks=len(chunks))

        # Truncate document for context window (keep first ~8k tokens)
        doc_tokens = self.encoding.encode(document)
        if len(doc_tokens) > 8000:
            truncated_doc = self.encoding.decode(doc_tokens[:8000])
            truncated_doc += "\n\n[Document truncated...]"
        else:
            truncated_doc = document

        for i, chunk in enumerate(chunks):
            try:
                context = self._generate_context(truncated_doc, chunk.text)
                chunk.context = context
                log.debug("context_added", chunk=i, context_len=len(context))
            except Exception as e:
                log.warning("context_generation_failed", chunk=i, error=str(e))
                # Continue without context for this chunk

        return chunks

    def _generate_context(self, document: str, chunk_text: str) -> str:
        """Generate contextual description for a chunk."""
        if not self.client:
            return ""

        prompt = self.CONTEXT_PROMPT.format(
            document=document,
            chunk=chunk_text,
        )

        response = self.client.chat.completions.create(
            model=self.context_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )

        return response.choices[0].message.content or ""

    def chunk_texts(
        self,
        texts: list[str],
        page_numbers: list[int] | None = None,
    ) -> list[Chunk]:
        """Chunk multiple texts (typically pages).

        Args:
            texts: List of text strings
            page_numbers: Optional page numbers for each text

        Returns:
            List of Chunk objects
        """
        if page_numbers is None:
            page_numbers = list(range(1, len(texts) + 1))

        full_document = "\n\n".join(texts)
        page_texts = list(zip(page_numbers, texts))

        return self.chunk_document(full_document, page_texts)
