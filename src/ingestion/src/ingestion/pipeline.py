"""Main ingestion pipeline orchestrator."""

import hashlib
from pathlib import Path
from typing import Callable, Literal

from common import get_logger, get_settings, ChunkingStrategy
from ingestion.ocr import DeepSeekOCR
from ingestion.processor import PDFProcessor, TextCleaner, TextChunker
from ingestion.vectorstore import ChromaVectorStore

log = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion: PDF -> OCR -> Clean -> Chunk -> Store."""

    def __init__(
        self,
        ocr_url: str | None = None,
        chroma_persist_dir: Path | str | None = None,
        openai_api_key: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunking_strategy: ChunkingStrategy | Literal["semantic", "contextual"] | None = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            ocr_url: DeepSeek OCR vLLM server URL
            chroma_persist_dir: ChromaDB persistence directory
            openai_api_key: OpenAI API key
            chunk_size: Chunk size in tokens
            chunk_overlap: Chunk overlap in tokens
            chunking_strategy: "semantic" (fast) or "contextual" (better retrieval)
        """
        settings = get_settings()

        # Determine chunking strategy
        if chunking_strategy is not None:
            if isinstance(chunking_strategy, str):
                strategy = ChunkingStrategy(chunking_strategy)
            else:
                strategy = chunking_strategy
        else:
            strategy = settings.chunking_strategy

        # Initialize components
        self.pdf_processor = PDFProcessor()

        self.ocr = DeepSeekOCR(
            base_url=ocr_url or settings.deepseek_ocr_url,
        )

        self.cleaner = TextCleaner()

        self.chunker = TextChunker(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            strategy=strategy,
            openai_api_key=openai_api_key or settings.openai_api_key,
        )

        self.vectorstore = ChromaVectorStore(
            persist_dir=chroma_persist_dir or settings.chroma_persist_dir,
            openai_api_key=openai_api_key or settings.openai_api_key,
        )

        log.info("pipeline_initialized", chunking_strategy=strategy.value)

    def _generate_document_id(self, pdf_path: Path) -> str:
        """Generate a unique ID for a document based on path and modification time."""
        stat = pdf_path.stat()
        content = f"{pdf_path.name}_{stat.st_size}_{stat.st_mtime}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def ingest(
        self,
        pdf_path: Path | str,
        start_page: int | None = None,
        end_page: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict:
        """Ingest a PDF document.

        Args:
            pdf_path: Path to PDF file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (1-indexed)
            progress_callback: Optional callback(stage, current, total)

        Returns:
            Dictionary with ingestion results
        """
        pdf_path = Path(pdf_path)
        document_id = self._generate_document_id(pdf_path)
        document_name = pdf_path.name

        log.info(
            "ingestion_started",
            document=document_name,
            document_id=document_id,
            start_page=start_page,
            end_page=end_page,
        )

        def report_progress(stage: str, current: int, total: int):
            if progress_callback:
                progress_callback(stage, current, total)
            log.info("progress", stage=stage, current=current, total=total)

        try:
            # Step 1: Extract pages as images
            report_progress("extracting_pages", 0, 1)
            images = self.pdf_processor.extract_pages(
                pdf_path,
                start_page=start_page,
                end_page=end_page,
            )
            total_pages = len(images)
            report_progress("extracting_pages", 1, 1)

            # Calculate actual page numbers
            actual_start = start_page if start_page else 1

            # Step 2: OCR each page
            page_texts: list[tuple[int, str]] = []
            for i, image in enumerate(images):
                report_progress("ocr", i, total_pages)
                page_num = actual_start + i
                result = self.ocr.extract_text(image, page_number=page_num)

                # Clean the OCR text
                cleaned_text = self.cleaner.clean(result.text)
                page_texts.append((page_num, cleaned_text))

            report_progress("ocr", total_pages, total_pages)

            # Step 3: Chunk the document
            report_progress("chunking", 0, 1)
            chunks = self.chunker.chunk_texts(
                texts=[text for _, text in page_texts],
                page_numbers=[page_num for page_num, _ in page_texts],
            )
            report_progress("chunking", 1, 1)

            # Step 4: Store in vector database
            report_progress("storing", 0, 1)
            chunk_ids = self.vectorstore.add_chunks(
                chunks=chunks,
                document_id=document_id,
                document_name=document_name,
            )
            report_progress("storing", 1, 1)

            result = {
                "success": True,
                "document_id": document_id,
                "document_name": document_name,
                "pages_processed": total_pages,
                "chunks_created": len(chunks),
                "chunk_ids": chunk_ids,
            }

            log.info(
                "ingestion_complete",
                **{k: v for k, v in result.items() if k != "chunk_ids"},
            )

            return result

        except Exception as e:
            log.error("ingestion_failed", error=str(e), document=document_name)
            return {
                "success": False,
                "document_name": document_name,
                "error": str(e),
            }

    def list_documents(self) -> list[dict]:
        """List all ingested documents."""
        return self.vectorstore.list_documents()

    def delete_document(self, document_id: str) -> int:
        """Delete a document from the vector store."""
        return self.vectorstore.delete_document(document_id)

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        """Query the vector store.

        Args:
            query_text: Query text
            n_results: Number of results
            document_id: Optional document filter

        Returns:
            List of matching chunks
        """
        return self.vectorstore.query(
            query_text=query_text,
            n_results=n_results,
            filter_document_id=document_id,
        )
