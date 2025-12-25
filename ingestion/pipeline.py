"""Main ingestion pipeline orchestrator.

Enhanced pipeline with:
- Image extraction from OCR output
- Memory-efficient streaming for large PDFs
- Configurable DPI and processing options
"""

import hashlib
from pathlib import Path
from typing import Callable, Literal

from common import get_logger, get_settings, ChunkingStrategy
from ingestion.ocr import DeepSeekOCR
from ingestion.processor import PDFProcessor, TextCleaner, TextChunker, ImageExtractor
from ingestion.vectorstore import ChromaVectorStore

log = get_logger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion: PDF -> OCR -> Extract Images -> Clean -> Chunk -> Store."""

    def __init__(
        self,
        ocr_url: str | None = None,
        chroma_persist_dir: Path | str | None = None,
        openai_api_key: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        chunking_strategy: ChunkingStrategy | Literal["semantic", "contextual"] | None = None,
        dpi: int | None = None,
        extract_images: bool = False,
        image_output_dir: Path | str | None = None,
    ):
        """Initialize ingestion pipeline.

        Args:
            ocr_url: DeepSeek OCR vLLM server URL
            chroma_persist_dir: ChromaDB persistence directory
            openai_api_key: OpenAI API key
            chunk_size: Chunk size in tokens
            chunk_overlap: Chunk overlap in tokens
            chunking_strategy: "semantic" (fast) or "contextual" (better retrieval)
            dpi: PDF rendering DPI (default: 150)
            extract_images: Whether to extract images from OCR output
            image_output_dir: Directory for extracted images
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

        # Store settings
        self.extract_images = extract_images
        self.image_output_dir = Path(image_output_dir) if image_output_dir else None

        # Initialize components
        self.pdf_processor = PDFProcessor(dpi=dpi or 150)

        self.ocr = DeepSeekOCR(
            base_url=ocr_url or settings.deepseek_ocr_url,
        )

        self.cleaner = TextCleaner()

        self.image_extractor = ImageExtractor(
            output_dir=self.image_output_dir
        ) if extract_images else None

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

        log.info(
            "pipeline_initialized",
            chunking_strategy=strategy.value,
            extract_images=extract_images,
        )

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
        use_streaming: bool = True,
    ) -> dict:
        """Ingest a PDF document.

        Args:
            pdf_path: Path to PDF file
            start_page: First page to process (1-indexed)
            end_page: Last page to process (1-indexed)
            progress_callback: Optional callback(stage, current, total)
            use_streaming: Use memory-efficient streaming for large PDFs

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
            streaming=use_streaming,
        )

        def report_progress(stage: str, current: int, total: int):
            if progress_callback:
                progress_callback(stage, current, total)
            log.info("progress", stage=stage, current=current, total=total)

        try:
            # Get total page count for progress reporting
            total_pages = self.pdf_processor.get_page_count(pdf_path)
            actual_start = start_page if start_page else 1
            actual_end = end_page if end_page else total_pages
            pages_to_process = actual_end - actual_start + 1

            # Set up image output directory if extracting images
            if self.extract_images and self.image_extractor:
                if self.image_output_dir:
                    doc_image_dir = self.image_output_dir / Path(document_name).stem
                else:
                    doc_image_dir = pdf_path.parent / "images" / Path(document_name).stem
                doc_image_dir.mkdir(parents=True, exist_ok=True)
                self.image_extractor.output_dir = doc_image_dir

            page_texts: list[tuple[int, str]] = []
            total_images_extracted = 0

            if use_streaming:
                # Memory-efficient streaming mode
                report_progress("processing", 0, pages_to_process)

                for page_num, image in self.pdf_processor.extract_pages_stream(
                    pdf_path,
                    start_page=start_page,
                    end_page=end_page,
                ):
                    current_page = page_num - actual_start
                    report_progress("processing", current_page, pages_to_process)

                    # OCR the page
                    result = self.ocr.extract_text(image, page_number=page_num)

                    # Extract images from OCR output if enabled
                    if self.extract_images and self.image_extractor:
                        processed_text, extracted_images = self.image_extractor.extract_images(
                            ocr_text=result.text,
                            page_image=image,
                            page_number=page_num,
                            document_name=document_name,
                            save_images=True,
                        )
                        total_images_extracted += len(extracted_images)
                    else:
                        processed_text = result.text

                    # Clean the text
                    cleaned_text = self.cleaner.clean(processed_text)

                    # Skip empty pages (garbage content filtered by cleaner)
                    if cleaned_text.strip():
                        page_texts.append((page_num, cleaned_text))

                report_progress("processing", pages_to_process, pages_to_process)

            else:
                # Non-streaming mode (loads all pages at once)
                report_progress("extracting_pages", 0, 1)
                images = self.pdf_processor.extract_pages(
                    pdf_path,
                    start_page=start_page,
                    end_page=end_page,
                )
                report_progress("extracting_pages", 1, 1)

                for i, image in enumerate(images):
                    report_progress("ocr", i, len(images))
                    page_num = actual_start + i

                    # OCR the page
                    result = self.ocr.extract_text(image, page_number=page_num)

                    # Extract images from OCR output if enabled
                    if self.extract_images and self.image_extractor:
                        processed_text, extracted_images = self.image_extractor.extract_images(
                            ocr_text=result.text,
                            page_image=image,
                            page_number=page_num,
                            document_name=document_name,
                            save_images=True,
                        )
                        total_images_extracted += len(extracted_images)
                    else:
                        processed_text = result.text

                    # Clean the text
                    cleaned_text = self.cleaner.clean(processed_text)

                    # Skip empty pages (garbage content filtered by cleaner)
                    if cleaned_text.strip():
                        page_texts.append((page_num, cleaned_text))

                report_progress("ocr", len(images), len(images))

            # Chunk the document
            report_progress("chunking", 0, 1)
            log.info("texts_ready_for_chunking", page_text=page_texts[0])
            chunks = self.chunker.chunk_texts(
                texts=[text for _, text in page_texts],
                page_numbers=[page_num for page_num, _ in page_texts],
            )
            report_progress("chunking", 1, 1)

            # Store in vector database
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
                "pages_processed": len(page_texts),
                "chunks_created": len(chunks),
                "images_extracted": total_images_extracted,
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
