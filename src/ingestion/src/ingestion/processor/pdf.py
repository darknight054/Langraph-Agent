"""PDF to image conversion."""

from pathlib import Path
from typing import Iterator

from pdf2image import convert_from_path
from PIL import Image

from common import get_logger

log = get_logger(__name__)


class PDFProcessor:
    """Convert PDF pages to images for OCR processing."""

    def __init__(self, dpi: int = 200):
        """Initialize PDF processor.

        Args:
            dpi: Resolution for PDF to image conversion
        """
        self.dpi = dpi

    def get_page_count(self, pdf_path: Path) -> int:
        """Get total number of pages in PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        from pdf2image.pdf2image import pdfinfo_from_path

        info = pdfinfo_from_path(str(pdf_path))
        return info.get("Pages", 0)

    def extract_pages(
        self,
        pdf_path: Path,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> list[Image.Image]:
        """Extract pages from PDF as images.

        Args:
            pdf_path: Path to PDF file
            start_page: First page to extract (1-indexed, inclusive)
            end_page: Last page to extract (1-indexed, inclusive)

        Returns:
            List of PIL Images
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        total_pages = self.get_page_count(pdf_path)
        log.info("pdf_processing", path=str(pdf_path), total_pages=total_pages)

        # Handle page range
        first = start_page if start_page else 1
        last = end_page if end_page else total_pages

        # Validate range
        if first < 1:
            first = 1
        if last > total_pages:
            last = total_pages
        if first > last:
            raise ValueError(f"Invalid page range: {first}-{last}")

        log.info("extracting_pages", start=first, end=last, count=last - first + 1)

        images = convert_from_path(
            str(pdf_path),
            dpi=self.dpi,
            first_page=first,
            last_page=last,
        )

        log.info("pages_extracted", count=len(images))
        return images

    def extract_pages_stream(
        self,
        pdf_path: Path,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> Iterator[tuple[int, Image.Image]]:
        """Extract pages from PDF as a generator to save memory.

        Args:
            pdf_path: Path to PDF file
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (1-indexed)

        Yields:
            Tuple of (page_number, PIL Image)
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        total_pages = self.get_page_count(pdf_path)

        first = start_page if start_page else 1
        last = end_page if end_page else total_pages

        if first < 1:
            first = 1
        if last > total_pages:
            last = total_pages

        log.info("streaming_pages", start=first, end=last)

        for page_num in range(first, last + 1):
            images = convert_from_path(
                str(pdf_path),
                dpi=self.dpi,
                first_page=page_num,
                last_page=page_num,
            )
            if images:
                yield page_num, images[0]
