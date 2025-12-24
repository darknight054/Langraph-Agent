"""PDF to image conversion using PyMuPDF (fitz).

PyMuPDF provides high-quality PDF rendering with better performance
and cleaner rasterization compared to poppler-based alternatives.
"""

from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF
from PIL import Image

from common import get_logger

log = get_logger(__name__)


class PDFProcessor:
    """Convert PDF pages to images for OCR processing using PyMuPDF."""

    def __init__(self, dpi: int = 150):
        """Initialize PDF processor.

        Args:
            dpi: Resolution for PDF to image conversion.
                 150 DPI is optimal for OCR (balance of quality and speed).
                 Use 200-300 for higher quality if needed.
        """
        self.dpi = dpi
        # PyMuPDF uses 72 DPI as base, so we need a zoom factor
        self.zoom = dpi / 72.0

    def get_page_count(self, pdf_path: Path) -> int:
        """Get total number of pages in PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Number of pages
        """
        pdf_path = Path(pdf_path)
        with fitz.open(str(pdf_path)) as doc:
            return len(doc)

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

        images = []
        mat = fitz.Matrix(self.zoom, self.zoom)

        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            log.info("pdf_processing", path=str(pdf_path), total_pages=total_pages)

            # Handle page range (convert to 0-indexed for fitz)
            first = (start_page if start_page else 1) - 1
            last = (end_page if end_page else total_pages)

            # Validate range
            if first < 0:
                first = 0
            if last > total_pages:
                last = total_pages
            if first >= last:
                raise ValueError(f"Invalid page range: {first + 1}-{last}")

            log.info("extracting_pages", start=first + 1, end=last, count=last - first)

            for page_num in range(first, last):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)

        log.info("pages_extracted", count=len(images))
        return images

    def extract_pages_stream(
        self,
        pdf_path: Path,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> Iterator[tuple[int, Image.Image]]:
        """Extract pages from PDF as a generator to save memory.

        This is more memory-efficient for large PDFs as it yields
        one page at a time instead of loading all pages into memory.

        Args:
            pdf_path: Path to PDF file
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (1-indexed)

        Yields:
            Tuple of (page_number, PIL Image) where page_number is 1-indexed
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        mat = fitz.Matrix(self.zoom, self.zoom)

        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)

            # Handle page range (convert to 0-indexed for fitz)
            first = (start_page if start_page else 1) - 1
            last = (end_page if end_page else total_pages)

            if first < 0:
                first = 0
            if last > total_pages:
                last = total_pages

            log.info("streaming_pages", start=first + 1, end=last)

            for page_idx in range(first, last):
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Yield 1-indexed page number
                yield page_idx + 1, img

    def get_page_image(self, pdf_path: Path, page_number: int) -> Image.Image:
        """Get a single page as an image.

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)

        Returns:
            PIL Image of the page
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        mat = fitz.Matrix(self.zoom, self.zoom)

        with fitz.open(str(pdf_path)) as doc:
            if page_number < 1 or page_number > len(doc):
                raise ValueError(f"Page {page_number} out of range (1-{len(doc)})")

            page = doc[page_number - 1]  # Convert to 0-indexed
            pix = page.get_pixmap(matrix=mat)

            return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
