"""Abstract base class for OCR providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass
class OCRResult:
    """Result from OCR processing."""

    text: str
    page_number: int
    confidence: float | None = None


class OCRProvider(ABC):
    """Abstract base class for OCR providers."""

    @abstractmethod
    def extract_text(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """Extract text from an image.

        Args:
            image: PIL Image to process
            page_number: Page number for metadata

        Returns:
            OCRResult with extracted text
        """
        pass

    @abstractmethod
    def extract_text_batch(
        self, images: list[Image.Image], start_page: int = 1
    ) -> list[OCRResult]:
        """Extract text from multiple images.

        Args:
            images: List of PIL Images to process
            start_page: Starting page number for metadata

        Returns:
            List of OCRResults
        """
        pass
