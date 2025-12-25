"""DeepSeek OCR provider using custom FastAPI server."""

import io
import re
from typing import Iterator

import httpx
from PIL import Image

from common import get_logger
from ingestion.ocr.base import OCRProvider, OCRResult

log = get_logger(__name__)


class DeepSeekOCR(OCRProvider):
    """DeepSeek OCR using custom FastAPI server endpoints.

    Connects to the DeepSeek-OCR server running via docker-compose,
    which exposes /ocr/image and /ocr/pdf endpoints.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 3600.0,
    ):
        """Initialize DeepSeek OCR client.

        Args:
            base_url: OCR server URL (without /v1 suffix)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)
        log.info("deepseek_ocr_initialized", base_url=self.base_url)

    def extract_text(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """Extract text from a single image using DeepSeek OCR.

        Args:
            image: PIL Image to process
            page_number: Page number for metadata

        Returns:
            OCRResult with extracted text
        """
        log.info("ocr_processing_page", page=page_number)

        # Convert image to bytes
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        buffer.seek(0)

        # Call /ocr/image endpoint with multipart form data
        try:
            response = self.client.post(
                f"{self.base_url}/ocr/image",
                files={"file": ("page.png", buffer, "image/png")},
            )
            response.raise_for_status()

            result = response.json()

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown OCR error")
                log.error("ocr_failed", page=page_number, error=error_msg)
                raise RuntimeError(f"OCR failed: {error_msg}")

            text = result.get("result", "")

        except httpx.HTTPStatusError as e:
            log.error("ocr_http_error", page=page_number, status=e.response.status_code)
            raise RuntimeError(f"OCR server error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            log.error("ocr_request_error", page=page_number, error=str(e))
            raise RuntimeError(f"OCR request failed: {e}") from e

        # Clean up special tokens if present
        text = self._clean_ocr_output(text)

        log.info(
            "ocr_page_complete",
            page=page_number,
            chars=len(text),
        )

        return OCRResult(text=text, page_number=page_number)

    def _clean_ocr_output(self, text: str) -> str:
        """Clean special tokens from OCR output.

        DeepSeek OCR outputs markdown-formatted text. This method cleans up
        special tokens while preserving the markdown structure and reference
        tags for downstream image extraction.

        Note: Whitespace normalization is handled by TextCleaner to avoid
        duplicate processing.
        """
        # Remove DeepSeek special tokens (including full-width unicode variants)
        special_tokens = [
            "<|ocr_start|>",
            "<|ocr_end|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|end▁of▁sentence|>",
            # Full-width unicode variants (Japanese-style)
            "<｜end▁of▁sentence｜>",
            "<｜ocr_start｜>",
            "<｜ocr_end｜>",
            "<｜im_start｜>",
            "<｜im_end｜>",
            "<｜endoftext｜>",
        ]
        for token in special_tokens:
            text = text.replace(token, "")

        # Remove grounding and quad tags (but preserve ref/det for image extraction)
        text = re.sub(r"<\|grounding\|>", "", text)
        text = re.sub(r"<\|quad_start\|>.*?<\|quad_end\|>", "", text, flags=re.DOTALL)

        # Clean up any remaining special token patterns like <|...|>
        # BUT preserve reference tags for image extraction: <|ref|>...<|/ref|><|det|>...<|/det|>
        # We'll clean these after image extraction in the pipeline
        text = re.sub(r"<\|(?!ref\|)(?!/ref\|)(?!det\|)(?!/det\|)[^|]+\|>", "", text)

        # Also handle full-width bars ｜
        text = re.sub(r"<｜(?!ref｜)(?!/ref｜)(?!det｜)(?!/det｜)[^｜]+｜>", "", text)

        # NOTE: Whitespace normalization removed - handled by TextCleaner centrally
        # to avoid duplicate processing

        return text.strip()

    def extract_text_batch(
        self, images: list[Image.Image], start_page: int = 1
    ) -> list[OCRResult]:
        """Extract text from multiple images sequentially.

        Args:
            images: List of PIL Images to process
            start_page: Starting page number for metadata

        Returns:
            List of OCRResults
        """
        results = []
        for i, image in enumerate(images):
            page_number = start_page + i
            result = self.extract_text(image, page_number)
            results.append(result)
        return results

    def extract_text_stream(
        self, images: Iterator[Image.Image], start_page: int = 1
    ) -> Iterator[OCRResult]:
        """Extract text from images as a generator.

        Args:
            images: Iterator of PIL Images
            start_page: Starting page number

        Yields:
            OCRResult for each processed image
        """
        for i, image in enumerate(images):
            page_number = start_page + i
            yield self.extract_text(image, page_number)

    def __del__(self):
        """Close the HTTP client on cleanup."""
        if hasattr(self, "client"):
            self.client.close()
