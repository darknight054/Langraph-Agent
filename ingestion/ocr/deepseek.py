"""DeepSeek OCR provider using vLLM OpenAI-compatible API."""

import base64
import io
from typing import Iterator

from openai import OpenAI
from PIL import Image

from common import get_logger
from ingestion.ocr.base import OCRProvider, OCRResult

log = get_logger(__name__)


class DeepSeekOCR(OCRProvider):
    """DeepSeek OCR using vLLM OpenAI-compatible API.

    Based on: https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-OCR.html
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "deepseek-ai/DeepSeek-OCR",
    ):
        """Initialize DeepSeek OCR client.

        Args:
            base_url: vLLM server URL
            api_key: API key (usually "EMPTY" for local vLLM)
            model: Model name
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=3600)
        self.model = model
        log.info("deepseek_ocr_initialized", base_url=base_url, model=model)

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URL."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        base64_data = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"

    def extract_text(self, image: Image.Image, page_number: int = 1) -> OCRResult:
        """Extract text from a single image using DeepSeek OCR.

        Args:
            image: PIL Image to process
            page_number: Page number for metadata

        Returns:
            OCRResult with extracted text
        """
        log.info("ocr_processing_page", page=page_number)

        # Convert image to base64
        image_url = self._image_to_base64(image.convert("RGB"))

        # Create request following vLLM DeepSeek-OCR recipe
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "<|grounding|>Convert the document to markdown."},
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=8192,
            temperature=0.0,  # Deterministic output for consistent OCR
            extra_body={
                "skip_special_tokens": False,
                "vllm_xargs": {
                    "ngram_size": 30,
                    "window_size": 90,
                    "whitelist_token_ids": [128821, 128822],
                },
            },
        )

        text = response.choices[0].message.content or ""

        # Clean up special tokens if present
        text = self._clean_ocr_output(text)

        log.info(
            "ocr_page_complete",
            page=page_number,
            chars=len(text),
            tokens=response.usage.total_tokens if response.usage else None,
        )

        return OCRResult(text=text, page_number=page_number)

    def _clean_ocr_output(self, text: str) -> str:
        """Clean special tokens and normalize OCR output.

        DeepSeek OCR outputs markdown-formatted text, which is great for
        structured documents. This method cleans up any special tokens
        while preserving the markdown structure for downstream processing.

        For handwritten notes: The model generally preserves the flow of
        handwritten content, converting it to readable text with appropriate
        line breaks and paragraph structure.

        Note: Reference tags with bounding boxes are preserved for downstream
        image extraction. They will be processed by ImageExtractor.
        """
        import re

        # Remove DeepSeek special tokens (including full-width unicode variants)
        special_tokens = [
            "<|ocr_start|>",
            "<|ocr_end|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            # Full-width unicode variants (Japanese-style)
            "<|end▁of▁sentence|>",
            "<|ocr_start|>",
            "<|ocr_end|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        for token in special_tokens:
            text = text.replace(token, "")

        # Remove grounding and quad tags (but preserve ref/box for image extraction)
        text = re.sub(r"<\|grounding\|>", "", text)
        text = re.sub(r"<\|quad_start\|>.*?<\|quad_end\|>", "", text, flags=re.DOTALL)

        # Clean up any remaining special token patterns like <|...|>
        # BUT preserve reference tags for image extraction: <|ref|>...<|/ref|><|box_start|>...<|box_end|>
        # We'll clean these after image extraction in the pipeline
        text = re.sub(r"<\|(?!ref\|)(?!/ref\|)(?!box_start\|)(?!box_end\|)[^|]+\|>", "", text)

        # Also handle full-width bars |
        text = re.sub(r"<|(?!ref|)(?!/ref|)(?!box_start|)(?!box_end|)[^|]+|>", "", text)

        # Normalize excessive whitespace while preserving paragraph breaks
        text = re.sub(r"\n{4,}", "\n\n\n", text)  # Max 3 newlines
        text = re.sub(r"[ \t]+", " ", text)  # Collapse horizontal whitespace

        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

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
