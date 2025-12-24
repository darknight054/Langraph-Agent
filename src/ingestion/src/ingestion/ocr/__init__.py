"""OCR providers for document text extraction."""

from ingestion.ocr.base import OCRProvider
from ingestion.ocr.deepseek import DeepSeekOCR

__all__ = ["OCRProvider", "DeepSeekOCR"]
