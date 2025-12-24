"""Text cleaning and normalization."""

import re
import unicodedata

from common import get_logger

log = get_logger(__name__)


class TextCleaner:
    """Clean and normalize OCR-extracted text."""

    def __init__(
        self,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        fix_encoding: bool = True,
    ):
        """Initialize text cleaner.

        Args:
            remove_headers_footers: Attempt to remove page headers/footers
            normalize_whitespace: Normalize whitespace characters
            fix_encoding: Fix common encoding issues
        """
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.fix_encoding = fix_encoding

    def clean(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text from OCR

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        original_len = len(text)

        # Fix encoding issues
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Remove common OCR artifacts
        text = self._remove_artifacts(text)

        log.debug(
            "text_cleaned",
            original_len=original_len,
            cleaned_len=len(text),
        )

        return text.strip()

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        # Common replacements for encoding artifacts
        replacements = {
            "â€™": "'",
            "â€œ": '"',
            "â€\x9d": '"',
            "â€": "—",
            "â€¦": "...",
            "Ã©": "é",
            "Ã¨": "è",
            "Ã ": "à",
            "Ã¢": "â",
            "Ã®": "î",
            "Ã´": "ô",
            "Ã»": "û",
            "Ã§": "ç",
            "Ã¼": "ü",
            "Ã¶": "ö",
            "Ã¤": "ä",
            "ï¬\x81": "fi",
            "ï¬‚": "fl",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various whitespace with regular space
        text = re.sub(r"[\t\r\f\v]+", " ", text)

        # Collapse multiple spaces
        text = re.sub(r" +", " ", text)

        # Collapse multiple newlines (keep paragraph breaks)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing whitespace from lines
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r"\n +", "\n", text)

        return text

    def _remove_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts."""
        # Remove isolated special characters that are likely artifacts
        text = re.sub(r"(?<!\S)[|\\/_]{1,3}(?!\S)", "", text)

        # Remove repeated punctuation (more than 3)
        text = re.sub(r"([.!?]){4,}", r"\1\1\1", text)

        # Remove page number patterns at start/end of content
        # e.g., "Page 1", "- 1 -", "1 of 10"
        text = re.sub(
            r"^(?:Page\s*)?\d+(?:\s*of\s*\d+)?[\s\-]*\n", "", text, flags=re.I
        )
        text = re.sub(
            r"\n[\s\-]*(?:Page\s*)?\d+(?:\s*of\s*\d+)?$", "", text, flags=re.I
        )

        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean multiple texts.

        Args:
            texts: List of raw texts

        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]
