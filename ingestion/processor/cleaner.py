"""Text cleaning and normalization.

Enhanced cleaning pipeline for OCR-extracted text including:
- Encoding fixes for UTF-8 artifacts
- Unicode normalization (NFKC)
- Full-width character conversion
- LaTeX symbol conversion for mathematical notation
- OCR artifact removal
"""

import re
import unicodedata

from unidecode import unidecode

from common import get_logger

log = get_logger(__name__)


class TextCleaner:
    """Clean and normalize OCR-extracted text."""

    # LaTeX symbol replacements for mathematical notation
    LATEX_REPLACEMENTS = {
        r"\\coloneqq": ":=",
        r"\\eqqcolon": "=:",
        r"\\leq": "<=",
        r"\\geq": ">=",
        r"\\neq": "!=",
        r"\\approx": "~",
        r"\\times": "x",
        r"\\div": "/",
        r"\\pm": "+/-",
        r"\\mp": "-/+",
        r"\\cdot": ".",
        r"\\ldots": "...",
        r"\\infty": "infinity",
        r"\\alpha": "alpha",
        r"\\beta": "beta",
        r"\\gamma": "gamma",
        r"\\delta": "delta",
        r"\\epsilon": "epsilon",
        r"\\theta": "theta",
        r"\\lambda": "lambda",
        r"\\mu": "mu",
        r"\\pi": "pi",
        r"\\sigma": "sigma",
        r"\\omega": "omega",
        r"\\sum": "sum",
        r"\\prod": "prod",
        r"\\int": "integral",
        r"\\partial": "d",
        r"\\nabla": "nabla",
        r"\\forall": "for all",
        r"\\exists": "exists",
        r"\\in": "in",
        r"\\notin": "not in",
        r"\\subset": "subset",
        r"\\supset": "superset",
        r"\\cup": "union",
        r"\\cap": "intersection",
        r"\\emptyset": "empty set",
        r"\\rightarrow": "->",
        r"\\leftarrow": "<-",
        r"\\leftrightarrow": "<->",
        r"\\Rightarrow": "=>",
        r"\\Leftarrow": "<=",
        r"\\Leftrightarrow": "<=>",
    }

    # Full-width to ASCII character mapping
    FULLWIDTH_MAP = str.maketrans(
        'ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ'
        '０１２３４５６７８９'
        '！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠'
        '［＼］＾＿｀｛|｝～　',
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        '!"#$%&\'()*+,-./:;<=>?@'
        '[\\]^_`{|}~ '
    )

    def __init__(
        self,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        fix_encoding: bool = True,
        convert_latex: bool = True,
        normalize_fullwidth: bool = True,
        normalize_unicode: bool = True,
    ):
        """Initialize text cleaner.

        Args:
            remove_headers_footers: Attempt to remove page headers/footers
            normalize_whitespace: Normalize whitespace characters
            fix_encoding: Fix common encoding issues
            convert_latex: Convert LaTeX symbols to plain text
            normalize_fullwidth: Convert full-width characters to ASCII
            normalize_unicode: Convert Unicode characters to ASCII (en-dash, smart quotes, etc.)
        """
        self.remove_headers_footers = remove_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.fix_encoding = fix_encoding
        self.convert_latex = convert_latex
        self.normalize_fullwidth = normalize_fullwidth
        self.normalize_unicode = normalize_unicode

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

        # Remove OCR-specific reference tags first
        text = self._remove_ocr_tags(text)

        # Fix encoding issues
        if self.fix_encoding:
            text = self._fix_encoding(text)

        # Normalize unicode to ASCII (en-dash, smart quotes, etc.)
        if self.normalize_unicode:
            text = self._normalize_unicode_to_ascii(text)

        # Normalize unicode forms
        text = unicodedata.normalize("NFKC", text)

        # Convert full-width characters to ASCII
        if self.normalize_fullwidth:
            text = self._normalize_fullwidth(text)

        # Convert LaTeX symbols
        if self.convert_latex:
            text = self._convert_latex_symbols(text)

        # Remove control characters (except newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize whitespace
        if self.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Remove common OCR artifacts
        text = self._remove_artifacts(text)

        # Remove URLs and website-related patterns
        text = self._remove_urls_and_weblinks(text)

        # Convert HTML tables to plain text
        text = self._convert_html_tables(text)

        # Clean remaining HTML tags
        text = self._clean_html_tags(text)

        # Remove repeated/garbage lines
        text = self._remove_repeated_lines(text)

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

    def _remove_urls_and_weblinks(self, text: str) -> str:
        """Remove URLs and website-related text patterns.

        Removes:
        - HTTP/HTTPS URLs
        - Domain patterns (www.example.com, example.com)
        - "Website - " or "Website:" prefixes
        - "Webjournal:" patterns
        """
        # Remove HTTP/HTTPS URLs
        text = re.sub(r'https?://[^\s\)\]]+', '', text)

        # Remove standalone domain patterns
        text = re.sub(r'(?:www\.)?[a-zA-Z0-9-]+\.(com|org|net|edu|io)\b/?', '', text, flags=re.IGNORECASE)

        # Remove "Website - " or "Website:" prefixes (with or without trailing content on same line)
        text = re.sub(r'Website\s*[-:]\s*\S*', '', text, flags=re.IGNORECASE)

        # Remove "Webjournal:" patterns
        text = re.sub(r'Webjournal\s*[:]\s*\S*', '', text, flags=re.IGNORECASE)

        return text

    def _convert_html_tables(self, text: str) -> str:
        """Convert HTML tables to plain text format.

        DeepSeek OCR outputs tables with HTML tags like:
        <table><tr><td>Cell1</td><td>Cell2</td></tr></table>

        This converts them to a readable plain text format.
        """
        # Check if there are any table tags
        if '<table>' not in text.lower():
            return text

        def table_to_text(match):
            """Convert a single table match to plain text."""
            table_html = match.group(0)

            # Extract rows
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, flags=re.DOTALL | re.IGNORECASE)

            result_rows = []
            for row in rows:
                # Extract cells (td or th)
                cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.DOTALL | re.IGNORECASE)

                # Clean cell content
                cleaned_cells = []
                for cell in cells:
                    # Remove nested HTML tags, convert <br> to space
                    cell = re.sub(r'<br\s*/?>', ' ', cell, flags=re.IGNORECASE)
                    cell = re.sub(r'<[^>]+>', '', cell)
                    cell = cell.strip()
                    cleaned_cells.append(cell)

                if cleaned_cells:
                    result_rows.append(' | '.join(cleaned_cells))

            return '\n'.join(result_rows)

        # Replace tables with plain text representation
        text = re.sub(
            r'<table[^>]*>.*?</table>',
            table_to_text,
            text,
            flags=re.DOTALL | re.IGNORECASE
        )

        return text

    def _clean_html_tags(self, text: str) -> str:
        """Clean remaining HTML tags from OCR output.

        Converts:
        - <br> and <br/> to newlines
        - Removes other HTML tags but keeps content
        """
        # Convert <br> tags to newlines
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)

        # Remove any remaining HTML tags but keep their content
        text = re.sub(r'<[^>]+>', '', text)

        # Clean up HTML entities
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
        }
        for entity, char in html_entities.items():
            text = text.replace(entity, char)

        return text

    def _remove_repeated_lines(self, text: str, max_repeats: int = 2) -> str:
        """Remove excessively repeated lines (OCR garbage).

        Detects patterns like "System -" repeated 100+ times
        and reduces them to max_repeats occurrences.

        Handles cases where empty lines appear between repeated content.

        Args:
            text: Input text
            max_repeats: Maximum times a line can repeat (default 2)

        Returns:
            Text with repeated lines reduced
        """
        lines = text.split('\n')
        if len(lines) < 2:
            return text

        result = []
        prev_non_empty = None
        repeat_count = 0
        pending_empty_lines = []

        for line in lines:
            stripped = line.strip()

            # Handle empty lines - buffer them
            if not stripped:
                pending_empty_lines.append(line)
                continue

            # Check if this non-empty line matches previous non-empty line
            if stripped == prev_non_empty:
                repeat_count += 1
                if repeat_count < max_repeats:
                    # Add buffered empty lines and this line
                    result.extend(pending_empty_lines)
                    result.append(line)
                # Skip if we've exceeded max repeats (don't add empty lines either)
                pending_empty_lines = []
            else:
                # Different line - add buffered empty lines and this line
                result.extend(pending_empty_lines)
                result.append(line)
                pending_empty_lines = []
                prev_non_empty = stripped
                repeat_count = 1  # This is the first occurrence

        # Add any remaining empty lines at the end
        result.extend(pending_empty_lines)

        return '\n'.join(result)

    def _remove_ocr_tags(self, text: str) -> str:
        """Remove DeepSeek OCR reference and detection tags.

        DeepSeek OCR outputs reference markers like:
        <|ref|>text<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>

        These need to be stripped for clean text output.
        """
        # Remove complete reference+det tag pairs
        text = re.sub(r'<\|ref\|>[^<]*<\|/ref\|><\|det\|>.*?<\|/det\|>', '', text, flags=re.DOTALL)

        # Remove incomplete or malformed reference tags
        text = re.sub(r'<\|ref\|>[^<]*<\|/ref\|>', '', text)
        text = re.sub(r'<\|det\|>.*?<\|/det\|>', '', text, flags=re.DOTALL)

        # Remove orphaned tags
        text = re.sub(r'<\|ref\|>', '', text)
        text = re.sub(r'<\|/ref\|>', '', text)
        text = re.sub(r'<\|det\|>', '', text)
        text = re.sub(r'<\|/det\|>', '', text)

        return text

    def _normalize_unicode_to_ascii(self, text: str) -> str:
        """Normalize Unicode characters to ASCII equivalents.

        Converts characters like:
        - – (en-dash) → -
        - — (em-dash) → --
        - " " (smart quotes) → "
        - ' ' (smart apostrophes) → '
        - é, ñ, etc. → e, n, etc.
        """
        return unidecode(text)

    def _normalize_fullwidth(self, text: str) -> str:
        """Convert full-width characters to ASCII equivalents.

        Full-width characters are common in CJK text and OCR output.
        This normalizes them to standard ASCII for consistency.
        """
        return text.translate(self.FULLWIDTH_MAP)

    def _convert_latex_symbols(self, text: str) -> str:
        """Convert LaTeX mathematical notation to plain text.

        This helps with documents containing mathematical formulas
        that OCR might output with LaTeX notation.
        """
        for latex, replacement in self.LATEX_REPLACEMENTS.items():
            text = text.replace(latex, replacement)
        return text

    def clean_batch(self, texts: list[str]) -> list[str]:
        """Clean multiple texts.

        Args:
            texts: List of raw texts

        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]
