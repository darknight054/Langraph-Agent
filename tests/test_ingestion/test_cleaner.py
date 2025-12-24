"""Tests for text cleaner."""

import pytest


def test_text_cleaner_import():
    """Test that TextCleaner can be imported."""
    from ingestion.processor import TextCleaner

    cleaner = TextCleaner()
    assert cleaner is not None


def test_clean_whitespace():
    """Test whitespace normalization."""
    from ingestion.processor import TextCleaner

    cleaner = TextCleaner()

    text = "Hello   world\n\n\n\nNew paragraph"
    cleaned = cleaner.clean(text)

    assert "   " not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_empty_string():
    """Test cleaning empty string."""
    from ingestion.processor import TextCleaner

    cleaner = TextCleaner()
    assert cleaner.clean("") == ""
    assert cleaner.clean("   ") == ""
