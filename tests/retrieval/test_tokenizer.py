"""Tests for entity-aware tokenizer (M3.P3.1.F1)."""

import pytest
from src.retrieval.tokenizer import tokenize


def test_tokenize_standard_text():
    """Test standard tokenization without protected entities."""
    text = "Hello world! My battery is draining fast."
    tokens = tokenize(text)
    assert tokens == ["hello", "world", "my", "battery", "is", "draining", "fast"]


def test_tokenize_empty_and_whitespace():
    """Test edge cases with empty or whitespace-only strings."""
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!@#$%^&*()") == []


def test_tokenize_device_model_preserved():
    """Test that device model codes like 'SM-T280' remain a single atomic token."""
    text = "My SM-T280 tablet will not charge anymore."
    tokens = tokenize(text)
    assert "sm-t280" in tokens
    # Ensure it is NOT split into 'sm' and 't280'
    assert "sm" not in tokens
    assert "t280" not in tokens


def test_tokenize_os_version_preserved():
    """Test that OS versions like 'iOS 11.1' remain a single atomic token."""
    text = "After upgrading to iOS 11.1 my apps crash."
    tokens = tokenize(text)
    assert "ios 11.1" in tokens
    # Ensure it is not split on the space or period
    assert "ios" not in tokens
    assert "11.1" not in tokens


def test_tokenize_punctuation_boundary_adjacent():
    """Test entity codes immediately adjacent to punctuation."""
    text = "Check this model: SM-T280, running (iOS 11.1)."
    tokens = tokenize(text)
    assert "sm-t280" in tokens
    assert "ios 11.1" in tokens
    # Ensure trailing comma or wrapping parentheses are not part of entity
    assert "sm-t280," not in tokens
    assert "(ios 11.1)" not in tokens
    assert "(ios" not in tokens


def test_tokenize_multiple_entities_in_sentence():
    """Test multiple protected entities occurring in the same sentence."""
    text = "I have an iPhone X on iOS 14.2 and an SM-G950F."
    tokens = tokenize(text)
    assert "iphone x" in tokens
    assert "ios 14.2" in tokens
    assert "sm-g950f" in tokens


def test_tokenize_pii_and_alphanumeric_code_preserved():
    """Test that alphanumeric codes and protected PII placeholders are preserved."""
    text = "Error code 0x80070005 sent to __email__."
    tokens = tokenize(text)
    assert "0x80070005" in tokens
    assert "__email__" in tokens

