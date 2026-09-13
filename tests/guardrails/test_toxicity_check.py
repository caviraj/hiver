"""Unit tests for toxicity scoring and validation."""

import pytest
import logging
from src.guardrails.toxicity_check import score_toxicity, check_toxicity


def test_benign_text_low_toxicity():
    """Customer service responses must have low toxicity scores."""
    text = "Thank you for reaching out! We are happy to help resolve your issue promptly."
    score = score_toxicity(text)
    assert 0.0 <= score < 0.3
    assert check_toxicity(text, threshold=0.7) is False


def test_overtly_toxic_text_high_toxicity():
    """Abusive, offensive, or hostile text must exceed the rejection threshold."""
    text = "You are a complete idiot and a moron. Go to hell and die."
    score = score_toxicity(text)
    assert score >= 0.7
    assert check_toxicity(text, threshold=0.7) is True


def test_custom_threshold():
    """Verify custom threshold configuration."""
    text = "This product is damn bad and annoying."
    score = score_toxicity(text)
    # If score is around 0.3-0.5
    assert check_toxicity(text, threshold=0.2) is (score > 0.2)
    assert check_toxicity(text, threshold=0.9) is (score > 0.9)


def test_empty_or_none_input():
    """Empty or None text should return 0.0 score and check_toxicity == False."""
    assert score_toxicity("") == 0.0
    assert check_toxicity("") is False
    assert score_toxicity(None) == 0.0
    assert check_toxicity(None) is False


def test_toxicity_fallback_warning(caplog):
    """Verify fallback mechanism logs a clear warning about reduced-accuracy fallback."""
    # Reset warning flag to test log capture
    import src.guardrails.toxicity_check as tc
    tc._FALLBACK_WARNED = False

    with caplog.at_level(logging.WARNING):
        # Trigger score_toxicity; if detoxify is not installed, it logs warning
        score = tc.score_toxicity("test query with some words")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    # If detoxify was not installed, verify warning message was captured
    if not tc.HAS_DETOXIFY:
        assert any("Detoxify library not available" in record.message for record in caplog.records)
