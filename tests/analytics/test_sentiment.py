"""
Unit tests for VADER sentiment analysis in analytics module.
"""

import pytest

from src.analytics.sentiment import categorize_sentiment, ensure_vader_downloaded


def test_ensure_vader_downloaded():
    """Verify ensure_vader_downloaded succeeds and makes VADER available."""
    assert ensure_vader_downloaded() is True


def test_sentiment_positive():
    """Verify clearly positive text receives 'positive' sentiment category."""
    text = "Thank you so much Apple Support! The issue is completely resolved and works amazing."
    category = categorize_sentiment(text)
    assert category == "positive"


def test_sentiment_negative():
    """Verify clearly negative text receives 'negative' sentiment category."""
    text = "My phone is totally broken and keeps crashing. Terrible experience and awful customer service."
    category = categorize_sentiment(text)
    assert category == "negative"


def test_sentiment_neutral():
    """Verify neutral/informational text receives 'neutral' sentiment category."""
    text = "Does the store open at 10 AM on Saturday?"
    category = categorize_sentiment(text)
    assert category == "neutral"


def test_sentiment_empty_string():
    """Verify empty string safely returns 'neutral'."""
    assert categorize_sentiment("") == "neutral"


def test_sentiment_whitespace():
    """Verify whitespace-only string safely returns 'neutral'."""
    assert categorize_sentiment("   \n\t  ") == "neutral"
