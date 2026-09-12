"""Sentiment classification using NLTK VADER lexicon."""

from typing import Literal, Optional
import nltk

_ANALYZER = None


def ensure_vader_downloaded() -> bool:
    """Ensure that the VADER lexicon is downloaded and available.

    Returns
    -------
    bool
        True if available and initialized successfully.
    """
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        SentimentIntensityAnalyzer()
        return True
    except (LookupError, AttributeError):
        try:
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            SentimentIntensityAnalyzer()
            return True
        except Exception as exc:
            raise ImportError(
                "NLTK VADER lexicon not found. Please run: python -c \"import nltk; nltk.download('vader_lexicon')\""
            ) from exc


def _get_analyzer():
    """Retrieve or initialize the SentimentIntensityAnalyzer with preflight check."""
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER

    ensure_vader_downloaded()
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    _ANALYZER = SentimentIntensityAnalyzer()
    return _ANALYZER


def categorize_sentiment(text: str) -> Literal["positive", "neutral", "negative"]:
    """Categorize text sentiment as positive, neutral, or negative using VADER.

    Empty or whitespace-only strings return "neutral".
    Standard VADER compound thresholds:
    - compound >= 0.05 -> "positive"
    - compound <= -0.05 -> "negative"
    - otherwise -> "neutral"

    Parameters
    ----------
    text : str
        Input text to analyze.

    Returns
    -------
    Literal["positive", "neutral", "negative"]
        Assigned sentiment category.
    """
    if not text or not text.strip():
        return "neutral"

    analyzer = _get_analyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)

    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"
