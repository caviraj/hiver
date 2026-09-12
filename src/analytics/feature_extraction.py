"""Feature extraction from conversation threads for volume modeling."""

import logging
from typing import Iterable, List, Literal
import pandas as pd

from src.analytics.schema import VolumeFeatureRow
from src.analytics.sentiment import categorize_sentiment
from src.data.thread_schema import Thread

logger = logging.getLogger(__name__)


def _compute_time_of_day(hour: int) -> Literal["business_hours", "evening", "overnight"]:
    """Bucket hour into business_hours [9-17), evening [17-23), overnight [23-9)."""
    if 9 <= hour < 17:
        return "business_hours"
    elif 17 <= hour < 23:
        return "evening"
    else:
        return "overnight"


def _compute_day_of_week(weekday: int) -> Literal["weekday", "weekend"]:
    """Bucket weekday (0=Mon, 6=Sun) into weekday or weekend."""
    if weekday < 5:
        return "weekday"
    return "weekend"


def build_feature_row(thread: Thread) -> VolumeFeatureRow:
    """Extract predictive features and response count from a Thread.

    Parameters
    ----------
    thread : Thread
        The conversation thread to extract features from.

    Returns
    -------
    VolumeFeatureRow
        Extracted feature row.

    Raises
    ------
    ValueError
        If the thread is initiated by a brand (root_tweet.inbound is False).
    """
    root = thread.root_tweet
    if not root.inbound:
        raise ValueError(
            f"Thread {thread.thread_id} was initiated by a brand (inbound=False); "
            "expected customer-initiated thread."
        )

    # Compute follow_up_count: customer turns strictly after first brand response
    follow_up_count = 0
    first_brand_idx = None
    for idx, tweet in enumerate(thread.turns):
        if not tweet.inbound:
            first_brand_idx = idx
            break

    if first_brand_idx is not None:
        for tweet in thread.turns[first_brand_idx + 1 :]:
            if tweet.inbound:
                follow_up_count += 1

    # Predictor: sentiment
    sentiment_category = categorize_sentiment(root.text)

    # Predictor: text length (whitespace-split word count)
    if root.text and root.text.strip():
        text_length = len(root.text.split())
    else:
        text_length = 0

    # Predictors: temporal markers
    dt = root.created_at
    time_of_day = _compute_time_of_day(dt.hour)
    day_of_week = _compute_day_of_week(dt.weekday())

    return VolumeFeatureRow(
        thread_id=thread.thread_id,
        follow_up_count=follow_up_count,
        sentiment_category=sentiment_category,
        text_length=text_length,
        time_of_day=time_of_day,
        day_of_week=day_of_week,
    )


def extract_features_from_threads(threads: Iterable[Thread]) -> List[VolumeFeatureRow]:
    """Batch-extract feature rows from an iterable of threads.

    Brand-initiated threads are skipped and logged rather than raising an exception.

    Parameters
    ----------
    threads : Iterable[Thread]
        Sequence of threads to process.

    Returns
    -------
    List[VolumeFeatureRow]
        Extracted feature rows for all valid customer-initiated threads.
    """
    rows = []
    skipped = 0
    for thread in threads:
        try:
            row = build_feature_row(thread)
            rows.append(row)
        except ValueError as exc:
            skipped += 1
            logger.debug("Skipping invalid thread %s: %s", getattr(thread, "thread_id", "?"), exc)

    if skipped > 0:
        logger.info("Extracted %d feature rows (skipped %d brand-initiated threads)", len(rows), skipped)
    return rows


def features_to_dataframe(features: List[VolumeFeatureRow]) -> pd.DataFrame:
    """Convert a list of VolumeFeatureRow objects into a pandas DataFrame.

    Parameters
    ----------
    features : List[VolumeFeatureRow]
        List of extracted feature rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns and typed categoricals.
    """
    if not features:
        return pd.DataFrame(
            columns=[
                "thread_id",
                "follow_up_count",
                "sentiment_category",
                "text_length",
                "time_of_day",
                "day_of_week",
            ]
        )

    records = [f.model_dump() for f in features]
    df = pd.DataFrame.from_records(records)

    # Cast to category for statsmodels formula encoding
    df["sentiment_category"] = pd.Categorical(
        df["sentiment_category"], categories=["neutral", "positive", "negative"]
    )
    df["time_of_day"] = pd.Categorical(
        df["time_of_day"], categories=["business_hours", "evening", "overnight"]
    )
    df["day_of_week"] = pd.Categorical(
        df["day_of_week"], categories=["weekday", "weekend"]
    )
    df["follow_up_count"] = df["follow_up_count"].astype(int)
    df["text_length"] = df["text_length"].astype(int)

    return df
