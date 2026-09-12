"""
Unit tests for analytics feature extraction pipeline.
"""

from datetime import datetime
import pytest

from src.data.schema import RawTweet
from src.data.thread_schema import Thread
from src.analytics.feature_extraction import (
    build_feature_row,
    extract_features_from_threads,
    features_to_dataframe,
)
from src.analytics.schema import VolumeFeatureRow


def make_tweet(
    tweet_id: str,
    author_id: str,
    inbound: bool,
    created_at_str: str,
    text: str = "Test message content",
) -> RawTweet:
    """Helper to create RawTweet objects for testing."""
    return RawTweet(
        tweet_id=tweet_id,
        author_id=author_id,
        inbound=inbound,
        created_at=datetime.fromisoformat(created_at_str),
        text=text,
    )


def test_build_feature_row_brand_initiated_raises():
    """Verify thread initiated by brand (inbound=False) raises ValueError."""
    brand_root = make_tweet("101", "AppleSupport", False, "2023-10-16T10:00:00")
    customer_reply = make_tweet("102", "user1", True, "2023-10-16T10:05:00")
    thread = Thread(
        thread_id="101",
        tweets=[brand_root, customer_reply],
        terminal=False,
    )

    with pytest.raises(ValueError, match="initiated by a brand"):
        build_feature_row(thread)


def test_build_feature_row_single_turn_thread():
    """Verify customer thread with no brand reply has follow_up_count=0."""
    customer_root = make_tweet(
        "201", "user1", True, "2023-10-16T10:00:00", text="Need help with iPhone"
    )
    thread = Thread(
        thread_id="201",
        tweets=[customer_root],
        terminal=False,
    )

    row = build_feature_row(thread)
    assert isinstance(row, VolumeFeatureRow)
    assert row.thread_id == "201"
    assert row.follow_up_count == 0
    assert row.text_length == 4
    assert row.time_of_day == "business_hours"
    assert row.day_of_week == "weekday"  # 2023-10-16 is Monday


def test_build_feature_row_multi_turn_follow_ups():
    """Verify follow_up_count correctly counts customer turns after first brand reply."""
    # Turn 0: Customer (root)
    # Turn 1: Brand reply 1
    # Turn 2: Customer follow-up 1
    # Turn 3: Brand reply 2
    # Turn 4: Customer follow-up 2
    # Turn 5: Customer follow-up 3 (immediate double message)
    t0 = make_tweet("301", "user1", True, "2023-10-16T10:00:00", "Problem with iPad")
    t1 = make_tweet("302", "AppleSupport", False, "2023-10-16T10:15:00", "DM us details")
    t2 = make_tweet("303", "user1", True, "2023-10-16T10:20:00", "Sent DM")
    t3 = make_tweet("304", "AppleSupport", False, "2023-10-16T10:25:00", "Looking now")
    t4 = make_tweet("305", "user1", True, "2023-10-16T10:30:00", "Thanks")
    t5 = make_tweet("306", "user1", True, "2023-10-16T10:31:00", "Also battery is hot")

    thread = Thread(
        thread_id="301",
        tweets=[t0, t1, t2, t3, t4, t5],
        terminal=False,
    )

    row = build_feature_row(thread)
    # Follow-ups strictly after t1: t2, t4, t5 -> 3
    assert row.follow_up_count == 3
    assert row.thread_id == "301"


def test_temporal_bucketing_evening_and_weekend():
    """Verify time_of_day and day_of_week categorization across ranges."""
    # 2023-10-21 is Saturday, 19:30 is evening
    t_weekend_evening = make_tweet(
        "401", "user1", True, "2023-10-21T19:30:00", "Screen is cracked"
    )
    thread = Thread(
        thread_id="401",
        tweets=[t_weekend_evening],
        terminal=False,
    )
    row = build_feature_row(thread)
    assert row.time_of_day == "evening"
    assert row.day_of_week == "weekend"

    # 2023-10-17 is Tuesday, 03:15 is overnight
    t_weekday_overnight = make_tweet(
        "402", "user2", True, "2023-10-17T03:15:00", "Help"
    )
    thread2 = Thread(
        thread_id="402",
        tweets=[t_weekday_overnight],
        terminal=False,
    )
    row2 = build_feature_row(thread2)
    assert row2.time_of_day == "overnight"
    assert row2.day_of_week == "weekday"


def test_extract_features_from_threads_skips_invalid():
    """Verify batch extraction skips invalid brand-initiated threads gracefully."""
    valid_t = make_tweet("501", "user1", True, "2023-10-16T10:00:00", "Valid thread")
    invalid_t = make_tweet("502", "AppleSupport", False, "2023-10-16T10:00:00", "Brand root")

    thread_valid = Thread(thread_id="501", tweets=[valid_t], terminal=False)
    thread_invalid = Thread(thread_id="502", tweets=[invalid_t], terminal=False)

    rows = extract_features_from_threads([thread_valid, thread_invalid])
    assert len(rows) == 1
    assert rows[0].thread_id == "501"


def test_features_to_dataframe():
    """Verify features_to_dataframe produces properly formatted DataFrame."""
    t1 = make_tweet("601", "user1", True, "2023-10-16T10:00:00", "Issue with MacBook")
    thread = Thread(thread_id="601", tweets=[t1], terminal=False)

    rows = extract_features_from_threads([thread])
    df = features_to_dataframe(rows)

    assert len(df) == 1
    expected_cols = {
        "thread_id",
        "follow_up_count",
        "sentiment_category",
        "text_length",
        "time_of_day",
        "day_of_week",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["follow_up_count"].dtype in ("int64", "int32")
    assert df["text_length"].dtype in ("int64", "int32")
