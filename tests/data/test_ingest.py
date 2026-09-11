"""Unit tests for dataset ingestion, schema validation, brand filtering, and persistence."""

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.schema import RawTweet
from src.data.validators import ValidationMetrics, sanitize_text, validate_and_clean_chunk
from src.data.ingest import load_config, load_dataset, filter_brand, save_to_parquet, run_ingestion_pipeline


# ---------------------------------------------------------------------------
# 1. Pydantic Schema Validation Tests
# ---------------------------------------------------------------------------

def test_raw_tweet_valid_schema():
    """Verify RawTweet instantiation with valid Twitter date and response IDs."""
    tweet_data = {
        "tweet_id": "119237",
        "author_id": "AppleSupport",
        "inbound": False,
        "created_at": "Tue Oct 31 22:10:47 +0000 2017",
        "text": "@user We'd love to help! Can you send us a DM?",
        "response_tweet_id": "119238, 119239",
        "in_response_to_tweet_id": "119236",
    }
    tweet = RawTweet(**tweet_data)

    assert tweet.tweet_id == "119237"
    assert tweet.author_id == "AppleSupport"
    assert tweet.inbound is False
    assert tweet.created_at == datetime(2017, 10, 31, 22, 10, 47, tzinfo=timezone.utc)
    # Preservation of raw string with whitespace
    assert tweet.response_tweet_id == "119238, 119239"
    assert tweet.in_response_to_tweet_id == "119236"


def test_raw_tweet_iso_datetime():
    """Verify RawTweet correctly parses standard ISO 8601 timestamps."""
    tweet = RawTweet(
        tweet_id="101",
        author_id="115714",
        inbound=True,
        created_at="2023-01-15T12:30:00Z",
        text="My iPhone screen is flickering.",
    )
    assert tweet.created_at.year == 2023
    assert tweet.created_at.month == 1
    assert tweet.created_at.day == 15


def test_raw_tweet_empty_id_raises_validation_error():
    """Empty or whitespace-only tweet_id must fail schema validation."""
    with pytest.raises(ValidationError):
        RawTweet(
            tweet_id="   ",
            author_id="AppleSupport",
            inbound=False,
            created_at="2023-01-15T12:00:00Z",
            text="Hello world",
        )


def test_raw_tweet_empty_author_raises_validation_error():
    """Empty or whitespace-only author_id must fail schema validation."""
    with pytest.raises(ValidationError):
        RawTweet(
            tweet_id="123",
            author_id="",
            inbound=False,
            created_at="2023-01-15T12:00:00Z",
            text="Hello world",
        )


# ---------------------------------------------------------------------------
# 2. Text Sanitization & Null-Byte Scrubbing
# ---------------------------------------------------------------------------

def test_sanitize_text_removes_null_bytes():
    """Null bytes must be completely stripped to safeguard PyArrow/Parquet."""
    dirty_text = "Corrupted\x00 text with\x00 embedded nulls"
    clean = sanitize_text(dirty_text)
    assert "\x00" not in clean
    assert clean == "Corrupted text with embedded nulls"


def test_sanitize_text_handles_none_and_non_utf8():
    """None and NaN values must return empty string, bytes decoded safely."""
    assert sanitize_text(None) == ""
    assert sanitize_text(float("nan")) == ""
    byte_input = b"Valid byte text \x00 with null"
    assert sanitize_text(byte_input) == "Valid byte text  with null"


# ---------------------------------------------------------------------------
# 3. Validator & Chunk Cleaning Tests
# ---------------------------------------------------------------------------

def test_validate_and_clean_drops_malformed_datetime():
    """Rows with unparseable created_at timestamps must be dropped."""
    df = pd.DataFrame([
        {
            "tweet_id": "1",
            "author_id": "user1",
            "inbound": True,
            "created_at": "Tue Oct 31 22:10:47 +0000 2017",
            "text": "Valid date tweet",
        },
        {
            "tweet_id": "2",
            "author_id": "user2",
            "inbound": True,
            "created_at": "not-a-real-date",
            "text": "Malformed date tweet",
        },
    ])
    metrics = ValidationMetrics()
    cleaned = validate_and_clean_chunk(df, metrics=metrics)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["tweet_id"] == "1"
    assert metrics.malformed_created_at_dropped == 1
    assert metrics.valid_rows == 1


def test_validate_and_clean_drops_null_or_empty_author():
    """Rows with missing, blank, or NaN author_id must be dropped."""
    df = pd.DataFrame([
        {"tweet_id": "1", "author_id": "AppleSupport", "inbound": False, "created_at": "2023-01-01", "text": "Valid"},
        {"tweet_id": "2", "author_id": "", "inbound": True, "created_at": "2023-01-01", "text": "Blank author"},
        {"tweet_id": "3", "author_id": "   ", "inbound": True, "created_at": "2023-01-01", "text": "Whitespace author"},
        {"tweet_id": "4", "author_id": None, "inbound": True, "created_at": "2023-01-01", "text": "None author"},
        {"tweet_id": "5", "author_id": "NaN", "inbound": True, "created_at": "2023-01-01", "text": "Literal NaN author"},
    ])
    metrics = ValidationMetrics()
    cleaned = validate_and_clean_chunk(df, metrics=metrics)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["tweet_id"] == "1"
    assert metrics.null_or_empty_author_dropped == 4


def test_validate_and_clean_drops_duplicates_and_preserves_first():
    """Duplicate tweet_ids within chunk and across chunks must be deduplicated."""
    df_chunk_1 = pd.DataFrame([
        {"tweet_id": "100", "author_id": "AppleSupport", "inbound": False, "created_at": "2023-01-01", "text": "First instance"},
        {"tweet_id": "100", "author_id": "AppleSupport", "inbound": False, "created_at": "2023-01-01", "text": "Duplicate in chunk"},
        {"tweet_id": "101", "author_id": "user1", "inbound": True, "created_at": "2023-01-01", "text": "Unique"},
    ])
    seen_ids = set()
    metrics = ValidationMetrics()

    cleaned_1 = validate_and_clean_chunk(df_chunk_1, metrics=metrics, seen_tweet_ids=seen_ids)
    assert len(cleaned_1) == 2
    assert cleaned_1.iloc[0]["text"] == "First instance"
    assert metrics.duplicates_dropped == 1

    # Second chunk has tweet_id 101 again
    df_chunk_2 = pd.DataFrame([
        {"tweet_id": "101", "author_id": "user1", "inbound": True, "created_at": "2023-01-01", "text": "Cross-chunk duplicate"},
        {"tweet_id": "102", "author_id": "user2", "inbound": True, "created_at": "2023-01-01", "text": "New row"},
    ])
    cleaned_2 = validate_and_clean_chunk(df_chunk_2, metrics=metrics, seen_tweet_ids=seen_ids)
    assert len(cleaned_2) == 1
    assert cleaned_2.iloc[0]["tweet_id"] == "102"
    assert metrics.duplicates_dropped == 2


def test_preservation_of_whitespace_padded_response_ids():
    """Response tweet IDs containing whitespace (e.g. '123, 456') must not be prematurely split."""
    df = pd.DataFrame([
        {
            "tweet_id": "200",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": "2023-01-01",
            "text": "Check your DM",
            "response_tweet_id": "201, 202, 203",
            "in_response_to_tweet_id": "199",
        }
    ])
    cleaned = validate_and_clean_chunk(df)
    assert cleaned.iloc[0]["response_tweet_id"] == "201, 202, 203"
    assert cleaned.iloc[0]["in_response_to_tweet_id"] == "199"


def test_inbound_boolean_coercion():
    """Inbound column values ('True', 'False', 1, 0) must be coerced to clean boolean."""
    df = pd.DataFrame([
        {"tweet_id": "1", "author_id": "u1", "inbound": "True", "created_at": "2023-01-01", "text": "T1"},
        {"tweet_id": "2", "author_id": "u2", "inbound": "FALSE", "created_at": "2023-01-01", "text": "T2"},
        {"tweet_id": "3", "author_id": "u3", "inbound": "1", "created_at": "2023-01-01", "text": "T3"},
        {"tweet_id": "4", "author_id": "u4", "inbound": "0", "created_at": "2023-01-01", "text": "T4"},
    ])
    cleaned = validate_and_clean_chunk(df)
    assert cleaned["inbound"].tolist() == [True, False, True, False]


# ---------------------------------------------------------------------------
# 4. Brand Filtering Tests
# ---------------------------------------------------------------------------

def test_filter_brand_isolates_brand_tweets():
    """filter_brand must isolate direct matches for the brand handle."""
    df = pd.DataFrame([
        {"tweet_id": "1", "author_id": "AppleSupport", "inbound": False, "created_at": "2023-01-01", "text": "Apple 1"},
        {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": False, "created_at": "2023-01-01", "text": "Amazon 1"},
        {"tweet_id": "3", "author_id": "AppleSupport", "inbound": False, "created_at": "2023-01-01", "text": "Apple 2"},
        {"tweet_id": "4", "author_id": "user123", "inbound": True, "created_at": "2023-01-01", "text": "Customer"},
    ])
    filtered = filter_brand(df, brand_handle="AppleSupport")

    assert len(filtered) == 2
    assert (filtered["author_id"] == "AppleSupport").all()
    assert set(filtered["tweet_id"]) == {"1", "3"}


def test_filter_brand_raises_value_error_on_zero_matches():
    """filter_brand must raise ValueError when 0 tweets match the handle."""
    df = pd.DataFrame([
        {"tweet_id": "1", "author_id": "AmazonHelp", "inbound": False, "created_at": "2023-01-01", "text": "Hello"},
        {"tweet_id": "2", "author_id": "Uber_Support", "inbound": False, "created_at": "2023-01-01", "text": "Help"},
    ])
    with pytest.raises(ValueError, match="No tweets found for brand handle 'AppleSupport'"):
        filter_brand(df, brand_handle="AppleSupport")


def test_filter_brand_raises_value_error_on_empty_df():
    """filter_brand on empty DataFrame must raise ValueError."""
    empty_df = pd.DataFrame(columns=["tweet_id", "author_id", "text"])
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        filter_brand(empty_df, brand_handle="AppleSupport")


# ---------------------------------------------------------------------------
# 5. Parquet Persistence & Ingestion Pipeline Tests
# ---------------------------------------------------------------------------

def test_save_to_parquet_and_read_back(tmp_path: Path):
    """Saving to Parquet and reading back must preserve all rows and schema types."""
    df = pd.DataFrame([
        {
            "tweet_id": "301",
            "author_id": "AppleSupport",
            "inbound": False,
            "created_at": pd.Timestamp("2023-05-01 10:00:00"),
            "text": "Direct message us for assistance.",
            "response_tweet_id": "302, 303",
            "in_response_to_tweet_id": "300",
        }
    ])
    output_file = tmp_path / "subdir" / "test_tweets.parquet"
    save_to_parquet(df, str(output_file))

    assert output_file.exists()

    reloaded_df = pd.read_parquet(output_file, engine="pyarrow")
    assert len(reloaded_df) == 1
    assert reloaded_df.iloc[0]["tweet_id"] == "301"
    assert reloaded_df.iloc[0]["response_tweet_id"] == "302, 303"
    assert bool(reloaded_df.iloc[0]["inbound"]) is False


def test_chunked_csv_loading_and_pipeline_integration(tmp_path: Path):
    """Verify chunked CSV loading, metrics tracking, filtering, and export."""
    csv_file = tmp_path / "synthetic_raw.csv"
    csv_content = (
        "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
        "1001,AppleSupport,False,Tue Oct 31 22:10:47 +0000 2017,How can we help?,1002,1000\n"
        "1002,user_1,True,Tue Oct 31 22:11:00 +0000 2017,My phone is hot,,1001\n"
        "1003,AppleSupport,False,Tue Oct 31 22:15:00 +0000 2017,Please DM us\x00 now,,1002\n"
        "1004,AmazonHelp,False,Tue Oct 31 22:20:00 +0000 2017,Order issue help,,\n"
        "1005,,True,Tue Oct 31 22:25:00 +0000 2017,Blank author row,,\n"
        "1006,user_2,True,invalid-date-format,Invalid date row,,\n"
        "1001,AppleSupport,False,Tue Oct 31 22:10:47 +0000 2017,Duplicate tweet,,\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")

    # Load dataset with chunksize=2 to test chunking loop and deduplication
    cleaned_df = load_dataset(path=str(csv_file), chunksize=2)

    # 7 raw rows:
    # - 1005 dropped (missing author)
    # - 1006 dropped (malformed date)
    # - second 1001 dropped (duplicate tweet_id)
    # Total valid rows = 4 (1001, 1002, 1003, 1004)
    assert len(cleaned_df) == 4
    assert set(cleaned_df["tweet_id"]) == {"1001", "1002", "1003", "1004"}

    # Text for 1003 should have null byte removed
    tweet_1003 = cleaned_df.loc[cleaned_df["tweet_id"] == "1003"].iloc[0]
    assert "\x00" not in tweet_1003["text"]
    assert "Please DM us" in tweet_1003["text"]

    # Filter brand
    apple_df = filter_brand(cleaned_df, brand_handle="AppleSupport")
    assert len(apple_df) == 2
    assert set(apple_df["tweet_id"]) == {"1001", "1003"}

    # Persist
    parquet_out = tmp_path / "processed" / "apple.parquet"
    save_to_parquet(apple_df, str(parquet_out))
    assert parquet_out.exists()


def test_load_config_fallback_and_custom(tmp_path: Path):
    """load_config should return dictionary when file exists, or empty dict when missing."""
    custom_yaml = tmp_path / "custom_config.yaml"
    custom_yaml.write_text("brand_handle: TestBrand\nchunksize: 1000\n", encoding="utf-8")

    cfg = load_config(str(custom_yaml))
    assert cfg["brand_handle"] == "TestBrand"
    assert cfg["chunksize"] == 1000

    missing_cfg = load_config(str(tmp_path / "nonexistent.yaml"))
    assert missing_cfg == {}
