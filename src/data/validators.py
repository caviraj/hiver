"""Row-level validation, sanitization, and rejection metrics for tweet data."""

from dataclasses import dataclass, field
import logging
from typing import Any, Optional, Set
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Tracks counts of processed, rejected, and sanitized records."""

    total_rows_processed: int = 0
    malformed_created_at_dropped: int = 0
    null_or_empty_author_dropped: int = 0
    null_or_empty_tweet_id_dropped: int = 0
    duplicates_dropped: int = 0
    null_bytes_sanitized: int = 0
    valid_rows: int = 0

    def total_dropped(self) -> int:
        """Calculate total number of dropped rows."""
        return (
            self.malformed_created_at_dropped
            + self.null_or_empty_author_dropped
            + self.null_or_empty_tweet_id_dropped
            + self.duplicates_dropped
        )

    def log_summary(self) -> str:
        """Log and return a human-readable summary of validation metrics."""
        summary = (
            f"Validation Summary:\n"
            f"  - Total rows ingested: {self.total_rows_processed}\n"
            f"  - Valid rows retained: {self.valid_rows}\n"
            f"  - Total dropped rows: {self.total_dropped()}\n"
            f"    * Malformed created_at dropped: {self.malformed_created_at_dropped}\n"
            f"    * Missing/empty author_id dropped: {self.null_or_empty_author_dropped}\n"
            f"    * Missing/empty tweet_id dropped: {self.null_or_empty_tweet_id_dropped}\n"
            f"    * Duplicate tweet_id dropped: {self.duplicates_dropped}\n"
            f"  - Text fields sanitized (null bytes removed): {self.null_bytes_sanitized}"
        )
        logger.info(summary)
        return summary


def sanitize_text(val: Any) -> str:
    """Sanitize text by decoding safely and eliminating null bytes.

    PyArrow / Parquet serialization fails when strings contain null bytes (\x00).
    Scraped tweets may also contain corrupted byte sequences.
    """
    if val is None or pd.isna(val):
        return ""
    if isinstance(val, bytes):
        s = val.decode("utf-8", errors="replace")
    else:
        s = str(val)

    # Clean null bytes which break Parquet/Arrow string columns
    if "\x00" in s:
        s = s.replace("\x00", "")
    return s


def validate_and_clean_chunk(
    df: pd.DataFrame,
    metrics: Optional[ValidationMetrics] = None,
    seen_tweet_ids: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Validate a chunk or full DataFrame, apply sanitization, and drop malformed rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw data chunk.
    metrics : Optional[ValidationMetrics]
        Validation metrics tracker to update with counts.
    seen_tweet_ids : Optional[Set[str]]
        Set of tweet_ids previously encountered (for cross-chunk deduplication).

    Returns
    -------
    pd.DataFrame
        Cleaned and validated DataFrame conforming to schema expectations.
    """
    if df.empty:
        return df.copy()

    if metrics is None:
        metrics = ValidationMetrics()

    initial_chunk_size = len(df)
    metrics.total_rows_processed += initial_chunk_size

    working_df = df.copy()

    # 1. Author ID validation: drop null or empty strings
    author_series = working_df["author_id"].astype(str).str.strip()
    author_is_invalid = (
        working_df["author_id"].isna()
        | (author_series == "")
        | (author_series.str.lower() == "nan")
        | (author_series.str.lower() == "none")
    )
    author_drop_count = int(author_is_invalid.sum())
    metrics.null_or_empty_author_dropped += author_drop_count
    if author_drop_count > 0:
        logger.debug("Dropping %d rows with empty or null author_id", author_drop_count)
    working_df = working_df.loc[~author_is_invalid].copy()

    # 2. Tweet ID validation: drop null or empty strings
    tweet_id_series = working_df["tweet_id"].astype(str).str.strip()
    tweet_id_is_invalid = (
        working_df["tweet_id"].isna()
        | (tweet_id_series == "")
        | (tweet_id_series.str.lower() == "nan")
        | (tweet_id_series.str.lower() == "none")
    )
    tweet_id_drop_count = int(tweet_id_is_invalid.sum())
    metrics.null_or_empty_tweet_id_dropped += tweet_id_drop_count
    if tweet_id_drop_count > 0:
        logger.debug("Dropping %d rows with empty or null tweet_id", tweet_id_drop_count)
    working_df = working_df.loc[~tweet_id_is_invalid].copy()

    # Normalize tweet_id to clean string
    working_df["tweet_id"] = working_df["tweet_id"].astype(str).str.strip()

    # 3. Deduplication: keep first occurrence
    # In-chunk deduplication
    before_dedup = len(working_df)
    working_df = working_df.drop_duplicates(subset=["tweet_id"], keep="first")
    in_chunk_dups = before_dedup - len(working_df)

    # Cross-chunk deduplication if tracker provided
    cross_chunk_dups = 0
    if seen_tweet_ids is not None:
        is_already_seen = working_df["tweet_id"].isin(seen_tweet_ids)
        cross_chunk_dups = int(is_already_seen.sum())
        if cross_chunk_dups > 0:
            working_df = working_df.loc[~is_already_seen].copy()
        # Add new tweet IDs to seen set
        seen_tweet_ids.update(working_df["tweet_id"].tolist())

    total_dups = in_chunk_dups + cross_chunk_dups
    metrics.duplicates_dropped += total_dups
    if total_dups > 0:
        logger.debug("Deduplicated %d duplicate tweet_id rows", total_dups)

    # 4. Datetime validation: coerce created_at, drop malformed rows
    parsed_dates = pd.to_datetime(working_df["created_at"], errors="coerce", format="mixed")
    malformed_date_mask = parsed_dates.isna()
    malformed_date_count = int(malformed_date_mask.sum())
    metrics.malformed_created_at_dropped += malformed_date_count
    if malformed_date_count > 0:
        logger.warning("Dropped %d malformed rows failing datetime parsing", malformed_date_count)
    working_df = working_df.loc[~malformed_date_mask].copy()
    working_df["created_at"] = parsed_dates.loc[~malformed_date_mask]

    # 5. Inbound boolean coercion
    if "inbound" in working_df.columns:
        if working_df["inbound"].dtype != bool:
            working_df["inbound"] = (
                working_df["inbound"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "t", "yes"])
            )

    # 6. Text sanitization: eliminate null bytes and handle non-UTF8
    null_byte_mask = working_df["text"].astype(str).str.contains("\x00", regex=False)
    null_byte_count = int(null_byte_mask.sum())
    metrics.null_bytes_sanitized += null_byte_count

    working_df["text"] = working_df["text"].apply(sanitize_text)

    # 7. Preserve raw response IDs faithfully (whitespace and commas kept)
    for col in ["response_tweet_id", "in_response_to_tweet_id"]:
        if col in working_df.columns:
            # Replace literal nan strings or None with pandas NA or clean string representation
            # but preserve valid ID strings with any internal spaces or commas
            working_df[col] = working_df[col].apply(
                lambda x: None if pd.isna(x) or str(x).lower() == "nan" or str(x).strip() == "" else str(x)
            )

    metrics.valid_rows += len(working_df)
    return working_df
