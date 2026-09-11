"""Pydantic schema definitions for raw tweet ingestion."""

from datetime import datetime
from typing import Any, Optional
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator


class RawTweet(BaseModel):
    """Pydantic model representing a single raw tweet record from the customer support corpus.

    Downstream graph reconstruction (P1.1.F2) handles splitting response_tweet_id
    into list structures; this layer faithfully stores it as a raw string.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    tweet_id: str
    author_id: str
    inbound: bool
    created_at: datetime
    text: str
    response_tweet_id: Optional[str] = None
    in_response_to_tweet_id: Optional[str] = None

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, value: Any) -> datetime:
        """Parse created_at from string, datetime, or pandas Timestamp."""
        if isinstance(value, datetime):
            return value
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, str):
            val_str = value.strip()
            if not val_str:
                raise ValueError("created_at cannot be empty")
            try:
                # Fast-path for ISO format
                return datetime.fromisoformat(val_str)
            except (ValueError, TypeError):
                # Robust fallback for Twitter format: 'Tue Oct 31 22:10:47 +0000 2017'
                parsed = pd.to_datetime(val_str, errors="raise")
                if isinstance(parsed, pd.Timestamp):
                    return parsed.to_pydatetime()
                raise ValueError(f"Unable to parse created_at datetime: {value}")
        raise ValueError(f"Invalid type for created_at: {type(value)}")

    @field_validator("tweet_id", "author_id", mode="before")
    @classmethod
    def validate_id_strings(cls, value: Any) -> str:
        """Ensure tweet_id and author_id are non-null and non-empty."""
        if value is None or pd.isna(value):
            raise ValueError("ID field cannot be null or NaN")
        s = str(value).strip()
        if not s:
            raise ValueError("ID field cannot be empty")
        return s

    @field_validator("response_tweet_id", "in_response_to_tweet_id", mode="before")
    @classmethod
    def preserve_raw_optional_ids(cls, value: Any) -> Optional[str]:
        """Preserve raw response IDs as string, faithfully keeping whitespace or commas."""
        if value is None or pd.isna(value):
            return None
        s = str(value)
        if s.lower() == "nan" or s.strip() == "":
            return None
        return s
