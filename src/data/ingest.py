"""Dataset ingestion, chunked CSV reading, brand filtering, and Parquet export."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Set
import pandas as pd
import yaml

from src.data.validators import ValidationMetrics, validate_and_clean_chunk

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "data_config.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file.

    Parameters
    ----------
    config_path : Optional[str]
        Path to YAML config file. If None, falls back to default project location.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    target_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not target_path.exists():
        logger.warning("Config file not found at %s. Returning empty config.", target_path)
        return {}

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config


def load_dataset(
    path: Optional[str] = None,
    chunksize: Optional[int] = None,
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load and validate raw dataset using chunked reading.

    Streams the CSV file in chunks (defaulting to config chunksize, e.g. 50,000),
    validates and cleans each chunk, and aggregates retained rows into a single
    DataFrame.

    Parameters
    ----------
    path : Optional[str]
        Path to raw CSV dataset. Defaults to raw_data_path in config.
    chunksize : Optional[int]
        Number of rows per chunk. Defaults to chunksize in config.
    config_path : Optional[str]
        Optional path to override default config file.

    Returns
    -------
    pd.DataFrame
        Aggregated, validated DataFrame.
    """
    config = load_config(config_path)
    file_path = path or config.get("raw_data_path", "data/raw/twcs.csv")
    chunk_size = chunksize if chunksize is not None else config.get("chunksize", 50000)
    encoding = config.get("encoding", "utf-8")
    encoding_errors = config.get("encoding_errors", "replace")
    column_mapping = config.get("columns", {})

    logger.info("Starting dataset ingestion from: %s (chunksize=%s)", file_path, chunk_size)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw dataset file not found at path: {file_path}")

    metrics = ValidationMetrics()
    seen_tweet_ids: Set[str] = set()
    cleaned_chunks = []

    # Read CSV with chunking to handle large files (e.g. Kaggle 3M row dataset)
    if chunk_size and chunk_size > 0:
        reader = pd.read_csv(
            file_path,
            chunksize=chunk_size,
            encoding=encoding,
            encoding_errors=encoding_errors,
            dtype=str,
        )
        for chunk_idx, chunk in enumerate(reader):
            # Optional column rename if mapping provided
            if column_mapping:
                chunk = chunk.rename(columns=column_mapping)

            cleaned_chunk = validate_and_clean_chunk(
                chunk,
                metrics=metrics,
                seen_tweet_ids=seen_tweet_ids,
            )
            if not cleaned_chunk.empty:
                cleaned_chunks.append(cleaned_chunk)
            logger.debug("Processed chunk %d, retained %d rows", chunk_idx, len(cleaned_chunk))
    else:
        # Single-pass read
        full_df = pd.read_csv(
            file_path,
            encoding=encoding,
            encoding_errors=encoding_errors,
            dtype=str,
        )
        if column_mapping:
            full_df = full_df.rename(columns=column_mapping)

        cleaned_chunk = validate_and_clean_chunk(
            full_df,
            metrics=metrics,
            seen_tweet_ids=seen_tweet_ids,
        )
        if not cleaned_chunk.empty:
            cleaned_chunks.append(cleaned_chunk)

    metrics.log_summary()

    if not cleaned_chunks:
        logger.warning("Ingestion completed with 0 valid rows retained.")
        return pd.DataFrame()

    aggregated_df = pd.concat(cleaned_chunks, ignore_index=True)
    logger.info("Dataset ingestion finished. Total valid rows: %d", len(aggregated_df))
    return aggregated_df


def filter_brand(
    df: pd.DataFrame,
    brand_handle: Optional[str] = None,
    config_path: Optional[str] = None,
    expand_threads: bool = True,
) -> pd.DataFrame:
    """Filter dataset for tweets matching a specific brand handle.

    Parameters
    ----------
    df : pd.DataFrame
        Validated tweet DataFrame.
    brand_handle : Optional[str]
        Brand handle to filter (e.g., 'AppleSupport'). Defaults to brand_handle in config.
    config_path : Optional[str]
        Optional path to YAML config file.
    expand_threads : bool, default=True
        Whether to expand brand filtering to include inbound consumer tweets that belong
        to a conversational thread containing at least one brand-authored tweet.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing tweets for the target brand and related thread tweets.

    Raises
    ------
    ValueError
        If 0 tweets match the specified brand handle or thread criteria.
    """
    config = load_config(config_path)
    target_brand = brand_handle or config.get("brand_handle", "AppleSupport")

    if df.empty:
        raise ValueError(f"Input DataFrame is empty. Cannot filter for brand '{target_brand}'.")

    if "author_id" not in df.columns:
        raise ValueError("DataFrame missing required 'author_id' column for brand filtering.")

    # Direct match on brand author
    is_brand_author = df["author_id"] == target_brand

    # Graph-expanded matching: include inbound==True rows belonging to threads with >=1 brand tweet
    has_graph_cols = {"response_tweet_id", "in_response_to_tweet_id", "tweet_id"}.issubset(df.columns)
    if expand_threads and has_graph_cols:
        from src.data.graph import build_graph, reconstruct_threads

        try:
            graph = build_graph(df)
            threads = reconstruct_threads(graph, brand_handle=target_brand)
            brand_thread_tweet_ids = {t.tweet_id for th in threads for t in th.tweets}

            if "inbound" in df.columns:
                inbound_series = df["inbound"]
                if pd.api.types.is_bool_dtype(inbound_series):
                    is_inbound = inbound_series
                else:
                    is_inbound = inbound_series.astype(str).str.strip().str.lower().isin(["true", "1", "t"])
                mask = is_brand_author | (is_inbound & df["tweet_id"].isin(brand_thread_tweet_ids))
            else:
                mask = is_brand_author | df["tweet_id"].isin(brand_thread_tweet_ids)
            filtered_df = df.loc[mask].copy()
        except Exception as exc:
            logger.warning("Graph-expanded brand filtering failed with %s; falling back to direct brand match.", exc)
            filtered_df = df.loc[is_brand_author].copy()
    else:
        filtered_df = df.loc[is_brand_author].copy()

    matched_count = len(filtered_df)
    if matched_count == 0:
        raise ValueError(
            f"No tweets found for brand handle '{target_brand}'. "
            "Please verify handle spelling or verify that the raw dataset contains this brand."
        )

    logger.info("Filtered %d tweets for brand '%s' (expand_threads=%s)", matched_count, target_brand, expand_threads)
    return filtered_df


def save_to_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Export DataFrame to Parquet format using PyArrow.

    Ensures parent directories exist before writing.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and filtered DataFrame to persist.
    output_path : str
        Destination path for the Parquet file.
    """
    dest_path = Path(output_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(dest_path, engine="pyarrow", index=False)
    logger.info("Successfully exported %d rows to Parquet: %s", len(df), dest_path)


def run_ingestion_pipeline(config_path: str = "config/data_config.yaml") -> pd.DataFrame:
    """Execute complete ingestion pipeline: load, validate, filter, and export.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file.

    Returns
    -------
    pd.DataFrame
        Final filtered DataFrame saved to Parquet.
    """
    config = load_config(config_path)
    raw_path = config.get("raw_data_path", "data/raw/twcs.csv")
    processed_path = config.get("processed_data_path", "data/processed/apple_support_raw.parquet")
    brand_handle = config.get("brand_handle", "AppleSupport")
    chunksize = config.get("chunksize", 50000)

    # 1. Load and validate
    cleaned_df = load_dataset(path=raw_path, chunksize=chunksize, config_path=config_path)

    # 2. Filter brand
    brand_df = filter_brand(cleaned_df, brand_handle=brand_handle, config_path=config_path)

    # 3. Export to Parquet
    save_to_parquet(brand_df, output_path=processed_path)

    return brand_df
