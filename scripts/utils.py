"""
utils.py — Shared helper functions, configuration utilities, and logging support.

This module provides:
- Centralized logging configuration
- Environment variable loading
- GCP / BigQuery client helpers
- File I/O utilities
- Retry decorators
- Data validation helpers

Author: Capstone Project
"""

import os
import json
import logging
import functools
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
from dotenv import load_dotenv

# ──────────────────────────────────────────────
# 1. ENVIRONMENT & CONFIGURATION
# ──────────────────────────────────────────────

# Load .env variables at import time
load_dotenv()


def get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """
    Retrieve an environment variable safely.

    Args:
        key:      The name of the environment variable.
        default:  Fallback value when the variable is absent.
        required: Raise ValueError if True and the variable is missing.

    Returns:
        The value of the environment variable as a string.

    Raises:
        ValueError: If required=True and the variable is not set.
    """
    value = os.getenv(key, default)
    if required and value is None:
        raise ValueError(f"Required environment variable '{key}' is not set.")
    return value


# Centralised project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# GCP / BigQuery settings (read from .env)
GCP_PROJECT_ID = get_env("GCP_PROJECT_ID", "my-gcp-project")
GCP_DATASET_BRONZE = get_env("GCP_DATASET_BRONZE", "bronze")
GCP_DATASET_SILVER = get_env("GCP_DATASET_SILVER", "silver")
GCP_DATASET_GOLD = get_env("GCP_DATASET_GOLD", "gold")
GCS_BUCKET = get_env("GCS_BUCKET", "capstone-data-lake")
OPENAI_API_KEY = get_env("OPENAI_API_KEY", "")


# ──────────────────────────────────────────────
# 2. LOGGING
# ──────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create and return a configured logger that writes to both
    stdout and a rotating log file.

    Args:
        name:  Logger name (usually __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Avoid adding duplicate handlers on re-import
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # File handler
    log_file = LOGS_DIR / f"{name.replace('.', '_')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────
# 3. RETRY DECORATOR
# ──────────────────────────────────────────────

def retry(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Exponential-backoff retry decorator.

    Args:
        max_attempts:   Maximum number of attempts before raising.
        backoff_factor: Multiplier applied to wait time on each retry.
        exceptions:     Tuple of exception types to catch and retry on.

    Example:
        @retry(max_attempts=3, backoff_factor=2.0)
        def flaky_api_call():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger("retry")
            delay = 1.0
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        logger.error(
                            "Function '%s' failed after %d attempts. Last error: %s",
                            func.__name__, max_attempts, exc,
                        )
                        raise
                    logger.warning(
                        "Attempt %d/%d for '%s' failed: %s. Retrying in %.1fs …",
                        attempt, max_attempts, func.__name__, exc, delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# 4. FILE I/O HELPERS
# ──────────────────────────────────────────────

def read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame with logging."""
    logger = get_logger("utils.io")
    path = Path(path)
    logger.info("Reading CSV: %s", path)
    df = pd.read_csv(path, **kwargs)
    logger.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path.name)
    return df


def read_json(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read a JSON (lines or array) file into a pandas DataFrame."""
    logger = get_logger("utils.io")
    path = Path(path)
    logger.info("Reading JSON: %s", path)
    df = pd.read_json(path, **kwargs)
    logger.info("Loaded %d rows × %d cols from %s", len(df), len(df.columns), path.name)
    return df


def write_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Write a DataFrame to CSV, creating parent directories as needed."""
    logger = get_logger("utils.io")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, **kwargs)
    logger.info("Written %d rows to %s", len(df), path)


def write_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> None:
    """Write a DataFrame to Parquet format."""
    logger = get_logger("utils.io")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, **kwargs)
    logger.info("Written %d rows to %s (parquet)", len(df), path)


def load_json_config(path: Union[str, Path]) -> Dict:
    """Load a JSON configuration file."""
    with open(path, "r") as fh:
        return json.load(fh)


# ──────────────────────────────────────────────
# 5. DATA VALIDATION HELPERS
# ──────────────────────────────────────────────

def check_nulls(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
    """
    Check for null values in specified columns.

    Returns:
        Dict mapping column name → null count.
    """
    return {col: int(df[col].isna().sum()) for col in columns}


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """Return the number of duplicate rows in df."""
    return int(df.duplicated(subset=subset).sum())


def assert_no_nulls(df: pd.DataFrame, columns: List[str]) -> None:
    """
    Raise ValueError if any of the specified columns contain nulls.
    """
    nulls = check_nulls(df, columns)
    bad = {k: v for k, v in nulls.items() if v > 0}
    if bad:
        raise ValueError(f"Null values found: {bad}")


def assert_unique(df: pd.DataFrame, subset: List[str]) -> None:
    """
    Raise ValueError if the combination of columns is not unique.
    """
    dupes = check_duplicates(df, subset=subset)
    if dupes:
        raise ValueError(f"Found {dupes} duplicate rows on columns {subset}")


def validate_schema(df: pd.DataFrame, expected_columns: List[str]) -> None:
    """
    Validate that a DataFrame contains all expected columns.

    Raises:
        ValueError: If any expected column is missing.
    """
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


# ──────────────────────────────────────────────
# 6. GCP / BIGQUERY HELPERS
# ──────────────────────────────────────────────

def get_bigquery_client():
    """
    Return an authenticated BigQuery client.

    Requires GOOGLE_APPLICATION_CREDENTIALS env var pointing to a
    service-account JSON key file, or Application Default Credentials.
    """
    try:
        from google.cloud import bigquery
        return bigquery.Client(project=GCP_PROJECT_ID)
    except ImportError:
        raise ImportError(
            "google-cloud-bigquery is not installed. "
            "Run: pip install google-cloud-bigquery"
        )


def upload_df_to_bigquery(
    df: pd.DataFrame,
    table_id: str,
    dataset: str,
    if_exists: str = "replace",
) -> None:
    """
    Upload a pandas DataFrame to a BigQuery table.

    Args:
        df:        DataFrame to upload.
        table_id:  Target table name.
        dataset:   Target BigQuery dataset.
        if_exists: 'replace' | 'append' | 'fail'
    """
    from google.cloud import bigquery

    logger = get_logger("utils.bq")
    client = get_bigquery_client()
    full_table = f"{GCP_PROJECT_ID}.{dataset}.{table_id}"

    job_config = bigquery.LoadJobConfig(
        write_disposition=(
            bigquery.WriteDisposition.WRITE_TRUNCATE
            if if_exists == "replace"
            else bigquery.WriteDisposition.WRITE_APPEND
        ),
        autodetect=True,
    )

    logger.info("Uploading %d rows to BigQuery table: %s", len(df), full_table)
    job = client.load_table_from_dataframe(df, full_table, job_config=job_config)
    job.result()  # Wait for completion
    logger.info("Upload complete → %s", full_table)


# ──────────────────────────────────────────────
# 7. SYNTHETIC DATA GENERATION
# ──────────────────────────────────────────────

def generate_ecommerce_events(n_rows: int = 5_000_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic e-commerce event data for pipeline testing.

    Schema:
        event_id, user_id, session_id, product_id, category,
        event_type, quantity, price, discount, timestamp, country

    Args:
        n_rows: Number of rows to generate.
        seed:   Random seed for reproducibility.

    Returns:
        DataFrame with synthetic e-commerce events.
    """
    import numpy as np
    from datetime import datetime, timedelta

    logger = get_logger("utils.datagen")
    logger.info("Generating %d synthetic e-commerce rows …", n_rows)

    rng = np.random.default_rng(seed)

    categories = ["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"]
    event_types = ["page_view", "add_to_cart", "purchase", "wishlist", "review"]
    countries = ["US", "UK", "IN", "DE", "FR", "CA", "AU", "BR"]

    base_date = datetime(2023, 1, 1)
    timestamps = [
        base_date + timedelta(seconds=int(s))
        for s in rng.integers(0, 365 * 24 * 3600, n_rows)
    ]

    df = pd.DataFrame({
        "event_id":   [f"EVT_{i:010d}" for i in range(n_rows)],
        "user_id":    rng.integers(1, 500_000, n_rows),
        "session_id": [f"SES_{x:012d}" for x in rng.integers(0, 10**12, n_rows)],
        "product_id": rng.integers(1, 50_000, n_rows),
        "category":   rng.choice(categories, n_rows),
        "event_type": rng.choice(event_types, n_rows),
        "quantity":   rng.integers(1, 10, n_rows),
        "price":      rng.uniform(0.99, 999.99, n_rows).round(2),
        "discount":   rng.uniform(0.0, 0.5, n_rows).round(2),
        "timestamp":  timestamps,
        "country":    rng.choice(countries, n_rows),
    })

    # Intentionally inject ~2% nulls to test quality checks
    null_mask = rng.random(n_rows) < 0.02
    df.loc[null_mask, "price"] = None

    logger.info("Data generation complete.")
    return df


if __name__ == "__main__":
    # Quick smoke-test
    logger = get_logger("utils.__main__")
    df = generate_ecommerce_events(n_rows=10_000)
    logger.info("Sample:\n%s", df.head(3).to_string())
    logger.info("Nulls: %s", check_nulls(df, ["price"]))
    logger.info("Duplicates: %d", check_duplicates(df, subset=["event_id"]))
