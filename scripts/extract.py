"""
extract.py — Bronze Layer: Data Ingestion & Raw Landing

Responsibilities:
- Ingest synthetic e-commerce events (CSV / JSON / simulated API)
- Write raw data as Parquet files to the Bronze zone (data/raw/)
- Upload Bronze files to GCS (Google Cloud Storage)
- Register Bronze tables in BigQuery

Medallion Architecture: this is Layer 1 of 3 (Bronze → Silver → Gold).
At this layer data is stored exactly as received — no transformations.

Author: Capstone Project
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from scripts.utils import (
    DATA_RAW_DIR,
    GCS_BUCKET,
    GCP_DATASET_BRONZE,
    get_logger,
    get_env,
    generate_ecommerce_events,
    write_parquet,
    write_csv,
    retry,
    upload_df_to_bigquery,
)

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

BATCH_DATE = datetime.utcnow().strftime("%Y-%m-%d")
BRONZE_EVENTS_PATH = DATA_RAW_DIR / f"events_{BATCH_DATE}.parquet"
BRONZE_PRODUCTS_PATH = DATA_RAW_DIR / "products.parquet"
BRONZE_USERS_PATH = DATA_RAW_DIR / "users.parquet"

N_EVENTS = int(get_env("N_EVENTS", "500000"))   # 5 M in prod; 500k for local dev


# ──────────────────────────────────────────────
# 1. INGEST: SYNTHETIC E-COMMERCE EVENTS
# ──────────────────────────────────────────────

def ingest_events(n_rows: int = N_EVENTS) -> pd.DataFrame:
    """
    Generate and persist synthetic e-commerce events to the Bronze zone.

    In a real project this would call an actual API endpoint or read
    from Kafka / Pub/Sub.  Here we use deterministic synthetic data
    so the project works without external dependencies.

    Args:
        n_rows: Number of event rows to generate.

    Returns:
        DataFrame of raw events (before any cleaning).
    """
    logger.info("=== BRONZE EXTRACTION: E-Commerce Events ===")
    logger.info("Generating %d synthetic events …", n_rows)

    df = generate_ecommerce_events(n_rows=n_rows)

    # Write to Bronze (raw) zone as Parquet
    write_parquet(df, BRONZE_EVENTS_PATH)
    logger.info("Bronze events written → %s", BRONZE_EVENTS_PATH)

    # Also emit a CSV sample for quick inspection
    sample_path = DATA_RAW_DIR / "events_sample_1000.csv"
    write_csv(df.head(1000), sample_path)
    logger.info("CSV sample (1000 rows) → %s", sample_path)

    return df


# ──────────────────────────────────────────────
# 2. INGEST: PRODUCT DIMENSION (simulated API)
# ──────────────────────────────────────────────

@retry(max_attempts=3, backoff_factor=2.0, exceptions=(requests.RequestException,))
def fetch_product_catalog_from_api(api_url: str) -> pd.DataFrame:
    """
    Fetch product metadata from a REST API with retry logic.

    Args:
        api_url: Endpoint that returns a JSON array of product objects.

    Returns:
        DataFrame with product records.
    """
    logger.info("Fetching product catalog from API: %s", api_url)
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data)
    logger.info("Fetched %d product records from API.", len(df))
    return df


def ingest_products(api_url: Optional[str] = None) -> pd.DataFrame:
    """
    Ingest product dimension data.

    Uses a live API if `api_url` is provided, otherwise generates
    synthetic product data to keep the project self-contained.

    Args:
        api_url: Optional REST endpoint for real product catalog data.

    Returns:
        DataFrame of raw product records.
    """
    logger.info("=== BRONZE EXTRACTION: Product Catalog ===")

    if api_url:
        df = fetch_product_catalog_from_api(api_url)
    else:
        # Synthetic product catalog
        import numpy as np
        rng = np.random.default_rng(99)
        n = 50_000
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"]
        brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE"]

        df = pd.DataFrame({
            "product_id":   range(1, n + 1),
            "product_name": [f"Product_{i:05d}" for i in range(1, n + 1)],
            "category":     rng.choice(categories, n),
            "brand":        rng.choice(brands, n),
            "cost_price":   rng.uniform(1.0, 500.0, n).round(2),
            "list_price":   rng.uniform(5.0, 999.0, n).round(2),
            "stock_qty":    rng.integers(0, 10_000, n),
            "is_active":    rng.choice([True, False], n, p=[0.95, 0.05]),
        })
        logger.info("Synthetic product catalog generated: %d rows", len(df))

    write_parquet(df, BRONZE_PRODUCTS_PATH)
    logger.info("Bronze products written → %s", BRONZE_PRODUCTS_PATH)
    return df


# ──────────────────────────────────────────────
# 3. INGEST: USER DIMENSION (from JSON file)
# ──────────────────────────────────────────────

def ingest_users(json_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Ingest user dimension data from a JSON file.

    If no path is provided, generates synthetic user records.

    Args:
        json_path: Optional path to a JSON file of user records.

    Returns:
        DataFrame of raw user records.
    """
    logger.info("=== BRONZE EXTRACTION: User Dimension ===")

    if json_path and Path(json_path).exists():
        logger.info("Loading users from JSON: %s", json_path)
        df = pd.read_json(json_path)
    else:
        # Synthetic user data
        import numpy as np
        rng = np.random.default_rng(7)
        n = 500_000
        tiers = ["Bronze", "Silver", "Gold", "Platinum"]
        countries = ["US", "UK", "IN", "DE", "FR", "CA", "AU", "BR"]

        df = pd.DataFrame({
            "user_id":        range(1, n + 1),
            "username":       [f"user_{i:08d}" for i in range(1, n + 1)],
            "email":          [f"user_{i}@example.com" for i in range(1, n + 1)],
            "country":        rng.choice(countries, n),
            "loyalty_tier":   rng.choice(tiers, n),
            "signup_date":    pd.date_range("2020-01-01", periods=n, freq="1min")[:n],
            "is_active":      rng.choice([True, False], n, p=[0.9, 0.1]),
        })
        logger.info("Synthetic user dimension generated: %d rows", len(df))

    write_parquet(df, BRONZE_USERS_PATH)
    logger.info("Bronze users written → %s", BRONZE_USERS_PATH)
    return df


# ──────────────────────────────────────────────
# 4. UPLOAD BRONZE FILES TO GCS
# ──────────────────────────────────────────────

def upload_bronze_to_gcs(local_path: Path, gcs_prefix: str = "bronze") -> str:
    """
    Upload a local Bronze Parquet file to Google Cloud Storage.

    Args:
        local_path:  Path to the local Parquet file.
        gcs_prefix:  GCS folder prefix (default: 'bronze').

    Returns:
        GCS URI of the uploaded file.
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.warning("google-cloud-storage not installed; skipping GCS upload.")
        return ""

    logger.info("Uploading %s to GCS bucket %s …", local_path.name, GCS_BUCKET)
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob_name = f"{gcs_prefix}/{local_path.name}"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    gcs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
    logger.info("GCS upload complete → %s", gcs_uri)
    return gcs_uri


# ──────────────────────────────────────────────
# 5. REGISTER BRONZE TABLES IN BIGQUERY
# ──────────────────────────────────────────────

def register_bronze_in_bigquery(df: pd.DataFrame, table_name: str) -> None:
    """
    Load a Bronze DataFrame into BigQuery (append mode for events,
    replace mode for dimensions).

    Args:
        df:         DataFrame to load.
        table_name: Target BigQuery table name within the Bronze dataset.
    """
    try:
        upload_df_to_bigquery(
            df=df,
            table_id=table_name,
            dataset=GCP_DATASET_BRONZE,
            if_exists="replace",
        )
    except Exception as exc:
        logger.error("BigQuery registration failed for '%s': %s", table_name, exc)
        logger.warning("Continuing — Bronze Parquet files are the source of truth.")


# ──────────────────────────────────────────────
# 6. ORCHESTRATION ENTRY POINT
# ──────────────────────────────────────────────

def run_bronze_extraction(
    n_events: int = N_EVENTS,
    upload_to_gcs: bool = False,
    register_bq: bool = False,
) -> dict:
    """
    Main extraction function called by the Airflow DAG.

    Runs all three Bronze ingestion jobs and optionally uploads to GCS
    and registers tables in BigQuery.

    Args:
        n_events:    Number of event rows to generate.
        upload_to_gcs: Whether to upload Parquet files to GCS.
        register_bq:   Whether to register tables in BigQuery.

    Returns:
        Dict with paths to generated Bronze files.
    """
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   BRONZE LAYER EXTRACTION STARTED    ║")
    logger.info("╚══════════════════════════════════════╝")

    # Ingest all three sources
    df_events   = ingest_events(n_rows=n_events)
    df_products = ingest_products()
    df_users    = ingest_users()

    paths = {
        "events":   str(BRONZE_EVENTS_PATH),
        "products": str(BRONZE_PRODUCTS_PATH),
        "users":    str(BRONZE_USERS_PATH),
    }

    # Optional GCS upload
    if upload_to_gcs:
        for source, path in [
            ("events",   BRONZE_EVENTS_PATH),
            ("products", BRONZE_PRODUCTS_PATH),
            ("users",    BRONZE_USERS_PATH),
        ]:
            paths[f"{source}_gcs"] = upload_bronze_to_gcs(path)

    # Optional BigQuery registration
    if register_bq:
        register_bronze_in_bigquery(df_events, "raw_events")
        register_bronze_in_bigquery(df_products, "raw_products")
        register_bronze_in_bigquery(df_users, "raw_users")

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   BRONZE LAYER EXTRACTION COMPLETE   ║")
    logger.info("╚══════════════════════════════════════╝")
    logger.info("Output paths: %s", json.dumps(paths, indent=2))
    return paths


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bronze Layer Extraction")
    parser.add_argument("--n-events", type=int, default=N_EVENTS,
                        help="Number of event rows to generate")
    parser.add_argument("--upload-gcs", action="store_true",
                        help="Upload Bronze files to GCS")
    parser.add_argument("--register-bq", action="store_true",
                        help="Register Bronze tables in BigQuery")
    args = parser.parse_args()

    run_bronze_extraction(
        n_events=args.n_events,
        upload_to_gcs=args.upload_gcs,
        register_bq=args.register_bq,
    )
