"""
load.py — Gold Layer: Analytics-Ready Loading

Responsibilities:
- Read Silver Parquet files
- Build star-schema fact and dimension tables
- Write Gold analytics tables to BigQuery
- Expose summary metrics for the Airflow DAG

Medallion Architecture: this is Layer 3 of 3 (Bronze → Silver → Gold).
Gold data is fully modelled, business-friendly, and dashboard-ready.

Author: Capstone Project
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

from scripts.utils import (
    DATA_PROCESSED_DIR,
    GCP_DATASET_GOLD,
    get_logger,
    write_parquet,
    upload_df_to_bigquery,
)

logger = get_logger(__name__)

BATCH_DATE = datetime.utcnow().strftime("%Y-%m-%d")

GOLD_DIR = DATA_PROCESSED_DIR / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 1. DIMENSION: DATE
# ──────────────────────────────────────────────

def build_dim_date(start: str = "2023-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """
    Build a Date dimension table spanning the project date range.

    Returns:
        DataFrame with date_key, full_date, year, quarter, month,
        week, day_of_week, is_weekend columns.
    """
    logger.info("Building dim_date (%s → %s) …", start, end)
    dates = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame({
        "date_key":      dates.strftime("%Y%m%d").astype(int),
        "full_date":     dates.date,
        "year":          dates.year,
        "quarter":       dates.quarter,
        "month":         dates.month,
        "month_name":    dates.strftime("%B"),
        "week":          dates.isocalendar().week.values,
        "day_of_week":   dates.dayofweek,
        "day_name":      dates.strftime("%A"),
        "is_weekend":    (dates.dayofweek >= 5).astype(int),
    })
    logger.info("dim_date built: %d rows", len(df))
    return df


# ──────────────────────────────────────────────
# 2. DIMENSION: PRODUCT
# ──────────────────────────────────────────────

def build_dim_product(df_products: pd.DataFrame) -> pd.DataFrame:
    """
    Build the Product dimension from the Silver product data.

    Returns:
        Conformed product dimension with surrogate key.
    """
    logger.info("Building dim_product …")
    df = df_products.copy()
    df = df.rename(columns={
        "product_id":   "product_key",
        "product_name": "product_name",
    })
    # Select and order SCD-1 attributes
    cols = [
        "product_key", "product_name", "category", "brand",
        "cost_price", "list_price", "margin_pct", "is_active",
    ]
    available = [c for c in cols if c in df.columns]
    df = df[available].drop_duplicates(subset=["product_key"])
    logger.info("dim_product built: %d rows", len(df))
    return df


# ──────────────────────────────────────────────
# 3. DIMENSION: USER
# ──────────────────────────────────────────────

def build_dim_user(df_users: pd.DataFrame) -> pd.DataFrame:
    """
    Build the User dimension from the Silver user data.

    Returns:
        Conformed user dimension with surrogate key.
    """
    logger.info("Building dim_user …")
    df = df_users.copy()
    df = df.rename(columns={"user_id": "user_key"})
    cols = [
        "user_key", "username", "country",
        "loyalty_tier", "signup_date", "is_active",
    ]
    available = [c for c in cols if c in df.columns]
    df = df[available].drop_duplicates(subset=["user_key"])
    logger.info("dim_user built: %d rows", len(df))
    return df


# ──────────────────────────────────────────────
# 4. FACT: SALES
# ──────────────────────────────────────────────

def build_fact_sales(
    df_events: pd.DataFrame,
    df_products: pd.DataFrame,
    df_users: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the central Sales fact table from Silver events.

    Grain: One row per purchase event.
    Measures: quantity, price, discount, revenue.
    Foreign keys: date_key, user_key, product_key.

    Args:
        df_events:   Silver events DataFrame.
        df_products: Silver products DataFrame (for dim lookup).
        df_users:    Silver users DataFrame (for dim lookup).

    Returns:
        Fact table ready for star-schema queries.
    """
    logger.info("Building fact_sales …")

    # Filter to purchase events only (fact grain)
    df = df_events[df_events["event_type"] == "purchase"].copy()
    logger.info("Purchase events: %d rows", len(df))

    # Generate surrogate date key
    df["date_key"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y%m%d").astype(int)

    # Enrich with product dimension attributes via lookup
    product_lookup = (
        df_products[["product_id", "brand", "margin_pct"]].copy()
        if "product_id" in df_products.columns
        else pd.DataFrame({"product_id": [], "brand": [], "margin_pct": []})
    )
    df = df.merge(
        product_lookup,
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    # Enrich with user dimension attributes
    user_lookup = (
        df_users[["user_id", "loyalty_tier", "country"]].copy()
        .rename(columns={"country": "user_country"})
        if "user_id" in df_users.columns
        else pd.DataFrame({"user_id": [], "loyalty_tier": [], "user_country": []})
    )
    df = df.merge(
        user_lookup,
        on="user_id",
        how="left",
        validate="many_to_one",
    )

    # Select and rename fact columns
    df = df.rename(columns={
        "event_id":   "sale_key",
        "user_id":    "user_key",
        "product_id": "product_key",
    })

    fact_cols = [
        "sale_key", "date_key", "event_date", "user_key", "product_key",
        "category", "country", "user_country",
        "event_type", "quantity", "price", "discount",
        "revenue", "brand", "loyalty_tier", "margin_pct",
        "event_hour", "_ingested_at",
    ]
    available = [c for c in fact_cols if c in df.columns]
    df = df[available]
    df["_loaded_at"] = datetime.utcnow()

    logger.info("fact_sales built: %d rows", len(df))
    return df


# ──────────────────────────────────────────────
# 5. AGGREGATE: DAILY SALES SUMMARY
# ──────────────────────────────────────────────

def build_agg_daily_sales(df_fact: pd.DataFrame) -> pd.DataFrame:
    """
    Build a pre-aggregated daily sales summary for dashboards.

    Returns:
        DataFrame aggregated at day × category level.
    """
    logger.info("Building agg_daily_sales …")

    # Handle missing columns gracefully
    group_cols = [c for c in ["date_key", "category", "country"] if c in df_fact.columns]

    df_agg = (
        df_fact.groupby(group_cols)
        .agg(
            total_orders    = ("sale_key",  "count"),
            total_quantity  = ("quantity",  "sum"),
            total_revenue   = ("revenue",   "sum"),
            avg_order_value = ("revenue",   "mean"),
            avg_discount    = ("discount",  "mean"),
        )
        .reset_index()
        .round(2)
    )

    df_agg["_loaded_at"] = datetime.utcnow()
    logger.info("agg_daily_sales built: %d rows", len(df_agg))
    return df_agg


# ──────────────────────────────────────────────
# 6. AGGREGATE: USER LIFETIME VALUE
# ──────────────────────────────────────────────

def build_agg_user_ltv(df_fact: pd.DataFrame) -> pd.DataFrame:
    """
    Build a User Lifetime Value (LTV) aggregate table.

    Returns:
        DataFrame with one row per user summarising spend and activity.
    """
    logger.info("Building agg_user_ltv …")

    group_cols = [c for c in ["user_key", "loyalty_tier", "user_country"] if c in df_fact.columns]

    df_ltv = (
        df_fact.groupby(group_cols)
        .agg(
            total_orders      = ("sale_key",  "count"),
            total_spend       = ("revenue",   "sum"),
            avg_order_value   = ("revenue",   "mean"),
            first_purchase    = ("date_key",  "min"),
            last_purchase     = ("date_key",  "max"),
        )
        .reset_index()
        .round(2)
    )

    if len(df_ltv):
        df_ltv["ltv_segment"] = pd.cut(
            df_ltv["total_spend"],
            bins=[0, 100, 500, 2000, float("inf")],
            labels=["Low", "Medium", "High", "VIP"],
        )

    df_ltv["_loaded_at"] = datetime.utcnow()
    logger.info("agg_user_ltv built: %d rows", len(df_ltv))
    return df_ltv


# ──────────────────────────────────────────────
# 7. PERSIST GOLD TABLES
# ──────────────────────────────────────────────

def persist_gold_tables(tables: Dict[str, pd.DataFrame], register_bq: bool = False) -> Dict[str, str]:
    """
    Write Gold tables to local Parquet and optionally to BigQuery.

    Args:
        tables:      Dict of {table_name: DataFrame}.
        register_bq: If True, also upload to BigQuery Gold dataset.

    Returns:
        Dict of {table_name: local_parquet_path}.
    """
    paths = {}
    for name, df in tables.items():
        if df is None or len(df) == 0:
            logger.warning("Skipping empty table: %s", name)
            continue
        path = GOLD_DIR / f"{name}.parquet"
        write_parquet(df, path)
        paths[name] = str(path)

        if register_bq:
            try:
                upload_df_to_bigquery(df, name, GCP_DATASET_GOLD)
            except Exception as exc:
                logger.error("BigQuery upload failed for %s: %s", name, exc)
    return paths


# ──────────────────────────────────────────────
# 8. ORCHESTRATION ENTRY POINT
# ──────────────────────────────────────────────

def run_gold_load(register_bq: bool = False) -> dict:
    """
    Main Gold Layer loading function called by the Airflow DAG.

    Args:
        register_bq: Whether to upload Gold tables to BigQuery.

    Returns:
        Dict with table paths and row counts.
    """
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║     GOLD LAYER LOADING STARTED       ║")
    logger.info("╚══════════════════════════════════════╝")

    # ── Load Silver files ─────────────────────────────────────────────
    silver_events_files = sorted(DATA_PROCESSED_DIR.glob("silver_events_*.parquet"))
    if not silver_events_files:
        raise FileNotFoundError(
            f"No Silver events file found in {DATA_PROCESSED_DIR}. "
            "Run transform.py first."
        )
    df_events = pd.read_parquet(silver_events_files[-1])
    logger.info("Loaded Silver events: %d rows", len(df_events))

    df_products = pd.DataFrame()
    products_path = DATA_PROCESSED_DIR / "silver_products.parquet"
    if products_path.exists():
        df_products = pd.read_parquet(products_path)

    df_users = pd.DataFrame()
    users_path = DATA_PROCESSED_DIR / "silver_users.parquet"
    if users_path.exists():
        df_users = pd.read_parquet(users_path)

    # ── Build Gold Tables ─────────────────────────────────────────────
    dim_date    = build_dim_date()
    dim_product = build_dim_product(df_products) if len(df_products) else pd.DataFrame()
    dim_user    = build_dim_user(df_users) if len(df_users) else pd.DataFrame()
    fact_sales  = build_fact_sales(df_events, df_products, df_users)
    agg_daily   = build_agg_daily_sales(fact_sales)
    agg_ltv     = build_agg_user_ltv(fact_sales)

    tables = {
        "dim_date":        dim_date,
        "dim_product":     dim_product,
        "dim_user":        dim_user,
        "fact_sales":      fact_sales,
        "agg_daily_sales": agg_daily,
        "agg_user_ltv":    agg_ltv,
    }

    paths = persist_gold_tables(tables, register_bq=register_bq)

    summary = {
        table: len(df)
        for table, df in tables.items()
        if df is not None
    }

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║     GOLD LAYER LOADING COMPLETE      ║")
    logger.info("╚══════════════════════════════════════╝")
    logger.info("Row counts: %s", json.dumps(summary, indent=2))
    return {"paths": paths, "row_counts": summary}


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Gold Layer Load")
    parser.add_argument("--register-bq", action="store_true",
                        help="Upload Gold tables to BigQuery")
    args = parser.parse_args()

    run_gold_load(register_bq=args.register_bq)
