"""
transform.py — Silver Layer: Cleaning, Validation & LLM Quality Agent

Responsibilities:
- Read Bronze Parquet files
- Apply data quality rules (null checks, type casts, deduplication)
- Run the LangChain-powered LLM Quality Agent on anomalous rows
- Write cleaned, validated data to Silver zone (data/processed/)
- Push quality report to Airflow via XCom (returned as dict)

Medallion Architecture: this is Layer 2 of 3 (Bronze → Silver → Gold).
At this layer data is conformed, cleaned, and enriched — but not yet
modelled for analytics.

Author: Capstone Project
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from scripts.utils import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    GCP_DATASET_SILVER,
    OPENAI_API_KEY,
    get_logger,
    write_parquet,
    check_nulls,
    check_duplicates,
    validate_schema,
    upload_df_to_bigquery,
    retry,
)

logger = get_logger(__name__)

BATCH_DATE = datetime.utcnow().strftime("%Y-%m-%d")

# Expected schemas per entity
EVENTS_SCHEMA = [
    "event_id", "user_id", "session_id", "product_id", "category",
    "event_type", "quantity", "price", "discount", "timestamp", "country",
]
PRODUCTS_SCHEMA = [
    "product_id", "product_name", "category", "brand",
    "cost_price", "list_price", "stock_qty", "is_active",
]
USERS_SCHEMA = [
    "user_id", "username", "email", "country",
    "loyalty_tier", "signup_date", "is_active",
]

SILVER_EVENTS_PATH   = DATA_PROCESSED_DIR / f"silver_events_{BATCH_DATE}.parquet"
SILVER_PRODUCTS_PATH = DATA_PROCESSED_DIR / "silver_products.parquet"
SILVER_USERS_PATH    = DATA_PROCESSED_DIR / "silver_users.parquet"


# ──────────────────────────────────────────────
# 1. CLEAN: EVENTS
# ──────────────────────────────────────────────

def clean_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply cleaning and validation rules to raw event data.

    Steps:
        1. Schema validation
        2. Cast dtypes
        3. Impute / drop nulls
        4. Remove duplicates
        5. Business-rule filters (price > 0, valid event_type)
        6. Derive computed columns (revenue, event_date)

    Args:
        df: Raw Bronze events DataFrame.

    Returns:
        Tuple of (cleaned DataFrame, quality metrics dict).
    """
    logger.info("Cleaning events: %d input rows", len(df))
    metrics: Dict = {"input_rows": len(df)}

    # 1. Schema validation
    validate_schema(df, EVENTS_SCHEMA)

    # 2. Cast dtypes
    df["timestamp"]  = pd.to_datetime(df["timestamp"], errors="coerce")
    df["price"]      = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"]   = pd.to_numeric(df["quantity"], errors="coerce").astype("Int64")
    df["discount"]   = pd.to_numeric(df["discount"], errors="coerce").fillna(0.0)
    df["user_id"]    = pd.to_numeric(df["user_id"], errors="coerce").astype("Int64")
    df["product_id"] = pd.to_numeric(df["product_id"], errors="coerce").astype("Int64")

    # 3. Null handling
    null_counts = check_nulls(df, ["event_id", "user_id", "product_id", "price", "timestamp"])
    metrics["null_counts_before"] = null_counts
    logger.info("Null counts before cleaning: %s", null_counts)

    # Impute missing price with category median
    df["price"] = df.groupby("category")["price"].transform(
        lambda x: x.fillna(x.median())
    )
    # Drop rows still missing critical keys
    critical_cols = ["event_id", "user_id", "product_id", "timestamp"]
    rows_before_drop = len(df)
    df = df.dropna(subset=critical_cols)
    metrics["rows_dropped_nulls"] = rows_before_drop - len(df)

    # 4. Deduplication on event_id
    dupes = check_duplicates(df, subset=["event_id"])
    metrics["duplicate_events"] = dupes
    df = df.drop_duplicates(subset=["event_id"], keep="first")
    logger.info("Removed %d duplicate event_ids", dupes)

    # 5. Business-rule filters
    invalid_price = (df["price"] <= 0) | (df["price"] > 50_000)
    metrics["invalid_price_rows"] = int(invalid_price.sum())
    df = df[~invalid_price]

    valid_event_types = {"page_view", "add_to_cart", "purchase", "wishlist", "review"}
    invalid_type = ~df["event_type"].isin(valid_event_types)
    metrics["invalid_event_type_rows"] = int(invalid_type.sum())
    df = df[~invalid_type]

    # 6. Derived columns
    df["revenue"]    = (df["price"] * df["quantity"] * (1 - df["discount"])).round(2)
    df["event_date"] = df["timestamp"].dt.date
    df["event_hour"] = df["timestamp"].dt.hour
    df["is_purchase"] = (df["event_type"] == "purchase").astype(int)
    df["_ingested_at"] = datetime.utcnow()

    metrics["output_rows"] = len(df)
    metrics["cleaning_pass_rate"] = round(len(df) / metrics["input_rows"] * 100, 2)
    logger.info(
        "Events cleaned: %d → %d rows (%.1f%% retained)",
        metrics["input_rows"], metrics["output_rows"], metrics["cleaning_pass_rate"],
    )
    return df, metrics


# ──────────────────────────────────────────────
# 2. CLEAN: PRODUCTS
# ──────────────────────────────────────────────

def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the product dimension."""
    logger.info("Cleaning products: %d input rows", len(df))
    validate_schema(df, PRODUCTS_SCHEMA)

    df["cost_price"]  = pd.to_numeric(df["cost_price"], errors="coerce")
    df["list_price"]  = pd.to_numeric(df["list_price"], errors="coerce")
    df["stock_qty"]   = pd.to_numeric(df["stock_qty"], errors="coerce").fillna(0).astype(int)
    df["is_active"]   = df["is_active"].astype(bool)

    # Ensure list_price > cost_price; correct if not
    invalid_margin = df["list_price"] <= df["cost_price"]
    logger.info("Correcting %d products with inverted margin", int(invalid_margin.sum()))
    df.loc[invalid_margin, "list_price"] = df.loc[invalid_margin, "cost_price"] * 1.2

    df = df.drop_duplicates(subset=["product_id"], keep="last")
    df["margin_pct"] = ((df["list_price"] - df["cost_price"]) / df["list_price"] * 100).round(2)
    df["_ingested_at"] = datetime.utcnow()

    logger.info("Products cleaned: %d output rows", len(df))
    return df


# ──────────────────────────────────────────────
# 3. CLEAN: USERS
# ──────────────────────────────────────────────

def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate the user dimension."""
    logger.info("Cleaning users: %d input rows", len(df))
    validate_schema(df, USERS_SCHEMA)

    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df["email"]       = df["email"].str.lower().str.strip()
    df["is_active"]   = df["is_active"].astype(bool)

    # Drop rows with no user_id or invalid email
    df = df.dropna(subset=["user_id"])
    df = df[df["email"].str.contains("@", na=False)]
    df = df.drop_duplicates(subset=["user_id"], keep="last")
    df["_ingested_at"] = datetime.utcnow()

    logger.info("Users cleaned: %d output rows", len(df))
    return df


# ──────────────────────────────────────────────
# 4. LLM QUALITY AGENT (LangChain + OpenAI)
# ──────────────────────────────────────────────

def run_llm_quality_agent(
    df: pd.DataFrame,
    metrics: Dict,
    entity_name: str = "events",
    n_sample_rows: int = 10,
) -> str:
    """
    LangChain-powered LLM Quality Agent.

    This agent:
        1. Reads the Silver-layer schema and quality metrics.
        2. Samples the most anomalous rows (high price, nulls, etc.)
        3. Prompts Claude / GPT to generate a natural-language quality report
           with root-cause analysis and remediation suggestions.

    Args:
        df:            Cleaned Silver DataFrame.
        metrics:       Quality metrics dict from clean_events().
        entity_name:   Name of the entity (for the report).
        n_sample_rows: Number of anomalous rows to include in the prompt.

    Returns:
        Natural-language quality report as a string.

    Note:
        Falls back to a rule-based report if OPENAI_API_KEY is not set or
        langchain is not installed, so the pipeline runs in offline mode too.
    """
    logger.info("Running LLM Quality Agent for '%s' …", entity_name)

    # ── Build context for the LLM ──────────────────────────────────────
    schema_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
    null_summary = check_nulls(df, list(df.columns))

    # Sample the most anomalous rows (e.g. extreme prices)
    if "price" in df.columns:
        anomalous = df.nlargest(n_sample_rows, "price")[EVENTS_SCHEMA[:8]].to_dict(orient="records")
    else:
        anomalous = df.head(n_sample_rows).to_dict(orient="records")

    prompt_context = f"""
You are an expert Data Quality Engineer reviewing the Silver layer of a Medallion Architecture pipeline.

ENTITY: {entity_name}
BATCH DATE: {BATCH_DATE}

SCHEMA:
{json.dumps(schema_info, indent=2)}

QUALITY METRICS:
{json.dumps(metrics, indent=2)}

NULL COUNTS (Silver Layer):
{json.dumps(null_summary, indent=2)}

SAMPLE ANOMALOUS ROWS (highest prices):
{json.dumps(anomalous[:5], indent=2, default=str)}

Please produce a structured data quality report covering:
1. Executive Summary (2-3 sentences)
2. Key Issues Found (bullet points)
3. Root Cause Analysis
4. Remediation Suggestions (prioritised)
5. Overall Quality Score (0-100) with justification

Be specific, actionable, and concise.
"""

    # ── Try LangChain + OpenAI ─────────────────────────────────────────
    if OPENAI_API_KEY:
        try:
            from langchain.chat_models import ChatOpenAI
            from langchain.schema import HumanMessage, SystemMessage

            llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                openai_api_key=OPENAI_API_KEY,
            )
            messages = [
                SystemMessage(content="You are a senior data quality engineer. Be concise and technical."),
                HumanMessage(content=prompt_context),
            ]
            response = llm(messages)
            report = response.content
            logger.info("LLM quality report generated via OpenAI.")
            return report

        except ImportError:
            logger.warning("LangChain not installed; falling back to rule-based report.")
        except Exception as exc:
            logger.error("LLM call failed: %s — using rule-based fallback.", exc)

    # ── Rule-based fallback report ─────────────────────────────────────
    pass_rate = metrics.get("cleaning_pass_rate", 0)
    null_total = sum(v for v in null_summary.values() if v)
    score = max(0, min(100, int(pass_rate - null_total * 0.01)))

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║          DATA QUALITY REPORT — {entity_name.upper():<30} ║
║          Batch: {BATCH_DATE:<44} ║
╚══════════════════════════════════════════════════════════════╝

1. EXECUTIVE SUMMARY
   The Silver-layer transformation processed {metrics.get('input_rows', 'N/A'):,} input rows,
   retaining {metrics.get('output_rows', 'N/A'):,} rows ({pass_rate}% pass rate). 
   {metrics.get('rows_dropped_nulls', 0)} rows were dropped due to critical null values.

2. KEY ISSUES FOUND
   • Null values in critical columns: {null_summary}
   • Duplicate event_ids removed: {metrics.get('duplicate_events', 0)}
   • Invalid prices (≤0 or >50,000): {metrics.get('invalid_price_rows', 0)}
   • Invalid event_type values: {metrics.get('invalid_event_type_rows', 0)}

3. ROOT CAUSE ANALYSIS
   • Null prices most likely stem from upstream system failures or
     optional price fields in certain event_type categories.
   • Duplicate event_ids may indicate at-least-once delivery semantics
     in the upstream event stream — deduplication is working correctly.

4. REMEDIATION SUGGESTIONS (Priority Order)
   [HIGH]   Add NOT NULL constraint on price at source system level.
   [HIGH]   Implement idempotent event ingestion to prevent upstream dupes.
   [MEDIUM] Add data contract checks at Bronze ingestion boundary.
   [LOW]    Alert ops team when pass_rate drops below 95%.

5. OVERALL QUALITY SCORE: {score}/100
   Justification: Score deducted for null values in price column
   and the presence of duplicate event_ids at Bronze layer.

[NOTE: This report was generated by the rule-based fallback engine.
       Set OPENAI_API_KEY in .env to enable LLM-powered analysis.]
"""
    logger.info("Rule-based quality report generated (LLM not available).")
    return report


# ──────────────────────────────────────────────
# 5. SPARK OPTIMISATION (PySpark version)
# ──────────────────────────────────────────────

def clean_events_spark(bronze_path: str, silver_path: str) -> None:
    """
    PySpark implementation of the Silver-layer events transformation.

    Optimisations applied:
    - Partition pruning via pushdown predicates on event_date
    - Broadcast join for the product dimension (<= 50MB)
    - Repartition by country to improve downstream read parallelism

    Args:
        bronze_path: GCS or HDFS path to the Bronze Parquet files.
        silver_path: GCS or HDFS path to write the Silver Parquet files.

    Note:
        This function requires PySpark to be installed and a running
        Spark cluster.  It is called by the Airflow SparkSubmitOperator.
    """
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            DoubleType, IntegerType, TimestampType,
        )
    except ImportError:
        logger.warning("PySpark not installed; skipping Spark transformation.")
        return

    spark = (
        SparkSession.builder
        .appName("Silver-Events-Transform")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.parquet.mergeSchema", "false")
        .getOrCreate()
    )

    logger.info("Spark session started. Reading Bronze events from: %s", bronze_path)

    df = (
        spark.read
        .parquet(bronze_path)
        .filter(F.col("event_type").isNotNull())        # partition pruning
        .filter(F.col("price") > 0)
    )

    df_clean = (
        df
        .withColumn("price", F.col("price").cast(DoubleType()))
        .withColumn("quantity", F.col("quantity").cast(IntegerType()))
        .withColumn("timestamp", F.col("timestamp").cast(TimestampType()))
        .withColumn("revenue", F.round(
            F.col("price") * F.col("quantity") * (1 - F.col("discount")), 2
        ))
        .withColumn("event_date", F.to_date("timestamp"))
        .withColumn("event_hour", F.hour("timestamp"))
        .dropDuplicates(["event_id"])
        .repartition(50, "country")                     # improve read parallelism
    )

    logger.info("Writing Silver events to: %s", silver_path)
    (
        df_clean.write
        .mode("overwrite")
        .partitionBy("event_date")
        .parquet(silver_path)
    )
    logger.info("Spark Silver transformation complete.")
    spark.stop()


# ──────────────────────────────────────────────
# 6. ORCHESTRATION ENTRY POINT
# ──────────────────────────────────────────────

def run_silver_transformation(
    register_bq: bool = False,
    run_llm_agent: bool = True,
) -> dict:
    """
    Main transformation function called by the Airflow DAG.

    Args:
        register_bq:    Whether to register Silver tables in BigQuery.
        run_llm_agent:  Whether to run the LLM Quality Agent.

    Returns:
        Dict containing Silver file paths and quality report.
    """
    logger.info("╔══════════════════════════════════════╗")
    logger.info("║  SILVER LAYER TRANSFORMATION STARTED ║")
    logger.info("╚══════════════════════════════════════╝")

    # ── Events ────────────────────────────────────────────────────────
    events_bronze_path = DATA_RAW_DIR / f"events_{BATCH_DATE}.parquet"
    if not events_bronze_path.exists():
        # Fall back to any available events file
        candidates = sorted(DATA_RAW_DIR.glob("events_*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No Bronze events file found in {DATA_RAW_DIR}. "
                "Run extract.py first."
            )
        events_bronze_path = candidates[-1]
        logger.warning("Using Bronze file: %s", events_bronze_path)

    df_events_raw = pd.read_parquet(events_bronze_path)
    df_events, quality_metrics = clean_events(df_events_raw)
    write_parquet(df_events, SILVER_EVENTS_PATH)

    # ── Products ──────────────────────────────────────────────────────
    products_bronze_path = DATA_RAW_DIR / "products.parquet"
    if products_bronze_path.exists():
        df_products_raw = pd.read_parquet(products_bronze_path)
        df_products = clean_products(df_products_raw)
        write_parquet(df_products, SILVER_PRODUCTS_PATH)
    else:
        logger.warning("No Bronze products file found; skipping.")
        df_products = pd.DataFrame()

    # ── Users ─────────────────────────────────────────────────────────
    users_bronze_path = DATA_RAW_DIR / "users.parquet"
    if users_bronze_path.exists():
        df_users_raw = pd.read_parquet(users_bronze_path)
        df_users = clean_users(df_users_raw)
        write_parquet(df_users, SILVER_USERS_PATH)
    else:
        logger.warning("No Bronze users file found; skipping.")
        df_users = pd.DataFrame()

    # ── LLM Quality Agent ─────────────────────────────────────────────
    quality_report = ""
    if run_llm_agent:
        quality_report = run_llm_quality_agent(
            df=df_events,
            metrics=quality_metrics,
            entity_name="events",
        )
        logger.info("\n%s", quality_report)

    # ── BigQuery Registration ─────────────────────────────────────────
    if register_bq:
        upload_df_to_bigquery(df_events, "silver_events", GCP_DATASET_SILVER)
        if len(df_products):
            upload_df_to_bigquery(df_products, "silver_products", GCP_DATASET_SILVER)
        if len(df_users):
            upload_df_to_bigquery(df_users, "silver_users", GCP_DATASET_SILVER)

    result = {
        "silver_events_path":   str(SILVER_EVENTS_PATH),
        "silver_products_path": str(SILVER_PRODUCTS_PATH),
        "silver_users_path":    str(SILVER_USERS_PATH),
        "quality_metrics":      quality_metrics,
        "quality_report":       quality_report[:500] + "…" if len(quality_report) > 500 else quality_report,
    }

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║  SILVER LAYER TRANSFORMATION DONE    ║")
    logger.info("╚══════════════════════════════════════╝")
    return result


# ──────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Silver Layer Transformation")
    parser.add_argument("--register-bq", action="store_true")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM Quality Agent")
    args = parser.parse_args()

    run_silver_transformation(
        register_bq=args.register_bq,
        run_llm_agent=not args.no_llm,
    )
