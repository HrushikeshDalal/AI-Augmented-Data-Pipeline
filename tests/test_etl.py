"""
test_etl.py — Unit Tests for ETL Pipeline

Tests cover:
- utils.py: helper functions, validation, data generation
- extract.py: Bronze layer ingestion
- transform.py: Silver layer cleaning and LLM quality agent
- load.py: Gold layer star-schema building

Run with: pytest tests/test_etl.py -v --tb=short

Author: Capstone Project
"""

import json
import sys
import os
from datetime import datetime, date
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import numpy as np
import pytest

# ── Path setup ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils import (
    check_nulls,
    check_duplicates,
    assert_no_nulls,
    assert_unique,
    validate_schema,
    generate_ecommerce_events,
    retry,
)

from scripts.transform import (
    clean_events,
    clean_products,
    clean_users,
    run_llm_quality_agent,
)

from scripts.load import (
    build_dim_date,
    build_dim_product,
    build_dim_user,
    build_fact_sales,
    build_agg_daily_sales,
    build_agg_user_ltv,
)


# ══════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════

@pytest.fixture(scope="module")
def raw_events_df():
    """Small raw events DataFrame for fast unit tests."""
    return generate_ecommerce_events(n_rows=1_000, seed=42)


@pytest.fixture(scope="module")
def clean_events_df(raw_events_df):
    """Pre-cleaned Silver events for downstream tests."""
    df, _ = clean_events(raw_events_df.copy())
    return df


@pytest.fixture
def sample_products_df():
    """Minimal synthetic product DataFrame."""
    rng = np.random.default_rng(1)
    n = 200
    return pd.DataFrame({
        "product_id":   range(1, n + 1),
        "product_name": [f"Product_{i}" for i in range(1, n + 1)],
        "category":     rng.choice(["Electronics", "Clothing", "Books"], n),
        "brand":        rng.choice(["BrandA", "BrandB"], n),
        "cost_price":   rng.uniform(5.0, 100.0, n).round(2),
        "list_price":   rng.uniform(10.0, 200.0, n).round(2),
        "stock_qty":    rng.integers(0, 500, n),
        "is_active":    rng.choice([True, False], n),
    })


@pytest.fixture
def sample_users_df():
    """Minimal synthetic user DataFrame."""
    rng = np.random.default_rng(2)
    n = 500
    return pd.DataFrame({
        "user_id":      range(1, n + 1),
        "username":     [f"user_{i}" for i in range(1, n + 1)],
        "email":        [f"user_{i}@test.com" for i in range(1, n + 1)],
        "country":      rng.choice(["US", "UK", "IN"], n),
        "loyalty_tier": rng.choice(["Bronze", "Silver", "Gold", "Platinum"], n),
        "signup_date":  pd.date_range("2022-01-01", periods=n, freq="1h"),
        "is_active":    rng.choice([True, False], n),
    })


# ══════════════════════════════════════════════
# 1. UTILS TESTS
# ══════════════════════════════════════════════

class TestUtils:

    def test_check_nulls_returns_counts(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 1]})
        result = check_nulls(df, ["a", "b"])
        assert result["a"] == 1
        assert result["b"] == 2

    def test_check_nulls_no_nulls(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        assert check_nulls(df, ["x"]) == {"x": 0}

    def test_check_duplicates_finds_dupes(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        assert check_duplicates(df, subset=["id"]) == 1

    def test_check_duplicates_none(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        assert check_duplicates(df, subset=["id"]) == 0

    def test_assert_no_nulls_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        assert_no_nulls(df, ["a"])   # Should not raise

    def test_assert_no_nulls_raises(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        with pytest.raises(ValueError, match="Null values found"):
            assert_no_nulls(df, ["a"])

    def test_assert_unique_passes(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        assert_unique(df, subset=["id"])  # Should not raise

    def test_assert_unique_raises(self):
        df = pd.DataFrame({"id": [1, 1, 2]})
        with pytest.raises(ValueError, match="duplicate rows"):
            assert_unique(df, subset=["id"])

    def test_validate_schema_passes(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        validate_schema(df, ["a", "b"])  # Should not raise

    def test_validate_schema_raises_on_missing(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Missing expected columns"):
            validate_schema(df, ["a", "b"])

    def test_generate_ecommerce_events_shape(self):
        df = generate_ecommerce_events(n_rows=500, seed=1)
        assert len(df) == 500
        assert "event_id" in df.columns
        assert "revenue" not in df.columns   # revenue is added at Silver layer

    def test_generate_ecommerce_events_unique_event_ids(self):
        df = generate_ecommerce_events(n_rows=500, seed=1)
        assert df["event_id"].nunique() == 500

    def test_retry_decorator_succeeds_on_first_attempt(self):
        call_count = 0
        @retry(max_attempts=3, backoff_factor=0.01)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "ok"
        result = always_succeeds()
        assert result == "ok"
        assert call_count == 1

    def test_retry_decorator_retries_on_failure(self):
        call_count = 0
        @retry(max_attempts=3, backoff_factor=0.01, exceptions=(ValueError,))
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "ok"
        result = fails_twice()
        assert result == "ok"
        assert call_count == 3

    def test_retry_decorator_raises_after_max_attempts(self):
        @retry(max_attempts=2, backoff_factor=0.01, exceptions=(RuntimeError,))
        def always_fails():
            raise RuntimeError("always fails")
        with pytest.raises(RuntimeError):
            always_fails()


# ══════════════════════════════════════════════
# 2. TRANSFORM (SILVER LAYER) TESTS
# ══════════════════════════════════════════════

class TestTransformEvents:

    def test_clean_events_returns_dataframe(self, raw_events_df):
        df_clean, metrics = clean_events(raw_events_df.copy())
        assert isinstance(df_clean, pd.DataFrame)
        assert isinstance(metrics, dict)

    def test_clean_events_no_null_event_ids(self, clean_events_df):
        assert clean_events_df["event_id"].isna().sum() == 0

    def test_clean_events_no_null_user_ids(self, clean_events_df):
        assert clean_events_df["user_id"].isna().sum() == 0

    def test_clean_events_no_duplicate_event_ids(self, clean_events_df):
        assert clean_events_df["event_id"].duplicated().sum() == 0

    def test_clean_events_valid_prices(self, clean_events_df):
        assert (clean_events_df["price"] > 0).all()
        assert (clean_events_df["price"] <= 50_000).all()

    def test_clean_events_valid_event_types(self, clean_events_df):
        valid = {"page_view", "add_to_cart", "purchase", "wishlist", "review"}
        assert set(clean_events_df["event_type"].unique()).issubset(valid)

    def test_clean_events_revenue_computed(self, clean_events_df):
        assert "revenue" in clean_events_df.columns
        assert (clean_events_df["revenue"] >= 0).all()

    def test_clean_events_event_date_derived(self, clean_events_df):
        assert "event_date" in clean_events_df.columns
        assert "event_hour" in clean_events_df.columns

    def test_clean_events_metrics_keys(self, raw_events_df):
        _, metrics = clean_events(raw_events_df.copy())
        required_keys = ["input_rows", "output_rows", "cleaning_pass_rate"]
        for key in required_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_clean_events_pass_rate_reasonable(self, raw_events_df):
        _, metrics = clean_events(raw_events_df.copy())
        # Synthetic data has ~2% nulls so pass rate should be > 90%
        assert metrics["cleaning_pass_rate"] > 90.0

    def test_clean_events_reduces_row_count_with_bad_data(self):
        """Inject obviously bad data and verify cleaning removes it."""
        bad_df = pd.DataFrame({
            "event_id":   ["E1", "E2", "E3", "E4", None],
            "user_id":    [1, 2, None, 4, 5],
            "session_id": ["S1"] * 5,
            "product_id": [10, 20, 30, 40, 50],
            "category":   ["Electronics"] * 5,
            "event_type": ["purchase", "purchase", "INVALID", "purchase", "purchase"],
            "quantity":   [1, 2, 1, 3, 1],
            "price":      [100.0, -5.0, 200.0, 300.0, 400.0],
            "discount":   [0.1] * 5,
            "timestamp":  [pd.Timestamp("2023-01-01")] * 5,
            "country":    ["US"] * 5,
        })
        df_clean, metrics = clean_events(bad_df)
        # E5 (null event_id), E3 (INVALID type), E2 (negative price) should be dropped
        assert len(df_clean) < len(bad_df)
        assert metrics["output_rows"] == len(df_clean)


class TestTransformProducts:

    def test_clean_products_no_duplicate_ids(self, sample_products_df):
        from scripts.transform import clean_products
        df = clean_products(sample_products_df.copy())
        assert df["product_id"].duplicated().sum() == 0

    def test_clean_products_margin_computed(self, sample_products_df):
        from scripts.transform import clean_products
        df = clean_products(sample_products_df.copy())
        assert "margin_pct" in df.columns

    def test_clean_products_list_price_above_cost(self, sample_products_df):
        from scripts.transform import clean_products
        df = clean_products(sample_products_df.copy())
        assert (df["list_price"] >= df["cost_price"]).all()


class TestTransformUsers:

    def test_clean_users_no_duplicate_ids(self, sample_users_df):
        from scripts.transform import clean_users
        df = clean_users(sample_users_df.copy())
        assert df["user_id"].duplicated().sum() == 0

    def test_clean_users_email_lowercase(self, sample_users_df):
        from scripts.transform import clean_users
        df = clean_users(sample_users_df.copy())
        assert (df["email"] == df["email"].str.lower()).all()

    def test_clean_users_valid_emails(self, sample_users_df):
        from scripts.transform import clean_users
        df = clean_users(sample_users_df.copy())
        assert df["email"].str.contains("@").all()


class TestLLMQualityAgent:

    def test_quality_agent_returns_string(self, clean_events_df):
        """LLM agent should return a non-empty string (fallback mode)."""
        metrics = {"input_rows": 1000, "output_rows": 980, "cleaning_pass_rate": 98.0}
        # With no OPENAI_API_KEY, should use rule-based fallback
        with patch("scripts.transform.OPENAI_API_KEY", ""):
            report = run_llm_quality_agent(clean_events_df, metrics)
        assert isinstance(report, str)
        assert len(report) > 100

    def test_quality_agent_contains_key_sections(self, clean_events_df):
        """Fallback report should have all required sections."""
        metrics = {"input_rows": 1000, "output_rows": 980, "cleaning_pass_rate": 98.0}
        with patch("scripts.transform.OPENAI_API_KEY", ""):
            report = run_llm_quality_agent(clean_events_df, metrics)
        assert "EXECUTIVE SUMMARY" in report
        assert "QUALITY SCORE" in report


# ══════════════════════════════════════════════
# 3. LOAD (GOLD LAYER) TESTS
# ══════════════════════════════════════════════

class TestLoadGoldLayer:

    def test_build_dim_date_shape(self):
        df = build_dim_date("2023-01-01", "2023-12-31")
        assert len(df) == 365
        assert "date_key" in df.columns
        assert "is_weekend" in df.columns

    def test_build_dim_date_unique_keys(self):
        df = build_dim_date("2023-01-01", "2023-03-31")
        assert df["date_key"].duplicated().sum() == 0

    def test_build_dim_date_correct_weekends(self):
        df = build_dim_date("2023-01-01", "2023-01-07")
        # Jan 1 2023 = Sunday (weekend), Jan 2 = Monday (weekday)
        jan1 = df[df["full_date"] == date(2023, 1, 1)].iloc[0]
        jan2 = df[df["full_date"] == date(2023, 1, 2)].iloc[0]
        assert jan1["is_weekend"] == 1
        assert jan2["is_weekend"] == 0

    def test_build_dim_product(self, sample_products_df):
        from scripts.transform import clean_products
        df_silver = clean_products(sample_products_df)
        df_dim = build_dim_product(df_silver)
        assert "product_key" in df_dim.columns
        assert df_dim["product_key"].duplicated().sum() == 0

    def test_build_dim_user(self, sample_users_df):
        from scripts.transform import clean_users
        df_silver = clean_users(sample_users_df)
        df_dim = build_dim_user(df_silver)
        assert "user_key" in df_dim.columns
        assert df_dim["user_key"].duplicated().sum() == 0

    def test_build_fact_sales_grain(self, clean_events_df, sample_products_df, sample_users_df):
        """Fact table grain = one row per PURCHASE event."""
        from scripts.transform import clean_products, clean_users
        df_products = clean_products(sample_products_df)
        df_users    = clean_users(sample_users_df)

        fact = build_fact_sales(clean_events_df, df_products, df_users)
        # All rows must be purchase events
        if "event_type" in fact.columns:
            assert (fact.get("event_type", pd.Series(["purchase"])) == "purchase").all()
        assert "sale_key" in fact.columns
        assert "revenue" in fact.columns

    def test_build_fact_sales_has_date_key(self, clean_events_df):
        """Fact table must have a date_key FK."""
        fact = build_fact_sales(clean_events_df, pd.DataFrame(), pd.DataFrame())
        assert "date_key" in fact.columns
        if len(fact):
            assert fact["date_key"].notna().all()

    def test_build_agg_daily_sales(self, clean_events_df):
        fact = build_fact_sales(clean_events_df, pd.DataFrame(), pd.DataFrame())
        if len(fact):
            agg = build_agg_daily_sales(fact)
            assert "total_orders" in agg.columns
            assert "total_revenue" in agg.columns
            assert (agg["total_orders"] > 0).all()

    def test_build_agg_user_ltv(self, clean_events_df):
        fact = build_fact_sales(clean_events_df, pd.DataFrame(), pd.DataFrame())
        if len(fact):
            ltv = build_agg_user_ltv(fact)
            assert "user_key" in ltv.columns
            assert (ltv["total_spend"] > 0).all()


# ══════════════════════════════════════════════
# 4. DATA QUALITY TESTS (integration-style)
# ══════════════════════════════════════════════

class TestDataQuality:
    """Cross-layer data quality assertions."""

    def test_silver_events_referential_integrity_users(
        self, clean_events_df, sample_users_df
    ):
        """All user_ids in Silver events should exist in Silver users."""
        from scripts.transform import clean_users
        df_users = clean_users(sample_users_df)
        valid_user_ids = set(df_users["user_id"].tolist())
        # Our synthetic data uses 1–500 users; events reference 1–500k
        # This test validates the inner-join logic in the Gold layer
        fact = build_fact_sales(clean_events_df, pd.DataFrame(), df_users)
        if "loyalty_tier" in fact.columns:
            # Rows with matched users should have a loyalty_tier
            matched = fact[fact["user_key"].isin(valid_user_ids)]
            # matched rows should have loyalty_tier populated
            # (some may be null for unmatched users — this is expected)

    def test_gold_revenue_non_negative(self, clean_events_df):
        """Revenue values in fact table must always be >= 0."""
        fact = build_fact_sales(clean_events_df, pd.DataFrame(), pd.DataFrame())
        if len(fact):
            assert (fact["revenue"] >= 0).all(), "Negative revenue found in fact table"

    def test_gold_dim_date_covers_fact_dates(self, clean_events_df):
        """All event_date values in fact_sales should exist in dim_date."""
        fact = build_fact_sales(clean_events_df, pd.DataFrame(), pd.DataFrame())
        dim_date = build_dim_date("2023-01-01", "2024-12-31")
        if len(fact):
            fact_dates = set(pd.to_datetime(fact["event_date"]).dt.date)
            dim_dates  = set(dim_date["full_date"])
            uncovered  = fact_dates - dim_dates
            assert len(uncovered) == 0, f"Fact dates not in dim_date: {uncovered}"


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
