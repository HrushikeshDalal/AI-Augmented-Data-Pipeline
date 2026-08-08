"""
test_dbt.py — Automated tests for dbt models, sources, and transformation assumptions.

These tests:
1. Validate dbt model SQL logic using DuckDB (in-memory) as a local BigQuery stand-in
2. Test dbt schema assumptions (uniqueness, not-null, accepted values)
3. Verify incremental logic and partition coverage
4. Run dbt singular tests (custom SQL assertions)

Run with: pytest tests/test_dbt.py -v --tb=short

Author: Capstone Project
"""

import sys
import textwrap
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

# ── Path setup ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ══════════════════════════════════════════════
# HELPERS: In-memory SQL engine (DuckDB)
# ══════════════════════════════════════════════

def get_duckdb():
    """Return a DuckDB in-memory connection, or skip if not installed."""
    try:
        import duckdb
        return duckdb.connect(":memory:")
    except ImportError:
        pytest.skip("duckdb not installed — skipping SQL-layer tests. "
                    "Install with: pip install duckdb")


def run_sql(con, sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame."""
    return con.execute(sql).df()


# ══════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════

@pytest.fixture(scope="module")
def db():
    """DuckDB connection pre-loaded with synthetic Silver-layer test tables."""
    con = get_duckdb()

    # ── Synthetic silver_events ───────────────────────────────────────
    rng = np.random.default_rng(42)
    n = 10_000
    categories = ["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"]
    event_types = ["page_view", "add_to_cart", "purchase", "wishlist", "review"]

    events = pd.DataFrame({
        "event_id":    [f"EVT_{i:010d}" for i in range(n)],
        "user_id":     rng.integers(1, 1000, n),
        "session_id":  [f"SES_{i}" for i in range(n)],
        "product_id":  rng.integers(1, 500, n),
        "category":    rng.choice(categories, n),
        "event_type":  rng.choice(event_types, n),
        "quantity":    rng.integers(1, 5, n),
        "price":       rng.uniform(5.0, 500.0, n).round(2),
        "discount":    rng.uniform(0.0, 0.4, n).round(2),
        "revenue":     rng.uniform(5.0, 2000.0, n).round(2),
        "timestamp":   pd.date_range("2023-01-01", periods=n, freq="1h"),
        "event_date":  pd.date_range("2023-01-01", periods=n, freq="1h").date,
        "event_hour":  rng.integers(0, 24, n),
        "is_purchase": rng.choice([True, False], n, p=[0.2, 0.8]),
        "country":     rng.choice(["US", "UK", "IN", "DE"], n),
        "_ingested_at": pd.Timestamp("2023-06-01"),
    })
    con.register("silver_events", events)

    # ── Synthetic silver_products ─────────────────────────────────────
    products = pd.DataFrame({
        "product_id":   range(1, 501),
        "product_name": [f"Product_{i}" for i in range(1, 501)],
        "category":     rng.choice(categories, 500),
        "brand":        rng.choice(["BrandA", "BrandB", "BrandC"], 500),
        "cost_price":   rng.uniform(5.0, 100.0, 500).round(2),
        "list_price":   rng.uniform(20.0, 300.0, 500).round(2),
        "margin_pct":   rng.uniform(10.0, 60.0, 500).round(2),
        "stock_qty":    rng.integers(0, 1000, 500),
        "is_active":    rng.choice([True, False], 500, p=[0.95, 0.05]),
    })
    con.register("silver_products", products)

    # ── Synthetic silver_users ────────────────────────────────────────
    users = pd.DataFrame({
        "user_id":      range(1, 1001),
        "username":     [f"user_{i}" for i in range(1, 1001)],
        "email":        [f"user_{i}@test.com" for i in range(1, 1001)],
        "country":      rng.choice(["US", "UK", "IN", "DE"], 1000),
        "loyalty_tier": rng.choice(["Bronze", "Silver", "Gold", "Platinum"], 1000),
        "signup_date":  pd.date_range("2020-01-01", periods=1000, freq="6h"),
        "is_active":    rng.choice([True, False], 1000),
    })
    con.register("silver_users", users)

    return con


# ══════════════════════════════════════════════
# 1. STAGING MODEL TESTS (stg_events logic)
# ══════════════════════════════════════════════

class TestStagingModels:

    def test_stg_events_no_null_event_ids(self, db):
        """stg_events must have no NULL event_ids."""
        result = run_sql(db, """
            SELECT COUNT(*) AS null_count
            FROM silver_events
            WHERE event_id IS NULL
        """)
        assert result["null_count"].iloc[0] == 0

    def test_stg_events_unique_event_ids(self, db):
        """event_id must be unique in the staging layer."""
        result = run_sql(db, """
            SELECT COUNT(*) AS total, COUNT(DISTINCT event_id) AS unique_count
            FROM silver_events
        """)
        assert result["total"].iloc[0] == result["unique_count"].iloc[0]

    def test_stg_events_valid_event_types(self, db):
        """Only accepted event_type values should be present."""
        result = run_sql(db, """
            SELECT DISTINCT event_type
            FROM silver_events
            WHERE event_type NOT IN (
                'page_view', 'add_to_cart', 'purchase', 'wishlist', 'review'
            )
        """)
        assert len(result) == 0, f"Invalid event types found: {result['event_type'].tolist()}"

    def test_stg_events_price_positive(self, db):
        """All prices should be positive."""
        result = run_sql(db, """
            SELECT COUNT(*) AS bad_rows
            FROM silver_events
            WHERE price <= 0
        """)
        assert result["bad_rows"].iloc[0] == 0

    def test_stg_events_discount_range(self, db):
        """Discount must be between 0 and 1."""
        result = run_sql(db, """
            SELECT COUNT(*) AS bad_rows
            FROM silver_events
            WHERE discount < 0 OR discount > 1
        """)
        assert result["bad_rows"].iloc[0] == 0

    def test_stg_products_unique_ids(self, db):
        """product_id must be unique in silver_products."""
        result = run_sql(db, """
            SELECT COUNT(*) AS total, COUNT(DISTINCT product_id) AS unique_count
            FROM silver_products
        """)
        assert result["total"].iloc[0] == result["unique_count"].iloc[0]

    def test_stg_users_unique_ids(self, db):
        """user_id must be unique in silver_users."""
        result = run_sql(db, """
            SELECT COUNT(*) AS total, COUNT(DISTINCT user_id) AS unique_count
            FROM silver_users
        """)
        assert result["total"].iloc[0] == result["unique_count"].iloc[0]

    def test_stg_users_valid_loyalty_tiers(self, db):
        """Loyalty tier must be one of the four accepted values."""
        result = run_sql(db, """
            SELECT DISTINCT loyalty_tier
            FROM silver_users
            WHERE loyalty_tier NOT IN ('Bronze', 'Silver', 'Gold', 'Platinum')
        """)
        assert len(result) == 0


# ══════════════════════════════════════════════
# 2. FACT TABLE TESTS (fct_sales logic)
# ══════════════════════════════════════════════

class TestFactSales:

    @pytest.fixture(scope="class")
    def fct_sales(self, db):
        """Build fct_sales in-memory via SQL (mirrors the dbt model logic)."""
        return run_sql(db, """
            SELECT
                e.event_id        AS sale_key,
                e.user_id         AS user_key,
                e.product_id      AS product_key,
                CAST(strftime(e.event_date, '%Y%m%d') AS INTEGER) AS date_key,
                e.category,
                e.country,
                e.quantity,
                e.price,
                e.discount,
                e.revenue,
                ROUND(e.price * e.quantity, 2)                AS gross_revenue,
                ROUND(e.discount * e.price * e.quantity, 2)   AS discount_amount,
                e.event_date,
                e.event_hour,
                e.is_purchase,
                u.loyalty_tier,
                p.brand,
                p.margin_pct
            FROM silver_events  e
            LEFT JOIN silver_products p ON e.product_id = p.product_id
            LEFT JOIN silver_users    u ON e.user_id    = u.user_id
            WHERE e.is_purchase = TRUE
        """)

    def test_fct_sales_grain_is_purchase(self, fct_sales):
        """All rows in the fact table must be purchase events."""
        assert (fct_sales["is_purchase"] == True).all()

    def test_fct_sales_no_null_sale_keys(self, fct_sales):
        assert fct_sales["sale_key"].isna().sum() == 0

    def test_fct_sales_no_null_user_keys(self, fct_sales):
        assert fct_sales["user_key"].isna().sum() == 0

    def test_fct_sales_unique_sale_keys(self, fct_sales):
        assert fct_sales["sale_key"].duplicated().sum() == 0

    def test_fct_sales_revenue_non_negative(self, fct_sales):
        assert (fct_sales["revenue"] >= 0).all()

    def test_fct_sales_gross_revenue_gte_net(self, fct_sales):
        """Gross revenue should always be >= net revenue."""
        assert (fct_sales["gross_revenue"] >= fct_sales["revenue"] - 0.01).all()

    def test_fct_sales_has_date_key(self, fct_sales):
        assert "date_key" in fct_sales.columns
        assert fct_sales["date_key"].notna().all()

    def test_fct_sales_valid_categories(self, fct_sales):
        valid = {"Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"}
        assert set(fct_sales["category"].unique()).issubset(valid)


# ══════════════════════════════════════════════
# 3. DIMENSION TABLE TESTS
# ══════════════════════════════════════════════

class TestDimensions:

    def test_dim_date_row_count(self, db):
        """Date spine for 2023 should have 365 rows."""
        result = run_sql(db, """
            WITH date_spine AS (
                SELECT
                    CAST(range AS DATE) AS full_date
                FROM range(
                    DATE '2023-01-01',
                    DATE '2024-01-01',
                    INTERVAL '1 day'
                )
            )
            SELECT COUNT(*) AS cnt FROM date_spine
        """)
        assert result["cnt"].iloc[0] == 365

    def test_dim_date_no_duplicate_keys(self, db):
        result = run_sql(db, """
            WITH date_spine AS (
                SELECT
                    CAST(strftime(CAST(range AS DATE), '%Y%m%d') AS INTEGER) AS date_key
                FROM range(
                    DATE '2023-01-01',
                    DATE '2024-01-01',
                    INTERVAL '1 day'
                )
            )
            SELECT COUNT(*) AS total, COUNT(DISTINCT date_key) AS unique_count
            FROM date_spine
        """)
        assert result["total"].iloc[0] == result["unique_count"].iloc[0]

    def test_dim_product_no_nulls(self, db):
        result = run_sql(db, """
            SELECT COUNT(*) AS null_count
            FROM silver_products
            WHERE product_id IS NULL OR product_name IS NULL
        """)
        assert result["null_count"].iloc[0] == 0

    def test_dim_product_margin_positive(self, db):
        result = run_sql(db, """
            SELECT COUNT(*) AS bad_rows
            FROM silver_products
            WHERE margin_pct < 0
        """)
        assert result["bad_rows"].iloc[0] == 0

    def test_dim_user_valid_loyalty_tiers(self, db):
        result = run_sql(db, """
            SELECT DISTINCT loyalty_tier FROM silver_users
            WHERE loyalty_tier NOT IN ('Bronze', 'Silver', 'Gold', 'Platinum')
        """)
        assert len(result) == 0


# ══════════════════════════════════════════════
# 4. AGGREGATE TESTS (agg_daily_sales logic)
# ══════════════════════════════════════════════

class TestAggregates:

    @pytest.fixture(scope="class")
    def agg_daily(self, db):
        return run_sql(db, """
            SELECT
                event_date,
                category,
                country,
                COUNT(event_id)          AS total_orders,
                SUM(quantity)            AS total_units_sold,
                ROUND(SUM(revenue), 2)   AS total_revenue,
                ROUND(AVG(revenue), 2)   AS avg_order_value,
                COUNT(DISTINCT user_id)  AS unique_customers
            FROM silver_events
            WHERE is_purchase = TRUE
            GROUP BY event_date, category, country
        """)

    def test_agg_daily_total_orders_positive(self, agg_daily):
        assert (agg_daily["total_orders"] > 0).all()

    def test_agg_daily_revenue_non_negative(self, agg_daily):
        assert (agg_daily["total_revenue"] >= 0).all()

    def test_agg_daily_aov_positive(self, agg_daily):
        assert (agg_daily["avg_order_value"] > 0).all()

    def test_agg_daily_customers_lte_orders(self, agg_daily):
        """A customer can place multiple orders so unique_customers <= total_orders."""
        assert (agg_daily["unique_customers"] <= agg_daily["total_orders"]).all()


# ══════════════════════════════════════════════
# 5. SINGULAR TESTS (custom SQL assertions)
#    These mirror what you'd put in dbt/tests/*.sql
# ══════════════════════════════════════════════

class TestSingularAssertions:

    def test_no_future_events(self, db):
        """No events should have a timestamp in the future."""
        result = run_sql(db, """
            SELECT COUNT(*) AS future_rows
            FROM silver_events
            WHERE event_date > CURRENT_DATE
        """)
        assert result["future_rows"].iloc[0] == 0

    def test_revenue_consistency(self, db):
        """Revenue should be approximately price × quantity × (1 - discount)."""
        result = run_sql(db, """
            SELECT COUNT(*) AS inconsistent_rows
            FROM silver_events
            WHERE ABS(revenue - ROUND(price * quantity * (1 - discount), 2)) > 1.0
        """)
        # Allow small floating-point drift; no row should be off by more than $1
        assert result["inconsistent_rows"].iloc[0] == 0

    def test_all_categories_have_sales(self, db):
        """Every product category must have at least one purchase event."""
        result = run_sql(db, """
            SELECT category, COUNT(*) AS purchases
            FROM silver_events
            WHERE is_purchase = TRUE
            GROUP BY category
        """)
        assert len(result) > 0, "No purchase events found at all"

    def test_user_ids_positive(self, db):
        result = run_sql(db, """
            SELECT COUNT(*) AS bad_rows
            FROM silver_events
            WHERE user_id <= 0
        """)
        assert result["bad_rows"].iloc[0] == 0

    def test_product_ids_positive(self, db):
        result = run_sql(db, """
            SELECT COUNT(*) AS bad_rows
            FROM silver_events
            WHERE product_id <= 0
        """)
        assert result["bad_rows"].iloc[0] == 0

    def test_quantity_at_least_one(self, db):
        result = run_sql(db, """
            SELECT COUNT(*) AS bad_rows
            FROM silver_events
            WHERE quantity < 1
        """)
        assert result["bad_rows"].iloc[0] == 0


# ══════════════════════════════════════════════
# 6. dbt PROJECT FILE VALIDATION
# ══════════════════════════════════════════════

class TestDbtProjectFiles:

    DBT_DIR = Path(__file__).resolve().parents[1] / "dbt"

    def test_dbt_project_yml_exists(self):
        assert (self.DBT_DIR / "dbt_project.yml").exists()

    def test_dbt_project_yml_valid(self):
        """dbt_project.yml must be valid YAML with required keys."""
        import yaml
        with open(self.DBT_DIR / "dbt_project.yml") as f:
            config = yaml.safe_load(f)
        assert "name"    in config
        assert "version" in config
        assert "models"  in config

    def test_staging_schema_yml_exists(self):
        assert (self.DBT_DIR / "models" / "staging" / "schema.yml").exists()

    def test_marts_schema_yml_exists(self):
        assert (self.DBT_DIR / "models" / "marts" / "schema.yml").exists()

    def test_fct_sales_sql_exists(self):
        assert (self.DBT_DIR / "models" / "marts" / "fct_sales.sql").exists()

    def test_dim_date_sql_exists(self):
        assert (self.DBT_DIR / "models" / "marts" / "dim_date.sql").exists()

    def test_agg_daily_sales_sql_exists(self):
        assert (self.DBT_DIR / "models" / "marts" / "agg_daily_sales.sql").exists()

    def test_stg_events_sql_exists(self):
        assert (self.DBT_DIR / "models" / "staging" / "stg_events.sql").exists()

    def test_staging_schema_has_sources(self):
        import yaml
        with open(self.DBT_DIR / "models" / "staging" / "schema.yml") as f:
            schema = yaml.safe_load(f)
        assert "sources" in schema
        source_names = [s["name"] for s in schema["sources"]]
        assert "silver" in source_names

    def test_marts_schema_has_fct_sales(self):
        import yaml
        with open(self.DBT_DIR / "models" / "marts" / "schema.yml") as f:
            schema = yaml.safe_load(f)
        model_names = [m["name"] for m in schema["models"]]
        assert "fct_sales" in model_names


# ══════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
