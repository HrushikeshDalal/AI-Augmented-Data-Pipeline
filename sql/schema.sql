-- schema.sql
-- BigQuery Schema: Capstone E-Commerce Data Warehouse
-- Medallion Architecture: Bronze → Silver → Gold
--
-- Usage: Run via `bq query --use_legacy_sql=false < schema.sql`
--        or apply via Terraform / dbt seeds
--
-- Author: Capstone Project

-- ──────────────────────────────────────────────
-- BRONZE DATASET (raw landing zone)
-- ──────────────────────────────────────────────

CREATE SCHEMA IF NOT EXISTS `${project_id}.bronze`
OPTIONS (
  description = "Bronze layer: raw ingested data, never modified",
  location    = "US"
);

CREATE TABLE IF NOT EXISTS `${project_id}.bronze.raw_events` (
  event_id    STRING     NOT NULL OPTIONS(description = "Unique event identifier"),
  user_id     INT64               OPTIONS(description = "User who triggered the event"),
  session_id  STRING              OPTIONS(description = "Browser/app session"),
  product_id  INT64               OPTIONS(description = "Product involved in event"),
  category    STRING              OPTIONS(description = "Product category"),
  event_type  STRING              OPTIONS(description = "Type: page_view, purchase, etc."),
  quantity    INT64               OPTIONS(description = "Quantity of product"),
  price       FLOAT64             OPTIONS(description = "Unit price at time of event"),
  discount    FLOAT64             OPTIONS(description = "Fractional discount (0.0–1.0)"),
  timestamp   TIMESTAMP           OPTIONS(description = "Event timestamp (UTC)"),
  country     STRING              OPTIONS(description = "ISO country code"),
  _ingested_at TIMESTAMP          OPTIONS(description = "Pipeline ingestion timestamp")
)
PARTITION BY DATE(timestamp)
CLUSTER BY country, category
OPTIONS (
  description     = "Raw e-commerce events — Bronze layer",
  require_partition_filter = FALSE
);


-- ──────────────────────────────────────────────
-- SILVER DATASET (cleaned, validated)
-- ──────────────────────────────────────────────

CREATE SCHEMA IF NOT EXISTS `${project_id}.silver`
OPTIONS (
  description = "Silver layer: cleaned and validated data",
  location    = "US"
);

CREATE TABLE IF NOT EXISTS `${project_id}.silver.silver_events` (
  event_id     STRING     NOT NULL,
  user_id      INT64      NOT NULL,
  session_id   STRING,
  product_id   INT64      NOT NULL,
  category     STRING     NOT NULL,
  event_type   STRING     NOT NULL,
  quantity     INT64      NOT NULL,
  price        FLOAT64    NOT NULL,
  discount     FLOAT64    NOT NULL DEFAULT 0.0,
  revenue      FLOAT64    NOT NULL,
  timestamp    TIMESTAMP  NOT NULL,
  event_date   DATE       NOT NULL,
  event_hour   INT64,
  is_purchase  BOOL       NOT NULL DEFAULT FALSE,
  country      STRING,
  _ingested_at TIMESTAMP
)
PARTITION BY event_date
CLUSTER BY country, category
OPTIONS (description = "Silver events: cleaned and validated");

CREATE TABLE IF NOT EXISTS `${project_id}.silver.silver_products` (
  product_id   INT64      NOT NULL,
  product_name STRING     NOT NULL,
  category     STRING     NOT NULL,
  brand        STRING,
  cost_price   FLOAT64    NOT NULL,
  list_price   FLOAT64    NOT NULL,
  margin_pct   FLOAT64,
  stock_qty    INT64      NOT NULL DEFAULT 0,
  is_active    BOOL       NOT NULL DEFAULT TRUE,
  _ingested_at TIMESTAMP
)
OPTIONS (description = "Silver products: cleaned product dimension");

CREATE TABLE IF NOT EXISTS `${project_id}.silver.silver_users` (
  user_id      INT64      NOT NULL,
  username     STRING,
  email        STRING,
  country      STRING,
  loyalty_tier STRING,
  signup_date  TIMESTAMP,
  is_active    BOOL       NOT NULL DEFAULT TRUE,
  _ingested_at TIMESTAMP
)
OPTIONS (description = "Silver users: cleaned user dimension");


-- ──────────────────────────────────────────────
-- GOLD DATASET (analytics-ready star schema)
-- ──────────────────────────────────────────────

CREATE SCHEMA IF NOT EXISTS `${project_id}.gold`
OPTIONS (
  description = "Gold layer: analytics-ready star schema",
  location    = "US"
);

-- Fact table
CREATE TABLE IF NOT EXISTS `${project_id}.gold.fct_sales` (
  sale_key         STRING     NOT NULL,
  date_key         INT64      NOT NULL,
  user_key         INT64      NOT NULL,
  product_key      INT64      NOT NULL,
  session_id       STRING,
  category         STRING,
  country          STRING,
  quantity         INT64      NOT NULL,
  price            FLOAT64    NOT NULL,
  discount         FLOAT64    NOT NULL,
  revenue          FLOAT64    NOT NULL,
  gross_revenue    FLOAT64,
  discount_amount  FLOAT64,
  event_date       DATE       NOT NULL,
  event_hour       INT64,
  is_weekend       BOOL,
  year             INT64,
  quarter          INT64,
  month            INT64,
  loyalty_tier     STRING,
  user_country     STRING,
  brand            STRING,
  margin_pct       FLOAT64,
  _ingested_at     TIMESTAMP,
  _loaded_at       TIMESTAMP
)
PARTITION BY event_date
CLUSTER BY category, country
OPTIONS (description = "Gold sales fact table. Grain: one row per purchase.");

-- Date dimension
CREATE TABLE IF NOT EXISTS `${project_id}.gold.dim_date` (
  date_key         INT64      NOT NULL,
  full_date        DATE       NOT NULL,
  year             INT64,
  quarter          INT64,
  month            INT64,
  month_name       STRING,
  week_of_year     INT64,
  day_of_week      INT64,
  day_name         STRING,
  is_weekend       BOOL,
  is_holiday_season BOOL,
  sales_season     STRING
)
OPTIONS (description = "Date dimension: 2023-01-01 to 2024-12-31");

-- Aggregate table
CREATE TABLE IF NOT EXISTS `${project_id}.gold.agg_daily_sales` (
  date_key         INT64,
  event_date       DATE,
  year             INT64,
  quarter          INT64,
  month            INT64,
  category         STRING,
  country          STRING,
  is_weekend       BOOL,
  total_orders     INT64,
  total_units_sold INT64,
  total_revenue    FLOAT64,
  avg_order_value  FLOAT64,
  unique_customers INT64,
  _loaded_at       TIMESTAMP
)
PARTITION BY event_date
OPTIONS (description = "Pre-aggregated daily sales. Powers dashboard queries.");
