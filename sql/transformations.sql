-- transformations.sql
-- Analytical SQL Transformations for the Gold Layer
-- Run directly in BigQuery or adapted for use in dbt models
--
-- Replace ${project_id} with your GCP project ID
-- e.g. SET project_id = 'my-gcp-project';
--
-- Author: Capstone Project


-- ──────────────────────────────────────────────
-- 1. REVENUE BY CATEGORY × MONTH
--    Window function: running monthly total
-- ──────────────────────────────────────────────

WITH monthly_category_revenue AS (

    SELECT
        year,
        month,
        category,
        SUM(revenue)                                     AS monthly_revenue,
        SUM(total_orders)                                AS monthly_orders,
        ROUND(SUM(revenue) / NULLIF(SUM(total_orders), 0), 2)
                                                         AS avg_order_value
    FROM `${project_id}.gold.agg_daily_sales`
    GROUP BY year, month, category

)

SELECT
    year,
    month,
    category,
    monthly_revenue,
    monthly_orders,
    avg_order_value,
    -- Running total revenue per category (YTD)
    SUM(monthly_revenue) OVER (
        PARTITION BY year, category
        ORDER BY month
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )                                                    AS ytd_revenue,
    -- Month-over-month revenue growth
    ROUND(
        (monthly_revenue - LAG(monthly_revenue, 1) OVER (
            PARTITION BY category ORDER BY year, month
        )) / NULLIF(LAG(monthly_revenue, 1) OVER (
            PARTITION BY category ORDER BY year, month
        ), 0) * 100, 2
    )                                                    AS mom_revenue_growth_pct,
    -- Revenue rank within month
    RANK() OVER (
        PARTITION BY year, month
        ORDER BY monthly_revenue DESC
    )                                                    AS revenue_rank_in_month

FROM monthly_category_revenue
ORDER BY year, month, revenue_rank_in_month;


-- ──────────────────────────────────────────────
-- 2. TOP 10 PRODUCTS BY REVENUE (with margin)
-- ──────────────────────────────────────────────

SELECT
    p.product_name,
    p.category,
    p.brand,
    p.margin_pct,
    COUNT(f.sale_key)                                    AS total_purchases,
    SUM(f.quantity)                                      AS total_units_sold,
    ROUND(SUM(f.revenue), 2)                             AS total_revenue,
    ROUND(AVG(f.revenue), 2)                             AS avg_order_value,
    ROUND(SUM(f.revenue) * p.margin_pct / 100, 2)       AS estimated_profit,
    DENSE_RANK() OVER (
        ORDER BY SUM(f.revenue) DESC
    )                                                    AS revenue_rank

FROM `${project_id}.gold.fct_sales`        f
JOIN `${project_id}.gold.dim_product`      p
  ON f.product_key = p.product_key
WHERE f.year = EXTRACT(YEAR FROM CURRENT_DATE())
GROUP BY
    p.product_name, p.category, p.brand, p.margin_pct
QUALIFY DENSE_RANK() OVER (ORDER BY SUM(f.revenue) DESC) <= 10
ORDER BY revenue_rank;


-- ──────────────────────────────────────────────
-- 3. USER COHORT ANALYSIS (first purchase month)
-- ──────────────────────────────────────────────

WITH first_purchases AS (

    SELECT
        user_key,
        MIN(event_date)                                  AS first_purchase_date,
        DATE_TRUNC(MIN(event_date), MONTH)               AS cohort_month
    FROM `${project_id}.gold.fct_sales`
    GROUP BY user_key

),

user_activity AS (

    SELECT
        f.user_key,
        fp.cohort_month,
        DATE_TRUNC(f.event_date, MONTH)                  AS activity_month,
        DATE_DIFF(
            DATE_TRUNC(f.event_date, MONTH),
            fp.cohort_month,
            MONTH
        )                                                AS months_since_first_purchase,
        SUM(f.revenue)                                   AS revenue
    FROM `${project_id}.gold.fct_sales` f
    JOIN first_purchases fp USING (user_key)
    GROUP BY f.user_key, fp.cohort_month, activity_month

)

SELECT
    cohort_month,
    months_since_first_purchase,
    COUNT(DISTINCT user_key)                             AS active_users,
    ROUND(SUM(revenue), 2)                               AS cohort_revenue,
    ROUND(
        COUNT(DISTINCT user_key) /
        MAX(COUNT(DISTINCT user_key)) OVER (
            PARTITION BY cohort_month
        ) * 100, 1
    )                                                    AS retention_rate_pct
FROM user_activity
GROUP BY cohort_month, months_since_first_purchase
ORDER BY cohort_month, months_since_first_purchase;


-- ──────────────────────────────────────────────
-- 4. BASKET ANALYSIS (co-purchased categories)
--    Which categories are bought together most?
-- ──────────────────────────────────────────────

WITH session_categories AS (

    SELECT
        session_id,
        ARRAY_AGG(DISTINCT category ORDER BY category)  AS categories_in_session
    FROM `${project_id}.gold.fct_sales`
    GROUP BY session_id
    HAVING ARRAY_LENGTH(ARRAY_AGG(DISTINCT category)) >= 2

),

category_pairs AS (

    SELECT
        cats[OFFSET(i)] AS category_a,
        cats[OFFSET(j)] AS category_b,
        COUNT(*)         AS co_purchase_count
    FROM session_categories,
         UNNEST(categories_in_session) AS cat WITH OFFSET pos,
         UNNEST(GENERATE_ARRAY(0, ARRAY_LENGTH(categories_in_session) - 1)) AS i,
         UNNEST(GENERATE_ARRAY(i + 1, ARRAY_LENGTH(categories_in_session) - 1)) AS j,
         UNNEST([categories_in_session]) AS cats
    WHERE i < j
    GROUP BY category_a, category_b

)

SELECT
    category_a,
    category_b,
    co_purchase_count,
    RANK() OVER (ORDER BY co_purchase_count DESC)        AS pair_rank
FROM category_pairs
ORDER BY co_purchase_count DESC
LIMIT 20;


-- ──────────────────────────────────────────────
-- 5. WEEKEND vs WEEKDAY REVENUE COMPARISON
-- ──────────────────────────────────────────────

SELECT
    year,
    quarter,
    category,
    CASE WHEN is_weekend THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(sale_key)                                      AS total_orders,
    ROUND(SUM(revenue), 2)                               AS total_revenue,
    ROUND(AVG(revenue), 2)                               AS avg_order_value,
    -- What % of weekly revenue falls on weekends?
    ROUND(
        SUM(revenue) / SUM(SUM(revenue)) OVER (
            PARTITION BY year, quarter, category
        ) * 100, 1
    )                                                    AS revenue_share_pct
FROM `${project_id}.gold.fct_sales`
GROUP BY year, quarter, category, is_weekend
ORDER BY year, quarter, category, day_type;


-- ──────────────────────────────────────────────
-- 6. LOYALTY TIER SPEND COMPARISON (CTE + Window)
-- ──────────────────────────────────────────────

WITH tier_stats AS (

    SELECT
        loyalty_tier,
        COUNT(DISTINCT user_key)                         AS customer_count,
        COUNT(sale_key)                                  AS total_orders,
        ROUND(SUM(revenue), 2)                           AS total_revenue,
        ROUND(AVG(revenue), 2)                           AS avg_order_value,
        ROUND(PERCENTILE_CONT(revenue, 0.5) OVER (
            PARTITION BY loyalty_tier
        ), 2)                                            AS median_order_value
    FROM `${project_id}.gold.fct_sales`
    GROUP BY loyalty_tier, user_key

)

SELECT
    loyalty_tier,
    SUM(customer_count)                                  AS customers,
    SUM(total_orders)                                    AS orders,
    SUM(total_revenue)                                   AS revenue,
    ROUND(AVG(avg_order_value), 2)                       AS avg_aov,
    ROUND(SUM(total_revenue) / NULLIF(SUM(customer_count), 0), 2)
                                                         AS ltv_per_customer,
    RANK() OVER (ORDER BY SUM(total_revenue) DESC)       AS revenue_rank
FROM tier_stats
GROUP BY loyalty_tier
ORDER BY revenue_rank;
