# AI-Augmented Data Pipeline - LLM-Powered Data Quality

> **Medallion Architecture** · Apache Airflow · PySpark · dbt · Delta Lake · LangChain · OpenAI API · BigQuery · GCP

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-2.8+-green.svg)](https://airflow.apache.org/)
[![dbt](https://img.shields.io/badge/dbt-1.7+-orange.svg)](https://www.getdbt.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This capstone project implements a **production-grade, end-to-end data engineering pipeline** for a synthetic e-commerce platform, processing **5M+ event rows per day** through a three-tier Medallion Architecture (Bronze → Silver → Gold).

The defining innovation is an **LLM-powered Data Quality Agent** - a LangChain agent that reads the Silver-layer schema and anomalous row samples, then generates a natural-language quality report with root-cause analysis and prioritised remediation suggestions. This replaces brittle hand-coded validation rules with dynamic, self-documenting quality checks.

---

## Key Features

| Feature | Details |
|---------|---------|
| **Medallion Architecture** | Bronze (raw) → Silver (clean) → Gold (star schema) |
| **LLM Quality Agent** | LangChain + GPT-3.5-Turbo generates NL quality reports; rule-based fallback for offline mode |
| **Airflow Orchestration** | Full Bronze-to-Gold DAG with exponential backoff, SLA monitoring, XCom, Slack alerts |
| **PySpark at Scale** | Partition pruning + broadcast joins cut Silver processing time by ~40% |
| **dbt Gold Layer** | Incremental star-schema models with Jinja templates, generic + singular tests |
| **BigQuery** | Partitioned/clustered fact table; pre-aggregated daily sales for dashboard performance |
| **Automated Testing** | 50+ unit tests covering ETL logic, SQL assertions, dbt project structure |

---

## Architecture

```
[Data Sources: CSV / JSON / API]
          │
          ▼
  ┌─────────────┐       ┌───────────────────────┐
  │   BRONZE    │──────▶│        SILVER          │
  │  GCS/Parquet│       │  Clean + Validate      │
  │  Raw ingest │       │  + LLM Quality Agent   │
  └─────────────┘       └──────────┬────────────┘
                                   │
                         ┌─────────▼─────────┐
                         │    Quality Gate    │
                         │  (Airflow Branch)  │
                         └──────┬──────┬─────┘
                         Pass ──┘      └── Fail → Slack Alert
                                │              (LLM Report)
                          ┌─────▼──────┐
                          │    GOLD    │
                          │ Star Schema│
                          │ BigQuery   │
                          │ + dbt      │
                          └─────┬──────┘
                                │
                   ┌────────────┼────────────┐
                   ▼            ▼            ▼
              Tableau        Power BI    Python Charts
```

---

## Project Structure

```
capstone-project/
├── data/
│   ├── raw/                    # Bronze: raw Parquet files
│   └── processed/              # Silver/Gold: cleaned Parquet files
│       └── gold/               # Gold star-schema tables
├── scripts/
│   ├── extract.py              # Bronze: data ingestion
│   ├── transform.py            # Silver: cleaning + LLM Quality Agent
│   ├── load.py                 # Gold: star-schema builder
│   └── utils.py                # Shared helpers, logging, validation
├── sql/
│   ├── schema.sql              # BigQuery DDL (all 3 layers)
│   └── transformations.sql     # Analytical SQL with window functions
├── airflow/
│   └── dag_capstone.py         # Full Airflow DAG (Bronze → Gold)
├── dbt/
│   ├── models/
│   │   ├── staging/            # Views over Silver tables
│   │   ├── intermediate/       # Business logic layer
│   │   └── marts/              # Gold: fct_sales, dims, aggregates
│   ├── seeds/                  # Static reference data
│   ├── macros/                 # Reusable Jinja macros
│   └── dbt_project.yml
├── viz/
│   └── charts.py               # Python dashboard (matplotlib)
├── tests/
│   ├── test_etl.py             # 40+ ETL unit tests (pytest)
│   └── test_dbt.py             # SQL-layer + dbt project tests
├── docs/
│   ├── architecture.md         # System design + data flow diagrams
│   └── deployment_guide.md     # Local + cloud setup instructions
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/your-org/capstone-project.git
cd capstone-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your GCP project and (optionally) OpenAI API key

# 3. Run the pipeline end-to-end
python scripts/extract.py --n-events 50000
python scripts/transform.py
python scripts/load.py

# 4. Generate dashboards
python viz/charts.py

# 5. Run all tests
pytest tests/ -v
```

---

## Running with Airflow

```bash
# Start Airflow (Docker)
docker compose up -d
# Open http://localhost:8080 (user: airflow / pw: airflow)

# Trigger the DAG manually
airflow dags trigger capstone_medallion_pipeline
```

---

## dbt

```bash
cd dbt/
dbt deps && dbt run && dbt test
dbt docs generate && dbt docs serve   # http://localhost:8080
```

---

## Sample Outputs

### LLM Quality Report (excerpt)
```
╔══════════════════════════════════════════════════════════════╗
║          DATA QUALITY REPORT - EVENTS                        ║
║          Batch: 2024-01-15                                    ║
╚══════════════════════════════════════════════════════════════╝

1. EXECUTIVE SUMMARY
   Processed 500,000 input rows; retained 489,432 rows (97.9% pass rate).
   1,024 rows dropped due to null critical values.

4. REMEDIATION SUGGESTIONS
   [HIGH]   Add NOT NULL constraint on price at source system.
   [MEDIUM] Implement idempotent event ingestion upstream.

5. OVERALL QUALITY SCORE: 94/100
```

### Gold Layer Row Counts
```json
{
  "dim_date":        731,
  "dim_product":     49982,
  "dim_user":        499501,
  "fact_sales":      97854,
  "agg_daily_sales": 4380,
  "agg_user_ltv":    45321
}
```

---

## Learning Path

See [LEARNING_PATH.md](docs/LEARNING_PATH.md) for the complete learning roadmap covering Python, SQL, Airflow, dbt, GCP, and DevOps from beginner to advanced.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | Apache Airflow 2.8 |
| Processing | PySpark 3.5 / pandas 2.x |
| Transformation | dbt 1.7 (BigQuery adapter) |
| Warehouse | Google BigQuery |
| Storage | Google Cloud Storage |
| LLM | LangChain + OpenAI GPT-3.5-Turbo |
| Testing | pytest + DuckDB |
| Visualisation | matplotlib / Tableau / Power BI |
| Language | Python 3.11 |

---

## License

MIT - see [LICENSE](LICENSE) for details.
