"""
dag_capstone.py — Airflow DAG: Bronze → Silver → Gold Medallion Pipeline

Orchestrates the full e-commerce data pipeline with:
- Medallion Architecture: Bronze → Silver → Gold
- LLM Quality Agent alert on Silver anomalies
- Exponential backoff retries on all tasks
- SLA-based monitoring
- Custom Slack failure notification operator
- XCom-based task communication

Schedule: Daily at 02:00 UTC
Owner:    data-engineering

Author: Capstone Project
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.models import Variable
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.dates import days_ago


# ──────────────────────────────────────────────
# DEFAULT ARGS
# ──────────────────────────────────────────────

DEFAULT_ARGS: Dict[str, Any] = {
    "owner":             "data-engineering",
    "depends_on_past":   False,
    "email":             ["data-alerts@company.com"],
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           3,
    "retry_delay":       timedelta(minutes=2),
    "retry_exponential_backoff": True,      # Exponential backoff
    "max_retry_delay":   timedelta(minutes=30),
    "execution_timeout": timedelta(hours=2),
    "sla":               timedelta(hours=4), # SLA: pipeline must finish in 4h
}

# Airflow Variables (set via UI or environment)
GCP_PROJECT    = Variable.get("GCP_PROJECT_ID", default_var="my-gcp-project")
SLACK_CONN_ID  = "slack_webhook_capstone"
N_EVENTS       = int(Variable.get("N_EVENTS", default_var="500000"))
UPLOAD_GCS     = Variable.get("UPLOAD_GCS", default_var="false").lower() == "true"
REGISTER_BQ    = Variable.get("REGISTER_BQ", default_var="false").lower() == "true"
QUALITY_THRESHOLD = float(Variable.get("QUALITY_THRESHOLD", default_var="95.0"))


# ──────────────────────────────────────────────
# SLACK NOTIFICATION HELPER
# ──────────────────────────────────────────────

def get_slack_failure_message(context: Dict) -> str:
    """Build a rich Slack failure notification message."""
    dag_id    = context["dag"].dag_id
    task_id   = context["task_instance"].task_id
    exec_date = context["execution_date"]
    log_url   = context["task_instance"].log_url

    return json.dumps({
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": "🚨 Airflow Pipeline Failure"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*DAG:* `{dag_id}`"},
                    {"type": "mrkdwn", "text": f"*Task:* `{task_id}`"},
                    {"type": "mrkdwn", "text": f"*Execution:* `{exec_date}`"},
                    {"type": "mrkdwn", "text": f"*Log:* <{log_url}|View Logs>"},
                ],
            },
        ]
    })


def slack_on_failure_callback(context: Dict) -> None:
    """Airflow on_failure_callback that sends a Slack alert."""
    msg = get_slack_failure_message(context)
    slack_op = SlackWebhookOperator(
        task_id="slack_failure_notification",
        slack_webhook_conn_id=SLACK_CONN_ID,
        message=msg,
        dag=context["dag"],
    )
    slack_op.execute(context)


# ──────────────────────────────────────────────
# PYTHON CALLABLE: BRONZE EXTRACTION
# ──────────────────────────────────────────────

def bronze_extraction(**context) -> str:
    """
    Run the Bronze layer extraction.
    Pushes output file paths to XCom.
    """
    # Import inside function to avoid Airflow serialization issues
    import sys
    sys.path.insert(0, "/opt/airflow/dags")

    from scripts.extract import run_bronze_extraction

    result = run_bronze_extraction(
        n_events=N_EVENTS,
        upload_to_gcs=UPLOAD_GCS,
        register_bq=REGISTER_BQ,
    )

    # Push to XCom for downstream tasks
    context["ti"].xcom_push(key="bronze_paths", value=result)
    return json.dumps(result)


# ──────────────────────────────────────────────
# PYTHON CALLABLE: SILVER TRANSFORMATION
# ──────────────────────────────────────────────

def silver_transformation(**context) -> str:
    """
    Run the Silver layer transformation + LLM Quality Agent.
    Pushes quality report and metrics to XCom.
    """
    import sys
    sys.path.insert(0, "/opt/airflow/dags")

    from scripts.transform import run_silver_transformation

    result = run_silver_transformation(
        register_bq=REGISTER_BQ,
        run_llm_agent=True,
    )

    # Push quality metrics + report to XCom
    context["ti"].xcom_push(key="quality_metrics", value=result["quality_metrics"])
    context["ti"].xcom_push(key="quality_report",  value=result["quality_report"])
    return json.dumps({k: v for k, v in result.items() if k != "quality_report"})


# ──────────────────────────────────────────────
# PYTHON CALLABLE: QUALITY GATE (Branch)
# ──────────────────────────────────────────────

def quality_gate(**context) -> str:
    """
    Branch operator: evaluate Silver quality metrics.
    Routes to 'proceed_to_gold' or 'alert_quality_failure'.

    Decision: pass_rate < QUALITY_THRESHOLD → alert branch.
    """
    metrics = context["ti"].xcom_pull(
        task_ids="silver_transformation",
        key="quality_metrics",
    )

    if not metrics:
        return "alert_quality_failure"

    pass_rate = metrics.get("cleaning_pass_rate", 0)
    print(f"Quality gate: pass_rate={pass_rate}% (threshold={QUALITY_THRESHOLD}%)")

    if pass_rate < QUALITY_THRESHOLD:
        print(f"⚠️  Quality below threshold ({pass_rate} < {QUALITY_THRESHOLD}). Alerting.")
        return "alert_quality_failure"

    print(f"✅  Quality passed: {pass_rate}% ≥ {QUALITY_THRESHOLD}%")
    return "proceed_to_gold"


# ──────────────────────────────────────────────
# PYTHON CALLABLE: QUALITY ALERT
# ──────────────────────────────────────────────

def send_quality_alert(**context) -> None:
    """
    Send a Slack alert when the quality gate fails.
    Includes the LLM quality report and remediation suggestions.
    """
    quality_report = context["ti"].xcom_pull(
        task_ids="silver_transformation",
        key="quality_report",
    ) or "No quality report available."

    metrics = context["ti"].xcom_pull(
        task_ids="silver_transformation",
        key="quality_metrics",
    ) or {}

    pass_rate = metrics.get("cleaning_pass_rate", "N/A")
    dag_id    = context["dag"].dag_id
    exec_date = context["execution_date"]

    slack_msg = (
        f"⚠️ *Data Quality Alert*\n"
        f"DAG: `{dag_id}` | Execution: `{exec_date}`\n"
        f"Pass Rate: `{pass_rate}%` (below {QUALITY_THRESHOLD}%)\n\n"
        f"*LLM Quality Report (excerpt):*\n```{quality_report[:800]}```\n\n"
        f"*Action:* Review Silver layer logs and consider reprocessing."
    )

    # In a real deployment, this would call the Slack API
    print(f"SLACK ALERT:\n{slack_msg}")


# ──────────────────────────────────────────────
# PYTHON CALLABLE: GOLD LOADING
# ──────────────────────────────────────────────

def gold_loading(**context) -> str:
    """Run the Gold layer loading (star schema build)."""
    import sys
    sys.path.insert(0, "/opt/airflow/dags")

    from scripts.load import run_gold_load

    result = run_gold_load(register_bq=REGISTER_BQ)
    context["ti"].xcom_push(key="gold_row_counts", value=result["row_counts"])
    return json.dumps(result)


# ──────────────────────────────────────────────
# PYTHON CALLABLE: dbt RUNNER
# ──────────────────────────────────────────────

def run_dbt_gold(**context) -> None:
    """
    Execute dbt Gold layer models via subprocess.
    In production, use the dbt Cloud API or Astronomer Cosmos.
    """
    import subprocess

    dbt_cmd = [
        "dbt", "run",
        "--project-dir", "/opt/airflow/dbt",
        "--profiles-dir", "/opt/airflow/dbt",
        "--select", "tag:gold",
        "--vars", f'{{"batch_date": "{datetime.utcnow().strftime("%Y-%m-%d")}"}}'
    ]

    print(f"Running dbt command: {' '.join(dbt_cmd)}")
    result = subprocess.run(dbt_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"dbt run failed:\n{result.stderr}")

    print(f"dbt stdout:\n{result.stdout}")


def run_dbt_test(**context) -> None:
    """Execute dbt tests on the Gold layer models."""
    import subprocess

    test_cmd = [
        "dbt", "test",
        "--project-dir", "/opt/airflow/dbt",
        "--profiles-dir", "/opt/airflow/dbt",
        "--select", "tag:gold",
    ]

    result = subprocess.run(test_cmd, capture_output=True, text=True)
    print(f"dbt test stdout:\n{result.stdout}")

    if result.returncode != 0:
        raise RuntimeError(f"dbt test failed:\n{result.stderr}")


# ──────────────────────────────────────────────
# PYTHON CALLABLE: PIPELINE SUMMARY
# ──────────────────────────────────────────────

def pipeline_summary(**context) -> None:
    """Log a final summary of the completed pipeline run."""
    gold_counts  = context["ti"].xcom_pull(task_ids="gold_loading", key="gold_row_counts") or {}
    q_metrics    = context["ti"].xcom_pull(task_ids="silver_transformation", key="quality_metrics") or {}

    summary = {
        "dag_id":           context["dag"].dag_id,
        "execution_date":   str(context["execution_date"]),
        "batch_date":       datetime.utcnow().strftime("%Y-%m-%d"),
        "quality_pass_rate": q_metrics.get("cleaning_pass_rate"),
        "gold_row_counts":  gold_counts,
        "status":           "SUCCESS",
    }
    print(f"╔══════════════════════════════════════╗")
    print(f"║      PIPELINE SUMMARY                ║")
    print(f"╚══════════════════════════════════════╝")
    print(json.dumps(summary, indent=2))


# ──────────────────────────────────────────────
# DAG DEFINITION
# ──────────────────────────────────────────────

with DAG(
    dag_id="capstone_medallion_pipeline",
    description="AI-Augmented Medallion Pipeline: Bronze → Silver (LLM QA) → Gold",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 2 * * *",     # Daily at 02:00 UTC
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["capstone", "medallion", "llm", "data-quality"],
    on_failure_callback=slack_on_failure_callback,
    doc_md="""
## Capstone Medallion Pipeline

### Architecture
```
[Bronze Extraction]
       │
       ▼
[Silver Transformation]
       │
       ▼
[Quality Gate (LLM Agent)]
    ┌──┴──┐
    │     │
[Alert]  [Gold Loading]
          │
          ▼
        [dbt run]
          │
          ▼
        [dbt test]
          │
          ▼
    [Summary]
```

### SLA
Pipeline must complete within **4 hours** of scheduled start.

### Retries
All tasks: 3 retries with exponential backoff (2min → 4min → 8min).
    """,
) as dag:

    # ── TASK 1: Start sentinel ────────────────────────────────────────
    start = EmptyOperator(
        task_id="start",
        doc_md="Pipeline start sentinel. No operation.",
    )

    # ── TASK 2: Bronze Extraction ─────────────────────────────────────
    bronze_task = PythonOperator(
        task_id="bronze_extraction",
        python_callable=bronze_extraction,
        doc_md="Ingests raw events, products, and users into the Bronze zone (Parquet + GCS).",
    )

    # ── TASK 3: Silver Transformation + LLM QA Agent ─────────────────
    silver_task = PythonOperator(
        task_id="silver_transformation",
        python_callable=silver_transformation,
        doc_md="Cleans Bronze data and runs the LangChain LLM Quality Agent.",
    )

    # ── TASK 4: Quality Gate (Branch) ────────────────────────────────
    quality_gate_task = BranchPythonOperator(
        task_id="quality_gate",
        python_callable=quality_gate,
        doc_md="Evaluates Silver quality metrics. Routes to gold or alert.",
    )

    # ── TASK 5a: Quality Alert ─────────────────────────────────────── 
    alert_task = PythonOperator(
        task_id="alert_quality_failure",
        python_callable=send_quality_alert,
        doc_md="Sends Slack alert with LLM quality report when pass_rate is too low.",
    )

    # ── TASK 5b: Proceed to Gold sentinel ────────────────────────────
    proceed_to_gold = EmptyOperator(
        task_id="proceed_to_gold",
        doc_md="Quality gate passed — proceed to Gold layer.",
    )

    # ── TASK 6: Gold Loading ──────────────────────────────────────────
    gold_task = PythonOperator(
        task_id="gold_loading",
        python_callable=gold_loading,
        trigger_rule=TriggerRule.ONE_SUCCESS,  # Run if either branch completes
        doc_md="Builds star-schema fact and dimension tables in the Gold zone.",
    )

    # ── TASK 7: dbt Run ───────────────────────────────────────────────
    dbt_run_task = PythonOperator(
        task_id="dbt_run",
        python_callable=run_dbt_gold,
        doc_md="Executes dbt Gold layer models (Jinja-templated SQL transformations).",
    )

    # ── TASK 8: dbt Test ──────────────────────────────────────────────
    dbt_test_task = PythonOperator(
        task_id="dbt_test",
        python_callable=run_dbt_test,
        doc_md="Runs dbt generic and singular tests on Gold layer models.",
    )

    # ── TASK 9: Pipeline Summary ──────────────────────────────────────
    summary_task = PythonOperator(
        task_id="pipeline_summary",
        python_callable=pipeline_summary,
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Logs final pipeline summary including row counts and quality metrics.",
    )

    # ── TASK 10: End sentinel ─────────────────────────────────────────
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.ALL_DONE,
        doc_md="Pipeline end sentinel.",
    )

    # ──────────────────────────────────────────────
    # TASK DEPENDENCIES
    # ──────────────────────────────────────────────
    #
    #   start
    #     │
    #   bronze_extraction
    #     │
    #   silver_transformation
    #     │
    #   quality_gate ──── alert_quality_failure ──┐
    #     │                                        │
    #   proceed_to_gold                            │
    #     │                                        │
    #   gold_loading  ◄─────────────────────────── ┘
    #     │
    #   dbt_run
    #     │
    #   dbt_test
    #     │
    #   pipeline_summary
    #     │
    #   end
    #

    (
        start
        >> bronze_task
        >> silver_task
        >> quality_gate_task
        >> [alert_task, proceed_to_gold]
    )

    alert_task     >> gold_task
    proceed_to_gold >> gold_task

    (
        gold_task
        >> dbt_run_task
        >> dbt_test_task
        >> summary_task
        >> end
    )
