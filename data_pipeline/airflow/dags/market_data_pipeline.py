"""
market_data_pipeline.py
-----------------------
Airflow DAG: runs all three ingestion scripts every day at 6:30 AM IST.
This is your first real data engineering artifact — DAGs are in every
data engineering / analytics engineering JD.

DAG structure:
    start
      ├── fetch_prices
      ├── fetch_news
      └── fetch_macro
            └── (all three) → compute_features → end
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# Import your pipeline functions
import sys
sys.path.append("/opt/airflow/dags/ingestion")

from fetch_prices import fetch_prices, upsert_prices, compute_and_store_features, WATCHLIST
from fetch_news   import fetch_news, upsert_articles, FINANCIAL_QUERIES
from fetch_macro  import fetch_indicator, upsert_macro, INDICATORS

# ── Default args ──────────────────────────────────────────────
default_args = {
    "owner":            "maithili",
    "depends_on_past":  False,
    "email_on_failure": False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# ── DAG definition ────────────────────────────────────────────
with DAG(
    dag_id="market_data_pipeline",
    default_args=default_args,
    description="Daily ingestion: prices, news, macro indicators",
    schedule_interval="30 1 * * 1-5",   # 6:30 AM IST = 1:00 AM UTC, weekdays only
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["market-intelligence", "ingestion"],
) as dag:

    # ── Task functions ────────────────────────────────────────

    def task_fetch_prices():
        df = fetch_prices(WATCHLIST, days_back=5)   # only last 5 days daily
        upsert_prices(df)

    def task_compute_features():
        # Re-fetch last 30 days to ensure feature windows are complete
        from fetch_prices import fetch_prices, compute_and_store_features, WATCHLIST
        df = fetch_prices(WATCHLIST, days_back=30)
        compute_and_store_features(df)

    def task_fetch_news():
        all_articles = []
        for query in FINANCIAL_QUERIES:
            articles = fetch_news(query, days_back=1)
            all_articles.extend(articles)
        # Deduplicate
        seen, unique = set(), []
        for a in all_articles:
            url = a.get("url", "")
            if url not in seen:
                seen.add(url)
                unique.append(a)
        upsert_articles(unique)

    def task_fetch_macro():
        all_rows = []
        for series_id in INDICATORS:
            rows = fetch_indicator(series_id, days_back=30)
            all_rows.extend(rows)
        upsert_macro(all_rows)

    # ── Tasks ─────────────────────────────────────────────────

    start = EmptyOperator(task_id="start")
    end   = EmptyOperator(task_id="end")

    prices = PythonOperator(
        task_id="fetch_prices",
        python_callable=task_fetch_prices,
    )

    news = PythonOperator(
        task_id="fetch_news",
        python_callable=task_fetch_news,
    )

    macro = PythonOperator(
        task_id="fetch_macro",
        python_callable=task_fetch_macro,
    )

    features = PythonOperator(
        task_id="compute_features",
        python_callable=task_compute_features,
    )

    # ── Dependencies ──────────────────────────────────────────
    # start → [prices, news, macro] run in parallel → features → end
    start >> [prices, news, macro] >> features >> end
