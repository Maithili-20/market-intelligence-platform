"""
fetch_macro.py
--------------
Pulls macroeconomic indicators from the FRED API (Federal Reserve Economic Data).
Free API key at: https://fred.stlouisfed.org/docs/api/api_key.html

Indicators fetched:
  FEDFUNDS  — Federal Funds Rate
  CPIAUCSL  — Consumer Price Index (inflation)
  UNRATE    — Unemployment Rate
  GDP       — Gross Domestic Product
  T10Y2Y    — 10Y-2Y Treasury yield spread (recession indicator)
  VIXCLS    — VIX volatility index
"""

import os
import logging
from datetime import datetime, timedelta

import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY", "")   # Free at fred.stlouisfed.org
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME", "market_intel"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

INDICATORS = {
    "FEDFUNDS": "Federal Funds Rate (%)",
    "CPIAUCSL": "Consumer Price Index",
    "UNRATE":   "Unemployment Rate (%)",
    "T10Y2Y":   "10Y-2Y Yield Spread",
    "VIXCLS":   "VIX Volatility Index",
    "GDP":      "Gross Domestic Product",
}


def fetch_indicator(series_id: str, days_back: int = 365) -> list[tuple]:
    """Fetch a single FRED series and return (indicator, date, value) tuples."""
    if not FRED_API_KEY:
        log.error("FRED_API_KEY not set")
        return []

    obs_start = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "series_id":         series_id,
        "observation_start": obs_start,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
    }

    resp = requests.get(FRED_BASE, params=params, timeout=10)
    resp.raise_for_status()
    observations = resp.json().get("observations", [])

    rows = []
    for obs in observations:
        value_str = obs.get("value", ".")
        if value_str == ".":    # FRED uses "." for missing values
            continue
        try:
            rows.append((series_id, obs["date"], float(value_str)))
        except (ValueError, KeyError):
            continue

    log.info(f"  {series_id}: {len(rows)} observations")
    return rows


def upsert_macro(rows: list[tuple]) -> None:
    """Upsert macro indicator rows into PostgreSQL."""
    if not rows:
        return

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    sql = """
        INSERT INTO macro_indicators (indicator, date, value)
        VALUES %s
        ON CONFLICT (indicator, date) DO UPDATE SET
            value = EXCLUDED.value
    """
    execute_values(cur, sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Upserted {len(rows)} macro rows")


if __name__ == "__main__":
    all_rows = []
    for series_id in INDICATORS:
        try:
            rows = fetch_indicator(series_id, days_back=365)
            all_rows.extend(rows)
        except Exception as e:
            log.error(f"Failed {series_id}: {e}")

    upsert_macro(all_rows)
    log.info("Macro pipeline complete ✓")
