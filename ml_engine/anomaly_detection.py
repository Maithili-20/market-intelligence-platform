"""
anomaly_detection.py
--------------------
Detects unusual market patterns using Isolation Forest —
the same algorithm used in production fraud detection and
market surveillance systems.

Detects three types of anomalies:
  1. Price spikes    — abnormal single-day returns
  2. Volume surges   — unusual trading volume
  3. Sentiment shift — sudden change in news sentiment
"""

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME", "market_intel"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

CONTAMINATION = 0.05   # expect ~5% of data points to be anomalies


def load_data() -> pd.DataFrame:
    """Load features for anomaly detection."""
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT
            f.symbol, f.date,
            f.returns_1d, f.returns_5d,
            f.volatility_20d, f.volume_ratio,
            f.rsi_14, f.avg_sentiment,
            p.volume, p.close
        FROM stock_features f
        JOIN stock_prices p USING (symbol, date)
        WHERE f.returns_1d IS NOT NULL
          AND f.volatility_20d IS NOT NULL
        ORDER BY f.symbol, f.date
    """, conn)
    conn.close()

    df["avg_sentiment"] = df["avg_sentiment"].fillna(0)
    df["volume_ratio"]  = df["volume_ratio"].fillna(1)

    log.info(f"Loaded {len(df):,} rows for anomaly detection")
    return df


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Run Isolation Forest and tag anomalies."""

    feature_cols = [
        "returns_1d", "returns_5d",
        "volatility_20d", "volume_ratio",
        "rsi_14", "avg_sentiment",
    ]

    df_model = df[feature_cols].copy().fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)

    iso = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )

    df["anomaly_score"]  = iso.fit_predict(X_scaled)    # -1 = anomaly, 1 = normal
    df["anomaly_raw"]    = iso.score_samples(X_scaled)  # lower = more anomalous
    df["is_anomaly"]     = df["anomaly_score"] == -1

    n_anomalies = df["is_anomaly"].sum()
    log.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}% of data)")

    return df


def classify_anomaly_type(row: pd.Series) -> str:
    """Classify what kind of anomaly was detected."""
    if abs(row.get("returns_1d", 0)) > 0.05:
        return "price_spike"
    elif row.get("volume_ratio", 1) > 3.0:
        return "volume_surge"
    elif abs(row.get("avg_sentiment", 0)) > 0.5:
        return "sentiment_shift"
    else:
        return "general_anomaly"


def save_anomalies_to_db(df: pd.DataFrame) -> None:
    """Save detected anomalies to the anomalies table."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Clear existing anomalies
    cur.execute("DELETE FROM anomalies")

    anomalies = df[df["is_anomaly"]].copy()
    anomalies["anomaly_type"] = anomalies.apply(classify_anomaly_type, axis=1)

    rows = []
    for _, row in anomalies.iterrows():
        description = (
            f"return_1d={row.get('returns_1d', 0):.3f}, "
            f"vol_ratio={row.get('volume_ratio', 0):.2f}, "
            f"rsi={row.get('rsi_14', 0):.1f}, "
            f"sentiment={row.get('avg_sentiment', 0):.3f}"
        )
        rows.append((
            row["symbol"],
            row["date"],
            row["anomaly_type"],
            round(float(row["anomaly_raw"]), 4),
            description,
        ))

    execute_values(cur, """
        INSERT INTO anomalies (symbol, date, anomaly_type, score, description)
        VALUES %s
    """, rows)

    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Saved {len(rows)} anomalies to database")

    # Print top anomalies
    log.info("\nTop 10 most anomalous events:")
    top = anomalies.nsmallest(10, "anomaly_raw")[
        ["symbol", "date", "anomaly_type", "returns_1d", "volume_ratio", "anomaly_raw"]
    ]
    for _, r in top.iterrows():
        log.info(
            f"  {r.symbol:8s} {str(r.date):12s} "
            f"type={r.anomaly_type:18s} "
            f"return={r.returns_1d:.3f} "
            f"vol_ratio={r.volume_ratio:.2f}"
        )


if __name__ == "__main__":
    df = load_data()

    if len(df) < 50:
        log.error("Not enough data for anomaly detection. Need at least 50 rows.")
        exit(1)

    df = detect_anomalies(df)
    save_anomalies_to_db(df)

    log.info("Anomaly detection pipeline complete ✓")
