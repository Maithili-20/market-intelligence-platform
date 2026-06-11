"""
fetch_prices.py (v2 — large scale)
------------------------------------
Three data sources, unified into one pipeline:

  1. Kaggle bulk CSV    → 8M+ historical rows (run once on first setup)
  2. yfinance live      → full S&P 500, daily updates
  3. Polygon.io         → higher quality live data (optional, free tier)

Usage:
  python fetch_prices.py --mode bulk      # first-time Kaggle load
  python fetch_prices.py --mode live      # daily update (scheduled by Airflow)
  python fetch_prices.py --mode polygon   # Polygon.io live feed
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

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

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
CHUNK_SIZE      = 50_000   # rows per DB upsert batch — safe for 8M row loads


# ═══════════════════════════════════════════════════════════════
# HELPER: Full S&P 500 watchlist from Wikipedia (503 symbols)
# ═══════════════════════════════════════════════════════════════

def get_sp500_symbols() -> list[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        symbols = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"Loaded {len(symbols)} S&P 500 symbols")
        return symbols
    except Exception as e:
        log.warning(f"Wikipedia scrape failed ({e}), using hardcoded fallback list")
        return [
            "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","V","JNJ",
            "UNH","XOM","WMT","MA","PG","HD","CVX","MRK","LLY","ABBV",
            "PEP","KO","AVGO","COST","MCD","TMO","ACN","DHR","ABT","CRM",
            "BAC","NFLX","ADBE","CSCO","WFC","TXN","NEE","RTX","QCOM","BMY",
            "INTU","AMGN","SCHW","IBM","GS","CAT","SPGI","BLK","AXP","ISRG",
        ]


# ═══════════════════════════════════════════════════════════════
# SOURCE 1: Kaggle bulk CSV loader (8M+ rows, run once)
# ═══════════════════════════════════════════════════════════════

def load_kaggle_bulk(csv_path: str) -> None:
    """
    Load the Kaggle 'Huge Stock Market Dataset' CSV into PostgreSQL in chunks.

    Download steps:
      pip install kaggle
      kaggle datasets download -d borismarjanovic/price-volume-data-for-all-us-stocks-etfs
      unzip the file — you'll get a folder of per-symbol .txt files

    OR download the S&P 500 dataset (single CSV, easier):
      kaggle datasets download -d camnugent/sandp500
      → gives you all_stocks_5yr.csv  (2.3M rows, Date,Open,High,Low,Close,Volume,Name)

    Pass the path to that CSV as csv_path.
    """
    path = Path(csv_path)
    if not path.exists():
        log.error(f"File not found: {csv_path}")
        log.error("Download it with: kaggle datasets download -d camnugent/sandp500")
        sys.exit(1)

    log.info(f"Loading bulk CSV: {path} ({path.stat().st_size / 1e6:.1f} MB)")

    conn = psycopg2.connect(**DB_CONFIG)
    total_inserted = 0

    # Read CSV in chunks — never load 8M rows into memory at once
    for chunk_num, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):

        # ── Normalise column names across different Kaggle datasets ──
        chunk.columns = [c.strip().lower() for c in chunk.columns]

        # Handle both formats:
        #   camnugent/sandp500:          date, open, high, low, close, volume, name
        #   borismarjanovic bulk folder: date, open, high, low, close, volume, openint
        if "name" in chunk.columns:
            chunk = chunk.rename(columns={"name": "symbol"})
        elif "symbol" not in chunk.columns:
            log.error("Cannot find 'symbol' or 'name' column in CSV")
            break

        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.date
        chunk = chunk.dropna(subset=["date", "symbol", "close"])
        chunk["symbol"] = chunk["symbol"].str.upper().str.strip()

        # Cast numerics safely
        for col in ["open", "high", "low", "close"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").round(4)
        chunk["volume"] = pd.to_numeric(chunk.get("volume", 0), errors="coerce").fillna(0).astype(int)

        chunk = chunk.dropna(subset=["close"])

        rows = [
            (r.symbol, r.date, r.open, r.high, r.low, r.close, r.volume, r.close)
            for r in chunk.itertuples()
        ]

        _upsert_price_rows(conn, rows)
        total_inserted += len(rows)

        if chunk_num % 10 == 0:
            log.info(f"  Chunk {chunk_num}: {total_inserted:,} rows inserted so far...")

    conn.close()
    log.info(f"Bulk load complete — {total_inserted:,} total rows inserted")


def load_kaggle_folder(folder_path: str) -> None:
    """
    Load the 'Huge Stock Market Dataset' which is a folder of per-symbol .txt files.
    Each file: Date,Open,High,Low,Close,Volume,OpenInt
    This dataset has 8M+ rows across ~7000 symbols.
    """
    folder = Path(folder_path)
    files  = list(folder.glob("*.txt")) + list(folder.glob("*.csv"))
    log.info(f"Found {len(files)} symbol files in {folder}")

    conn = psycopg2.connect(**DB_CONFIG)
    total_inserted = 0
    batch_rows = []

    for i, fpath in enumerate(files):
        symbol = fpath.stem.upper()
        try:
            df = pd.read_csv(fpath, header=0)
            df.columns = [c.strip().lower() for c in df.columns]
            df["date"]   = pd.to_datetime(df["date"], errors="coerce").dt.date
            df["symbol"] = symbol
            df = df.dropna(subset=["date", "close"])

            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
            df["volume"] = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0).astype(int)

            for r in df.itertuples():
                batch_rows.append((r.symbol, r.date, r.open, r.high, r.low, r.close, r.volume, r.close))

        except Exception as e:
            log.warning(f"Skipping {fpath.name}: {e}")

        # Flush every CHUNK_SIZE rows
        if len(batch_rows) >= CHUNK_SIZE:
            _upsert_price_rows(conn, batch_rows)
            total_inserted += len(batch_rows)
            batch_rows = []
            log.info(f"  {i+1}/{len(files)} files — {total_inserted:,} rows inserted")

    # Flush remainder
    if batch_rows:
        _upsert_price_rows(conn, batch_rows)
        total_inserted += len(batch_rows)

    conn.close()
    log.info(f"Folder load complete — {total_inserted:,} rows from {len(files)} symbols")


# ═══════════════════════════════════════════════════════════════
# SOURCE 2: yfinance live — full S&P 500 daily update
# ═══════════════════════════════════════════════════════════════

def fetch_live_yfinance(symbols: list[str] = None, days_back: int = 5) -> None:
    """
    Pull recent OHLCV for all S&P 500 symbols and upsert.
    Batches symbols into groups of 50 to avoid yfinance rate limits.
    """
    import yfinance as yf

    if symbols is None:
        symbols = get_sp500_symbols()

    end   = datetime.today()
    start = (end - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end   = end.strftime("%Y-%m-%d")

    log.info(f"Fetching live data: {len(symbols)} symbols, {start} → {end}")

    conn = psycopg2.connect(**DB_CONFIG)
    total = 0

    # Batch into groups of 50 — yfinance handles multi-tickers well up to ~100
    batch_size = 50
    for batch_start in range(0, len(symbols), batch_size):
        batch = symbols[batch_start : batch_start + batch_size]
        try:
            raw = yf.download(
                tickers=batch,
                start=start,
                end=end,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )

            rows = []
            for symbol in batch:
                try:
                    df = raw[symbol] if len(batch) > 1 else raw
                    df = df.dropna(subset=["Close"])
                    for date, row in df.iterrows():
                        rows.append((
                            symbol,
                            date.date(),
                            round(float(row["Open"]),  4),
                            round(float(row["High"]),  4),
                            round(float(row["Low"]),   4),
                            round(float(row["Close"]), 4),
                            int(row["Volume"]),
                            round(float(row["Close"]), 4),
                        ))
                except Exception:
                    pass

            if rows:
                _upsert_price_rows(conn, rows)
                total += len(rows)

            log.info(f"  Batch {batch_start//batch_size + 1}: {len(rows)} rows")

        except Exception as e:
            log.warning(f"Batch failed: {e}")

    conn.close()
    log.info(f"Live fetch complete — {total:,} rows upserted")

    # Compute features after live update
    _compute_features_from_db(conn_config=DB_CONFIG, days_back=days_back + 30)


# ═══════════════════════════════════════════════════════════════
# SOURCE 3: Polygon.io — higher quality, free tier
# ═══════════════════════════════════════════════════════════════

def fetch_polygon(symbols: list[str] = None, days_back: int = 5) -> None:
    """
    Pull daily OHLCV from Polygon.io for all symbols.
    Free tier: unlimited historical data (2 years), 5 API calls/minute.

    Get your free key at: https://polygon.io/dashboard/signup
    Add to .env: POLYGON_API_KEY=your_key_here
    """
    import requests
    import time

    if not POLYGON_API_KEY:
        log.error("POLYGON_API_KEY not set in .env — get a free key at polygon.io")
        return

    if symbols is None:
        symbols = get_sp500_symbols()

    end_date   = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    base_url   = "https://api.polygon.io/v2/aggs/ticker"

    conn = psycopg2.connect(**DB_CONFIG)
    total = 0
    batch_rows = []

    for i, symbol in enumerate(symbols):
        url = (
            f"{base_url}/{symbol}/range/1/day/{start_date}/{end_date}"
            f"?adjusted=true&sort=asc&limit=500&apiKey={POLYGON_API_KEY}"
        )
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 429:
                log.warning("Rate limited — sleeping 60s")
                time.sleep(60)
                resp = requests.get(url, timeout=10)

            resp.raise_for_status()
            results = resp.json().get("results", [])

            for r in results:
                date = datetime.fromtimestamp(r["t"] / 1000).date()
                batch_rows.append((
                    symbol, date,
                    round(r.get("o", 0), 4),
                    round(r.get("h", 0), 4),
                    round(r.get("l", 0), 4),
                    round(r.get("c", 0), 4),
                    int(r.get("v", 0)),
                    round(r.get("c", 0), 4),
                ))

        except Exception as e:
            log.warning(f"Polygon: {symbol} failed — {e}")

        # Flush every CHUNK_SIZE rows
        if len(batch_rows) >= CHUNK_SIZE:
            _upsert_price_rows(conn, batch_rows)
            total += len(batch_rows)
            batch_rows = []

        # Polygon free tier: 5 req/min → sleep every 5 symbols
        if i % 5 == 4:
            time.sleep(12)

        if i % 50 == 0:
            log.info(f"  Polygon: {i}/{len(symbols)} symbols — {total:,} rows so far")

    if batch_rows:
        _upsert_price_rows(conn, batch_rows)
        total += len(batch_rows)

    conn.close()
    log.info(f"Polygon fetch complete — {total:,} rows upserted")


# ═══════════════════════════════════════════════════════════════
# SHARED: DB upsert + feature engineering
# ═══════════════════════════════════════════════════════════════

def _upsert_price_rows(conn, rows: list[tuple]) -> None:
    """Batch upsert into stock_prices. Reuses open connection."""
    if not rows:
        return
    cur = conn.cursor()
    sql = """
        INSERT INTO stock_prices (symbol, date, open, high, low, close, volume, adj_close)
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            open      = EXCLUDED.open,
            high      = EXCLUDED.high,
            low       = EXCLUDED.low,
            close     = EXCLUDED.close,
            volume    = EXCLUDED.volume,
            adj_close = EXCLUDED.adj_close
    """
    execute_values(cur, sql, rows, page_size=5000)
    conn.commit()
    cur.close()


def _compute_features_from_db(conn_config: dict, days_back: int = 60) -> None:
    """
    Read recent prices from DB, compute technical features, upsert to stock_features.
    Called after every live update.
    """
    conn = psycopg2.connect(**conn_config)

    since = (datetime.today() - timedelta(days=days_back + 30)).strftime("%Y-%m-%d")
    df = pd.read_sql(
        f"SELECT symbol, date, open, high, low, close, volume FROM stock_prices WHERE date >= '{since}' ORDER BY symbol, date",
        conn,
    )

    if df.empty:
        log.warning("No price data found for feature computation")
        conn.close()
        return

    log.info(f"Computing features for {df['symbol'].nunique()} symbols, {len(df):,} rows")
    feature_rows = []

    for symbol, g in df.groupby("symbol"):
        g = g.sort_values("date").copy()

        # Skip symbols with fewer than 30 rows (not enough for indicators)
        if len(g) < 30:
            continue

        g["r1"]  = g["close"].pct_change(1)
        g["r5"]  = g["close"].pct_change(5)
        g["r20"] = g["close"].pct_change(20)

        delta = g["close"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        g["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

        ema12 = g["close"].ewm(span=12).mean()
        ema26 = g["close"].ewm(span=26).mean()
        g["macd"]   = ema12 - ema26
        g["macd_s"] = g["macd"].ewm(span=9).mean()

        g["vol20"] = g["r1"].rolling(20).std()
        ma20       = g["close"].rolling(20).mean()
        std20      = g["close"].rolling(20).std()
        g["bb_up"] = ma20 + 2 * std20
        g["bb_lo"] = ma20 - 2 * std20

        g["vma20"]   = g["volume"].rolling(20).mean()
        g["vol_rat"] = g["volume"] / g["vma20"].replace(0, np.nan)

        g["target"] = g["close"].pct_change(5).shift(-5)

        # Only keep rows within the update window (last days_back days)
        cutoff = (datetime.today() - timedelta(days=days_back)).date()
        g = g[g["date"] >= cutoff].dropna(subset=["rsi"])

        for r in g.itertuples():
            feature_rows.append((
                symbol, r.date,
                r.r1, r.r5, r.r20,
                r.rsi, r.macd, r.macd_s,
                r.vol20, r.bb_up, r.bb_lo,
                r.vma20, r.vol_rat,
                None, None,
                r.target,
            ))

    if not feature_rows:
        conn.close()
        return

    cur = conn.cursor()
    sql = """
        INSERT INTO stock_features (
            symbol, date, returns_1d, returns_5d, returns_20d,
            rsi_14, macd, macd_signal, volatility_20d,
            bollinger_upper, bollinger_lower, volume_ma_20, volume_ratio,
            avg_sentiment, news_count, target_5d
        ) VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            returns_1d     = EXCLUDED.returns_1d,
            returns_5d     = EXCLUDED.returns_5d,
            volatility_20d = EXCLUDED.volatility_20d,
            rsi_14         = EXCLUDED.rsi_14,
            target_5d      = EXCLUDED.target_5d
    """
    execute_values(cur, sql, feature_rows, page_size=5000)
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Features computed — {len(feature_rows):,} rows upserted")


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Intelligence — price ingestion")
    parser.add_argument(
        "--mode",
        choices=["bulk", "folder", "live", "polygon"],
        required=True,
        help=(
            "bulk    → load Kaggle single CSV (camnugent/sandp500)\n"
            "folder  → load Kaggle folder of .txt files (borismarjanovic)\n"
            "live    → daily yfinance update for full S&P 500\n"
            "polygon → daily Polygon.io update for full S&P 500"
        ),
    )
    parser.add_argument("--path",     default="all_stocks_5yr.csv", help="Path to Kaggle CSV or folder")
    parser.add_argument("--days",     default=5,   type=int, help="Days back for live/polygon mode")
    parser.add_argument("--symbols",  default=None, help="Comma-separated symbols (default: full S&P 500)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None

    if args.mode == "bulk":
        load_kaggle_bulk(args.path)
        _compute_features_from_db(DB_CONFIG, days_back=365 * 5)

    elif args.mode == "folder":
        load_kaggle_folder(args.path)
        _compute_features_from_db(DB_CONFIG, days_back=365 * 20)

    elif args.mode == "live":
        fetch_live_yfinance(symbols=symbols, days_back=args.days)

    elif args.mode == "polygon":
        fetch_polygon(symbols=symbols, days_back=args.days)

    log.info("Done ✓")
