"""
fetch_news.py
-------------
Pulls financial news headlines from NewsAPI and stores them in PostgreSQL.
Sentiment scoring (FinBERT) happens in Week 2 — this script just stores raw text.

Run manually:   python fetch_news.py
Scheduled by:   Airflow DAG (market_data_pipeline)
"""

import os
import logging
import re
from datetime import datetime, timedelta

import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")   # Free key at newsapi.org
BASE_URL     = "https://newsapi.org/v2/everything"

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME", "market_intel"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Map query keywords → stock symbols for tagging
SYMBOL_KEYWORDS = {
    "AAPL":       ["Apple", "AAPL", "iPhone", "Tim Cook"],
    "MSFT":       ["Microsoft", "MSFT", "Azure", "Satya Nadella"],
    "GOOGL":      ["Google", "Alphabet", "GOOGL", "Sundar Pichai"],
    "AMZN":       ["Amazon", "AMZN", "AWS", "Andy Jassy"],
    "META":       ["Meta", "Facebook", "Instagram", "Zuckerberg"],
    "NVDA":       ["Nvidia", "NVDA", "Jensen Huang", "GPU"],
    "TSLA":       ["Tesla", "TSLA", "Elon Musk"],
    "RELIANCE.NS":["Reliance", "Mukesh Ambani", "Jio"],
    "TCS.NS":     ["TCS", "Tata Consultancy"],
    "INFY.NS":    ["Infosys", "INFY"],
}

FINANCIAL_QUERIES = [
    "stock market earnings",
    "Federal Reserve interest rates",
    "tech stocks quarterly results",
    "India stock market NSE BSE",
    "cryptocurrency bitcoin",
]


def fetch_news(query: str, days_back: int = 1) -> list[dict]:
    """Fetch articles from NewsAPI for a given query."""
    if not NEWS_API_KEY:
        log.error("NEWS_API_KEY not set — get a free key at newsapi.org")
        return []

    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": 50,
        "apiKey":   NEWS_API_KEY,
    }

    resp = requests.get(BASE_URL, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    log.info(f"  '{query}' → {len(articles)} articles")
    return articles


def detect_symbols(text: str) -> list[str]:
    """Tag which stocks are mentioned in a headline."""
    text_upper = text.upper()
    found = []
    for symbol, keywords in SYMBOL_KEYWORDS.items():
        if any(kw.upper() in text_upper for kw in keywords):
            found.append(symbol)
    return found


def upsert_articles(articles: list[dict]) -> None:
    """Store raw news articles in PostgreSQL."""
    if not articles:
        return

    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    rows = []
    seen_urls = set()

    for a in articles:
        url      = a.get("url", "")
        headline = a.get("title", "") or ""
        if not headline or url in seen_urls:
            continue
        seen_urls.add(url)

        published_raw = a.get("publishedAt")
        try:
            published_at = datetime.strptime(published_raw, "%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            published_at = None

        symbols = detect_symbols(headline + " " + (a.get("description") or ""))

        rows.append((
            headline,
            a.get("source", {}).get("name"),
            url,
            published_at,
            symbols if symbols else None,
        ))

    sql = """
        INSERT INTO news_articles (headline, source, url, published_at, symbols_mentioned)
        VALUES %s
        ON CONFLICT DO NOTHING
    """
    execute_values(cur, sql, rows)
    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Inserted {len(rows)} new articles")


if __name__ == "__main__":
    all_articles = []
    for query in FINANCIAL_QUERIES:
        try:
            articles = fetch_news(query, days_back=1)
            all_articles.extend(articles)
        except Exception as e:
            log.error(f"Failed query '{query}': {e}")

    # Deduplicate across queries before inserting
    seen = set()
    unique = []
    for a in all_articles:
        url = a.get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(a)

    log.info(f"Total unique articles: {len(unique)}")
    upsert_articles(unique)
    log.info("News pipeline complete ✓")
