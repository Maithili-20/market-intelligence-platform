"""
sentiment_finbert.py
--------------------
Scores all news articles in the database using FinBERT —
a BERT model fine-tuned specifically on financial text.

Much more accurate than VADER/TextBlob for financial news.
Output: updates news_articles.raw_sentiment and sentiment_label
        updates stock_features.avg_sentiment and news_count
"""

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 16


def load_finbert():
    """Download and load FinBERT model (cached after first download)."""
    log.info("Loading FinBERT model (first run downloads ~440MB, cached after)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log.info(f"FinBERT loaded on {device}")
    return tokenizer, model, device


def score_headlines(headlines: list[str], tokenizer, model, device) -> list[tuple[float, str]]:
    """
    Score a list of headlines.
    Returns list of (score, label) where:
      score: -1.0 (very negative) to +1.0 (very positive)
      label: 'positive', 'negative', or 'neutral'
    """
    results = []

    for i in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()

        # FinBERT label order: positive=0, negative=1, neutral=2
        for prob in probs:
            pos, neg, neu = prob[0], prob[1], prob[2]
            # Composite score: positive pulls toward +1, negative toward -1
            score = float(pos - neg)
            label = ["positive", "negative", "neutral"][prob.argmax()]
            results.append((round(score, 4), label))

    return results


def update_article_sentiments(scores: list[tuple], article_ids: list[int]) -> None:
    """Write sentiment scores back to news_articles table."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    for (score, label), article_id in zip(scores, article_ids):
        cur.execute(
            "UPDATE news_articles SET raw_sentiment=%s, sentiment_label=%s WHERE id=%s",
            (score, label, article_id)
        )

    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Updated {len(scores)} articles with sentiment scores")


def update_feature_sentiments() -> None:
    """
    Join news sentiment to stock_features:
    For each (symbol, date), compute average sentiment from articles
    that mention the symbol and were published on that date.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Get all articles with sentiment scores
    articles = pd.read_sql("""
        SELECT id, symbols_mentioned, published_at::date as pub_date, raw_sentiment
        FROM news_articles
        WHERE raw_sentiment IS NOT NULL
    """, conn)

    if articles.empty:
        log.warning("No scored articles found")
        conn.close()
        return

    # Explode symbols array and group by symbol+date
    rows = []
    for _, row in articles.iterrows():
        symbols = row["symbols_mentioned"]
        if not symbols:
            continue
        for symbol in symbols:
            rows.append({
                "symbol":        symbol,
                "date":          row["pub_date"],
                "raw_sentiment": row["raw_sentiment"],
            })

    if not rows:
        log.warning("No symbol-tagged articles found")
        conn.close()
        return

    df = pd.DataFrame(rows)
    grouped = df.groupby(["symbol", "date"]).agg(
        avg_sentiment=("raw_sentiment", "mean"),
        news_count=("raw_sentiment", "count"),
    ).reset_index()

    # Update stock_features
    updated = 0
    for _, r in grouped.iterrows():
        cur.execute("""
            UPDATE stock_features
            SET avg_sentiment=%s, news_count=%s
            WHERE symbol=%s AND date=%s
        """, (round(float(r.avg_sentiment), 4), int(r.news_count), r.symbol, r.date))
        updated += cur.rowcount

    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Updated sentiment for {updated} stock_feature rows")


def print_sentiment_summary(scores: list[tuple]) -> None:
    """Print a quick summary of sentiment distribution."""
    labels = [s[1] for s in scores]
    pos = labels.count("positive")
    neg = labels.count("negative")
    neu = labels.count("neutral")
    avg = sum(s[0] for s in scores) / len(scores)

    log.info("─" * 40)
    log.info(f"Sentiment summary ({len(scores)} articles):")
    log.info(f"  Positive : {pos} ({pos/len(scores)*100:.0f}%)")
    log.info(f"  Negative : {neg} ({neg/len(scores)*100:.0f}%)")
    log.info(f"  Neutral  : {neu} ({neu/len(scores)*100:.0f}%)")
    log.info(f"  Avg score: {avg:.4f}")
    log.info("─" * 40)


if __name__ == "__main__":
    # Load all unscored articles
    conn = psycopg2.connect(**DB_CONFIG)
    df   = pd.read_sql("""
        SELECT id, headline
        FROM news_articles
        WHERE raw_sentiment IS NULL
        ORDER BY published_at DESC
    """, conn)
    conn.close()

    if df.empty:
        log.info("No unscored articles — re-scoring all articles")
        conn = psycopg2.connect(**DB_CONFIG)
        df   = pd.read_sql("SELECT id, headline FROM news_articles ORDER BY published_at DESC", conn)
        conn.close()

    log.info(f"Scoring {len(df)} headlines with FinBERT...")

    tokenizer, model, device = load_finbert()
    scores = score_headlines(df["headline"].tolist(), tokenizer, model, device)

    update_article_sentiments(scores, df["id"].tolist())
    print_sentiment_summary(scores)
    update_feature_sentiments()

    log.info("FinBERT sentiment pipeline complete ✓")
