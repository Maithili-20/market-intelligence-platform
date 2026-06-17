"""
main.py
-------
FastAPI backend for the Market Intelligence Platform.

Endpoints:
  GET  /                        - Health check
  GET  /predict/{symbol}        - XGBoost price prediction + SHAP
  GET  /sentiment               - Latest news sentiment
  GET  /anomalies               - Detected market anomalies
  GET  /risk-score/{symbol}     - Composite risk score
  GET  /stocks                  - List of tracked stocks
  GET  /macro                   - Latest macro indicators
  WS   /ws/prices               - WebSocket live price stream

Swagger UI: http://localhost:8000/docs
"""

import os
import json
import asyncio
import logging
from datetime import datetime, date
from typing import Optional

import psycopg2
import pandas as pd
import joblib
import numpy as np
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Database config ────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", 5432),
    "dbname":   os.getenv("DB_NAME", "market_intel"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# ── Load ML models once at startup ────────────────────────────────────────────
MODEL_PATH  = "ml_engine/xgboost_model.pkl"
SCALER_PATH = "ml_engine/scaler.pkl"

try:
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    log.info("ML models loaded successfully")
except Exception as e:
    log.warning(f"Could not load ML models: {e}")
    model  = None
    scaler = None

FEATURE_COLS = [
    "returns_1d", "returns_5d", "returns_20d",
    "rsi_14", "macd", "macd_signal",
    "volatility_20d", "bollinger_upper", "bollinger_lower",
    "volume_ratio", "avg_sentiment", "news_count",
]

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Market Intelligence Platform API",
    description="Real-time stock market intelligence with ML predictions, sentiment analysis, and anomaly detection.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper ─────────────────────────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(**DB_CONFIG)

def row_to_dict(cursor, row):
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))

def serialize(obj):
    """Make objects JSON-serializable."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")

# ── Response models ────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    db_connected: bool
    models_loaded: bool

class PredictionResponse(BaseModel):
    symbol: str
    predicted_return_5d: float
    confidence: str
    top_feature: str
    current_rsi: Optional[float]
    current_macd: Optional[float]
    signal: str

class RiskResponse(BaseModel):
    symbol: str
    risk_score: float
    risk_level: str
    components: dict

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API health and connectivity."""
    db_ok = False
    try:
        conn = get_db()
        conn.close()
        db_ok = True
    except:
        pass

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "db_connected": db_ok,
        "models_loaded": model is not None,
    }


@app.get("/stocks", tags=["Market Data"])
def list_stocks():
    """List all tracked stock symbols."""
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT symbol, COUNT(*) as days_tracked,
               MAX(date) as last_updated,
               AVG(close) as avg_price
        FROM stock_prices
        GROUP BY symbol
        ORDER BY symbol
    """)
    rows = [row_to_dict(cur, r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"count": len(rows), "stocks": json.loads(json.dumps(rows, default=serialize))}


@app.get("/predict/{symbol}", tags=["ML Predictions"])
def predict(symbol: str):
    """
    XGBoost 5-day return prediction for a given stock symbol.
    Returns prediction, confidence, top SHAP feature, and trading signal.
    """
    symbol = symbol.upper()

    # Try stored prediction first
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT p.predicted_return_5d, p.shap_top_feature, p.shap_top_value,
               f.rsi_14, f.macd, f.volatility_20d, f.returns_1d
        FROM ml_predictions p
        JOIN stock_features f USING (symbol, date)
        WHERE p.symbol = %s
        ORDER BY p.date DESC
        LIMIT 1
    """, (symbol,))
    row = cur.fetchone()

    if not row:
        # Try live prediction
        cur.execute("""
            SELECT returns_1d, returns_5d, returns_20d, rsi_14, macd, macd_signal,
                   volatility_20d, bollinger_upper, bollinger_lower,
                   volume_ratio, avg_sentiment, news_count
            FROM stock_features
            WHERE symbol = %s
            ORDER BY date DESC LIMIT 1
        """, (symbol,))
        feat_row = cur.fetchone()
        cur.close()
        conn.close()

        if not feat_row:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        if model is None:
            raise HTTPException(status_code=503, detail="ML model not loaded")

        features = pd.DataFrame([feat_row], columns=FEATURE_COLS).fillna(0)
        X_scaled = pd.DataFrame(scaler.transform(features), columns=FEATURE_COLS)
        pred = float(model.predict(X_scaled)[0])
        top_feature = "unknown"
    else:
        pred, top_feature, top_val, rsi, macd, vol, ret1d = row
        cur.close()
        conn.close()
        pred = float(pred)

    # Generate signal
    if pred > 0.02:
        signal = "BUY"
        confidence = "High" if pred > 0.05 else "Medium"
    elif pred < -0.02:
        signal = "SELL"
        confidence = "High" if pred < -0.05 else "Medium"
    else:
        signal = "HOLD"
        confidence = "Low"

    return {
        "symbol": symbol,
        "predicted_return_5d": round(pred * 100, 2),  # as percentage
        "predicted_return_5d_raw": round(pred, 6),
        "confidence": confidence,
        "top_feature": top_feature,
        "signal": signal,
        "current_rsi": round(float(rsi), 2) if row and rsi else None,
        "current_macd": round(float(macd), 4) if row and macd else None,
    }


@app.get("/sentiment", tags=["Sentiment"])
def get_sentiment(limit: int = 20):
    """Latest news articles with FinBERT sentiment scores."""
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT headline, source, published_at, raw_sentiment,
               sentiment_label, symbols_mentioned
        FROM news_articles
        WHERE raw_sentiment IS NOT NULL
        ORDER BY published_at DESC
        LIMIT %s
    """, (limit,))
    rows = [row_to_dict(cur, r) for r in cur.fetchall()]
    cur.close()
    conn.close()

    # Summary stats
    if rows:
        scores = [r["raw_sentiment"] for r in rows if r["raw_sentiment"]]
        avg    = sum(scores) / len(scores) if scores else 0
        labels = [r["sentiment_label"] for r in rows]
        summary = {
            "avg_sentiment": round(float(avg), 4),
            "positive_pct": round(labels.count("positive") / len(labels) * 100, 1),
            "negative_pct": round(labels.count("negative") / len(labels) * 100, 1),
            "neutral_pct":  round(labels.count("neutral")  / len(labels) * 100, 1),
            "overall": "Bullish" if avg > 0.1 else "Bearish" if avg < -0.1 else "Neutral",
        }
    else:
        summary = {}

    return {
        "summary": summary,
        "articles": json.loads(json.dumps(rows, default=serialize)),
    }


@app.get("/anomalies", tags=["Anomalies"])
def get_anomalies(limit: int = 20, anomaly_type: Optional[str] = None):
    """Market anomalies detected by Isolation Forest."""
    conn = get_db()
    cur  = conn.cursor()

    if anomaly_type:
        cur.execute("""
            SELECT symbol, date, anomaly_type, score, description
            FROM anomalies
            WHERE anomaly_type = %s
            ORDER BY score ASC
            LIMIT %s
        """, (anomaly_type, limit))
    else:
        cur.execute("""
            SELECT symbol, date, anomaly_type, score, description
            FROM anomalies
            ORDER BY score ASC
            LIMIT %s
        """, (limit,))

    rows = [row_to_dict(cur, r) for r in cur.fetchall()]
    cur.close()
    conn.close()

    return {
        "count": len(rows),
        "anomalies": json.loads(json.dumps(rows, default=serialize)),
    }


@app.get("/risk-score/{symbol}", tags=["Risk"])
def risk_score(symbol: str):
    """
    Composite risk score for a stock (0-100, higher = riskier).
    Combines volatility, RSI extremes, anomaly count, and sentiment.
    """
    symbol = symbol.upper()
    conn = get_db()
    cur  = conn.cursor()

    # Get latest features
    cur.execute("""
        SELECT volatility_20d, rsi_14, avg_sentiment, returns_1d, volume_ratio
        FROM stock_features
        WHERE symbol = %s
        ORDER BY date DESC LIMIT 1
    """, (symbol,))
    row = cur.fetchone()

    if not row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    vol, rsi, sentiment, ret1d, vol_ratio = row

    # Anomaly count in last 30 days
    cur.execute("""
        SELECT COUNT(*) FROM anomalies
        WHERE symbol = %s
        AND date >= CURRENT_DATE - INTERVAL '30 days'
    """, (symbol,))
    anomaly_count = cur.fetchone()[0]
    cur.close()
    conn.close()

    # Score components (0-100 each)
    vol_score      = min(float(vol or 0) * 1000, 100)
    rsi_score      = abs(float(rsi or 50) - 50) * 2  # extremes = risky
    sentiment_score = max(0, -float(sentiment or 0) * 100)  # negative = risky
    anomaly_score  = min(anomaly_count * 20, 100)

    composite = (vol_score * 0.4 + rsi_score * 0.3 +
                 sentiment_score * 0.2 + anomaly_score * 0.1)
    composite = round(min(composite, 100), 2)

    if composite < 30:
        risk_level = "Low"
    elif composite < 60:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return {
        "symbol": symbol,
        "risk_score": composite,
        "risk_level": risk_level,
        "components": {
            "volatility_score": round(vol_score, 2),
            "rsi_score":        round(rsi_score, 2),
            "sentiment_score":  round(sentiment_score, 2),
            "anomaly_score":    round(anomaly_score, 2),
        },
    }


@app.get("/macro", tags=["Macro"])
def get_macro():
    """Latest macroeconomic indicators."""
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (indicator)
            indicator, value, date
        FROM macro_indicators
        ORDER BY indicator, date DESC
    """)
    rows = [row_to_dict(cur, r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return {"indicators": json.loads(json.dumps(rows, default=serialize))}


# ── WebSocket live price stream ────────────────────────────────────────────────
@app.websocket("/ws/prices")
async def websocket_prices(websocket: WebSocket):
    """
    WebSocket endpoint — streams latest stock prices every 5 seconds.
    Connect with: ws://localhost:8000/ws/prices
    """
    await websocket.accept()
    log.info("WebSocket client connected")

    try:
        while True:
            conn = get_db()
            cur  = conn.cursor()
            cur.execute("""
                SELECT DISTINCT ON (symbol)
                    symbol, close, volume, date
                FROM stock_prices
                ORDER BY symbol, date DESC
                LIMIT 20
            """)
            rows = [row_to_dict(cur, r) for r in cur.fetchall()]
            cur.close()
            conn.close()

            payload = {
                "type": "price_update",
                "timestamp": datetime.now().isoformat(),
                "data": json.loads(json.dumps(rows, default=serialize)),
            }
            await websocket.send_json(payload)
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
