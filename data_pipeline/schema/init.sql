-- ============================================================
-- Market Intelligence Platform — Database Schema
-- Run once to initialize your PostgreSQL database
-- ============================================================

-- Raw daily OHLCV prices for each stock
CREATE TABLE IF NOT EXISTS stock_prices (
    id          SERIAL PRIMARY KEY,
    symbol      VARCHAR(10)    NOT NULL,
    date        DATE           NOT NULL,
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4),
    volume      BIGINT,
    adj_close   NUMERIC(12,4),
    created_at  TIMESTAMP      DEFAULT NOW(),
    UNIQUE (symbol, date)
);

-- Financial news headlines + sentiment scores
CREATE TABLE IF NOT EXISTS news_articles (
    id              SERIAL PRIMARY KEY,
    headline        TEXT           NOT NULL,
    source          VARCHAR(100),
    url             TEXT,
    published_at    TIMESTAMP,
    symbols_mentioned TEXT[],          -- e.g. {AAPL, MSFT}
    raw_sentiment   NUMERIC(5,4),      -- FinBERT score (-1 to 1)
    sentiment_label VARCHAR(10),       -- positive / negative / neutral
    created_at      TIMESTAMP      DEFAULT NOW()
);

-- Macro economic indicators from FRED
CREATE TABLE IF NOT EXISTS macro_indicators (
    id          SERIAL PRIMARY KEY,
    indicator   VARCHAR(50)    NOT NULL,  -- e.g. CPI, FEDFUNDS, UNRATE
    date        DATE           NOT NULL,
    value       NUMERIC(12,4),
    created_at  TIMESTAMP      DEFAULT NOW(),
    UNIQUE (indicator, date)
);

-- Engineered features for ML models (pre-computed daily)
CREATE TABLE IF NOT EXISTS stock_features (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(10)    NOT NULL,
    date            DATE           NOT NULL,
    -- Price features
    returns_1d      NUMERIC(8,6),
    returns_5d      NUMERIC(8,6),
    returns_20d     NUMERIC(8,6),
    -- Momentum indicators
    rsi_14          NUMERIC(8,4),
    macd            NUMERIC(10,6),
    macd_signal     NUMERIC(10,6),
    -- Volatility
    volatility_20d  NUMERIC(8,6),
    bollinger_upper NUMERIC(12,4),
    bollinger_lower NUMERIC(12,4),
    -- Volume
    volume_ma_20    NUMERIC(16,2),
    volume_ratio    NUMERIC(8,4),
    -- Sentiment (joined from news)
    avg_sentiment   NUMERIC(5,4),
    news_count      INTEGER,
    -- Target variable for ML (next 5-day return)
    target_5d       NUMERIC(8,6),
    created_at      TIMESTAMP      DEFAULT NOW(),
    UNIQUE (symbol, date)
);

-- Anomaly detection results
CREATE TABLE IF NOT EXISTS anomalies (
    id              SERIAL PRIMARY KEY,
    symbol          VARCHAR(10)    NOT NULL,
    date            DATE           NOT NULL,
    anomaly_type    VARCHAR(50),       -- price_spike, volume_surge, sentiment_shift
    score           NUMERIC(8,4),     -- isolation forest score
    description     TEXT,
    created_at      TIMESTAMP      DEFAULT NOW()
);

-- Useful indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_prices_symbol_date   ON stock_prices (symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_news_published        ON news_articles (published_at DESC);
CREATE INDEX IF NOT EXISTS idx_features_symbol_date  ON stock_features (symbol, date DESC);
CREATE INDEX IF NOT EXISTS idx_anomalies_symbol_date ON anomalies (symbol, date DESC);

-- ============================================================
-- Useful analytical views (these are what interviewers love)
-- ============================================================

-- 30-day rolling avg sentiment per stock
CREATE OR REPLACE VIEW vw_stock_sentiment_30d AS
SELECT
    symbol,
    date,
    avg_sentiment,
    AVG(avg_sentiment) OVER (
        PARTITION BY symbol
        ORDER BY date
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS sentiment_ma_30d,
    news_count
FROM stock_features
WHERE avg_sentiment IS NOT NULL;

-- Daily portfolio risk snapshot
CREATE OR REPLACE VIEW vw_portfolio_risk AS
SELECT
    symbol,
    date,
    close,
    returns_1d,
    volatility_20d,
    rsi_14,
    avg_sentiment,
    CASE
        WHEN volatility_20d > 0.04 AND avg_sentiment < -0.2 THEN 'HIGH'
        WHEN volatility_20d > 0.02 OR avg_sentiment < 0     THEN 'MEDIUM'
        ELSE 'LOW'
    END AS risk_level
FROM stock_features
JOIN stock_prices USING (symbol, date);