"""
predict_xgboost.py
------------------
Trains an XGBoost model to predict 5-day stock returns
using 15+ engineered features from stock_features table.

Also generates SHAP values to explain each prediction —
this is what separates a student project from a professional one.
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

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import shap
import joblib

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

# Features used for prediction
FEATURE_COLS = [
    "returns_1d", "returns_5d", "returns_20d",
    "rsi_14", "macd", "macd_signal",
    "volatility_20d", "bollinger_upper", "bollinger_lower",
    "volume_ratio", "avg_sentiment", "news_count",
]

TARGET_COL = "target_5d"
MODEL_PATH = "ml_engine/xgboost_model.pkl"
SCALER_PATH = "ml_engine/scaler.pkl"


def load_features() -> pd.DataFrame:
    """Load feature data from PostgreSQL."""
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql(f"""
        SELECT
            symbol, date,
            {', '.join(FEATURE_COLS)},
            {TARGET_COL}
        FROM stock_features
        WHERE target_5d IS NOT NULL
          AND returns_1d IS NOT NULL
        ORDER BY date ASC
    """, conn)
    conn.close()

    log.info(f"Loaded {len(df):,} rows from stock_features")
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean and prepare features for training."""
    df = df.copy()

    # Fill sentiment NaN with 0 (neutral)
    df["avg_sentiment"] = df["avg_sentiment"].fillna(0)
    df["news_count"]    = df["news_count"].fillna(0)

    # Drop rows with missing features
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # Remove extreme outliers in target (> 50% move in 5 days)
    df = df[df[TARGET_COL].abs() < 0.5]

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    log.info(f"After preprocessing: {len(df):,} rows, {len(FEATURE_COLS)} features")
    return X, y, df


def train_model(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train XGBoost with time-series cross-validation."""

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    # Time-series split — never use future data to predict past
    tscv = TimeSeriesSplit(n_splits=5)

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # Cross-validation scores
    cv_maes = []
    cv_r2s  = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx],         y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        mae   = mean_absolute_error(y_val, preds)
        r2    = r2_score(y_val, preds)
        cv_maes.append(mae)
        cv_r2s.append(r2)
        log.info(f"  Fold {fold+1}: MAE={mae:.4f}, R²={r2:.4f}")

    log.info(f"CV MAE: {np.mean(cv_maes):.4f} ± {np.std(cv_maes):.4f}")
    log.info(f"CV R²:  {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")

    # Final model on all data
    model.fit(X_scaled, y, verbose=False)

    return model, scaler


def compute_shap(model, X_scaled: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SHAP values — explains WHY the model makes each prediction.
    This is what banks and hedge funds use to understand ML models.
    """
    log.info("Computing SHAP values...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    shap_df = pd.DataFrame(shap_values, columns=X_scaled.columns)

    # Feature importance from SHAP (mean absolute SHAP value)
    importance = shap_df.abs().mean().sort_values(ascending=False)

    log.info("Top feature importances (SHAP):")
    for feat, imp in importance.items():
        log.info(f"  {feat:25s}: {imp:.6f}")

    return shap_df, importance


def save_predictions_to_db(df: pd.DataFrame, model, scaler, shap_df: pd.DataFrame) -> None:
    """Store model predictions in a new table for the API to serve."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur  = conn.cursor()

    # Create predictions table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_predictions (
            id          SERIAL PRIMARY KEY,
            symbol      VARCHAR(10) NOT NULL,
            date        DATE        NOT NULL,
            predicted_return_5d NUMERIC(8,6),
            shap_top_feature    VARCHAR(50),
            shap_top_value      NUMERIC(8,6),
            created_at  TIMESTAMP DEFAULT NOW(),
            UNIQUE (symbol, date)
        )
    """)

    X = df[FEATURE_COLS].fillna(0)
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    preds = model.predict(X_scaled)

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        top_feat = shap_df.iloc[i].abs().idxmax()
        top_val  = float(shap_df.iloc[i][top_feat])
        rows.append((
            row["symbol"],
            row["date"],
            round(float(preds[i]), 6),
            top_feat,
            round(top_val, 6),
        ))

    execute_values(cur, """
        INSERT INTO ml_predictions (symbol, date, predicted_return_5d, shap_top_feature, shap_top_value)
        VALUES %s
        ON CONFLICT (symbol, date) DO UPDATE SET
            predicted_return_5d = EXCLUDED.predicted_return_5d,
            shap_top_feature    = EXCLUDED.shap_top_feature,
            shap_top_value      = EXCLUDED.shap_top_value
    """, rows)

    conn.commit()
    cur.close()
    conn.close()
    log.info(f"Saved {len(rows):,} predictions to ml_predictions table")


if __name__ == "__main__":
    # Load data
    df = load_features()

    if len(df) < 100:
        log.error(f"Not enough data to train ({len(df)} rows). Need at least 100.")
        log.error("Run fetch_prices.py --mode live --days 90 first.")
        exit(1)

    # Preprocess
    X, y, df_clean = preprocess(df)

    # Train
    log.info("Training XGBoost model...")
    model, scaler = train_model(X, y)

    # SHAP
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )
    shap_df, importance = compute_shap(model, X_scaled)

    # Save model
    os.makedirs("ml_engine", exist_ok=True)
    joblib.dump(model,  MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Model saved to {MODEL_PATH}")

    # Save predictions to DB
    save_predictions_to_db(df_clean, model, scaler, shap_df)

    log.info("XGBoost + SHAP pipeline complete ✓")
