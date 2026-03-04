"""
FraudSense — Model Training Script
=====================================
Trains two models:
  1. Isolation Forest  → anomaly/outlier detection (unsupervised)
  2. Logistic Regression → risk probability scoring (supervised)

Dataset: Kaggle Credit Card Fraud Detection
  → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  → Download creditcard.csv and place it in fraudsense/data/

If you don't have the dataset yet, this script also generates
synthetic training data so you can test the full pipeline immediately.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)

# ─── PATHS ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "../data/creditcard.csv")
MODEL_DIR  = BASE_DIR  # save models here


# ─── SYNTHETIC DATA (fallback if no Kaggle CSV) ───────────────────────────────
def generate_synthetic_data(n_samples=10000, fraud_ratio=0.02):
    """
    Generates realistic synthetic transaction data for testing.
    Features mirror the engineered features used at inference time.
    """
    print("⚠  No dataset found — generating synthetic data for demonstration.")
    rng = np.random.default_rng(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def legit():
        return {
            "amount":            rng.lognormal(4.0, 1.2, n_legit),   # ~$55 median
            "hour":              rng.integers(8, 21, n_legit),        # daytime
            "velocity_1min":     rng.integers(1, 3, n_legit),
            "velocity_5min":     rng.integers(1, 5, n_legit),
            "velocity_1hr":      rng.integers(1, 15, n_legit),
            "amount_zscore":     rng.normal(0, 1, n_legit),
            "geo_risk_score":    rng.uniform(0, 0.3, n_legit),
            "is_new_device":     rng.choice([0, 1], n_legit, p=[0.95, 0.05]),
            "is_new_location":   rng.choice([0, 1], n_legit, p=[0.90, 0.10]),
            "time_since_last_tx":rng.exponential(3600, n_legit),      # seconds
            "is_foreign":        rng.choice([0, 1], n_legit, p=[0.85, 0.15]),
            "is_vpn":            rng.choice([0, 1], n_legit, p=[0.98, 0.02]),
            "label":             np.zeros(n_legit, dtype=int),
        }

    def fraud():
        return {
            "amount":            rng.lognormal(6.5, 1.5, n_fraud),    # large amounts
            "hour":              rng.choice([0,1,2,3,22,23], n_fraud),# odd hours
            "velocity_1min":     rng.integers(5, 20, n_fraud),        # burst
            "velocity_5min":     rng.integers(10, 40, n_fraud),
            "velocity_1hr":      rng.integers(20, 80, n_fraud),
            "amount_zscore":     rng.uniform(3, 8, n_fraud),          # far from baseline
            "geo_risk_score":    rng.uniform(0.6, 1.0, n_fraud),
            "is_new_device":     rng.choice([0, 1], n_fraud, p=[0.2, 0.8]),
            "is_new_location":   rng.choice([0, 1], n_fraud, p=[0.1, 0.9]),
            "time_since_last_tx":rng.exponential(30, n_fraud),        # rapid
            "is_foreign":        rng.choice([0, 1], n_fraud, p=[0.2, 0.8]),
            "is_vpn":            rng.choice([0, 1], n_fraud, p=[0.4, 0.6]),
            "label":             np.ones(n_fraud, dtype=int),
        }

    df = pd.concat([
        pd.DataFrame(legit()),
        pd.DataFrame(fraud()),
    ]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# ─── FEATURE COLUMNS ──────────────────────────────────────────────────────────
FEATURE_COLS = [
    "amount", "hour", "velocity_1min", "velocity_5min", "velocity_1hr",
    "amount_zscore", "geo_risk_score", "is_new_device", "is_new_location",
    "time_since_last_tx", "is_foreign", "is_vpn",
]


# ─── LOAD OR GENERATE DATA ────────────────────────────────────────────────────
def load_data():
    if os.path.exists(DATA_PATH):
        print(f"✓ Loading Kaggle dataset from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)

        # Kaggle dataset has V1–V28 PCA features + Amount + Class
        # Map to our feature names for consistency
        df = df.rename(columns={"Amount": "amount", "Class": "label"})
        df["hour"]              = (df["Time"] % 86400) // 3600
        df["velocity_1min"]     = np.abs(df.get("V1", 0))
        df["velocity_5min"]     = np.abs(df.get("V2", 0))
        df["velocity_1hr"]      = np.abs(df.get("V3", 0))
        df["amount_zscore"]     = (df["amount"] - df["amount"].mean()) / df["amount"].std()
        df["geo_risk_score"]    = np.clip(np.abs(df.get("V4", 0)) / 10, 0, 1)
        df["is_new_device"]     = (np.abs(df.get("V5", 0)) > 2).astype(int)
        df["is_new_location"]   = (np.abs(df.get("V6", 0)) > 2).astype(int)
        df["time_since_last_tx"]= df["Time"].diff().fillna(0).clip(0)
        df["is_foreign"]        = (np.abs(df.get("V7", 0)) > 1.5).astype(int)
        df["is_vpn"]            = (np.abs(df.get("V8", 0)) > 2).astype(int)
    else:
        df = generate_synthetic_data()

    return df


# ─── TRAIN ────────────────────────────────────────────────────────────────────
def train():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  FraudSense — Model Training")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    df = load_data()
    X  = df[FEATURE_COLS]
    y  = df["label"]

    print(f"  Dataset: {len(df):,} transactions | "
          f"{y.sum():,} fraud ({y.mean()*100:.2f}%)\n")

    # ── Scale features ────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 1. Isolation Forest ───────────────────────────────────────────────────
    print("  [1/2] Training Isolation Forest (anomaly detection)...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=float(y.mean()),  # match actual fraud rate
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_train)

    # Isolation Forest gives -1 (anomaly) or 1 (normal)
    iso_preds = iso.predict(X_test)
    iso_binary = (iso_preds == -1).astype(int)  # convert to 0/1
    print(f"     Anomaly detection rate: {iso_binary.mean()*100:.1f}%")

    # ── 2. Logistic Regression ────────────────────────────────────────────────
    print("  [2/2] Training Logistic Regression (risk scoring)...")
    lr = LogisticRegression(
        class_weight="balanced",  # handles class imbalance
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    lr.fit(X_train, y_train)
    lr_preds  = lr.predict(X_test)
    lr_proba  = lr.predict_proba(X_test)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    precision = precision_score(y_test, lr_preds)
    recall    = recall_score(y_test, lr_preds)
    f1        = f1_score(y_test, lr_preds)
    fp_rate   = 1 - precision

    print(f"\n  ┌─ Model Metrics ─────────────────────┐")
    print(f"  │  Precision   : {precision*100:.1f}%                 │")
    print(f"  │  Recall      : {recall*100:.1f}%                 │")
    print(f"  │  F1 Score    : {f1*100:.1f}%                 │")
    print(f"  │  False Pos.  : {fp_rate*100:.1f}%                  │")
    print(f"  └─────────────────────────────────────┘")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, lr_preds, target_names=["Legit","Fraud"]))

    # ── Save models ───────────────────────────────────────────────────────────
    joblib.dump(iso,    os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(lr,     os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    # Save feature column order (critical for consistent inference)
    pd.Series(FEATURE_COLS).to_csv(
        os.path.join(MODEL_DIR, "feature_cols.csv"), index=False
    )

    print("\n  ✓ Models saved to fraudsense/models/")
    print("     → isolation_forest.joblib")
    print("     → logistic_regression.joblib")
    print("     → scaler.joblib")
    print("     → feature_cols.csv\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp_rate,
    }


if __name__ == "__main__":
    train()


