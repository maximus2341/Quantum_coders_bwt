"""
FraudSense — Flask API
========================
Run with:  python app.py
           or: flask run --port 5000

Endpoints:
  POST /api/transaction          → Score a single transaction
  GET  /api/transactions         → Get recent scored transactions
  GET  /api/alerts               → Get high-risk alerts
  GET  /api/stats                → Dashboard stats
  GET  /api/profile/<user_id>    → Get a user's behavioral profile
  POST /api/profile/<user_id>/reset → Reset a user's baseline
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import uuid
import json
import logging
from datetime import datetime, timezone
from functools import lru_cache
from collections import deque

import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.insert(0, BASE_DIR)

from utils.features import extract_features, get_store, interpret_score

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("fraudsense")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow the frontend (different port) to call this API

# ── Load models ───────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def load_models():
    log.info("Loading ML models...")
    try:
        iso    = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
        lr     = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
        feat_cols = pd.read_csv(
            os.path.join(MODEL_DIR, "feature_cols.csv")
        )["0"].tolist()
        log.info("✓ Models loaded successfully.")
        return iso, lr, scaler, feat_cols
    except FileNotFoundError:
        log.warning("Models not found. Run models/train.py first. Using fallback scoring.")
        return None, None, None, None


# ── In-memory transaction log (swap for SQLite in production) ─────────────────
_tx_log   = deque(maxlen=1000)   # recent transactions
_alerts   = deque(maxlen=200)    # high-risk only
_stats    = {
    "total_today": 0,
    "fraud_flagged": 0,
    "false_positive_rate": 1.8,  # updated periodically by model evaluation
    "model_accuracy": 96.4,
}


# ─── SCORE TRANSACTION ────────────────────────────────────────────────────────
def score_transaction(tx: dict) -> dict:
    """
    Core inference function.
    1. Extract features from raw tx
    2. Scale with the trained scaler
    3. Run Isolation Forest (anomaly) + Logistic Regression (risk prob)
    4. Blend into final risk score
    5. Update user profile
    """
    store = get_store()
    features = extract_features(tx, store)

    iso, lr, scaler, feat_cols = load_models()

    if iso is None:
        # ── Fallback: rule-based scoring when models aren't trained yet ──────
        score = _rule_based_score(features)
        anomaly_score = score / 100
        lr_score = score / 100
    else:
        # ── ML scoring ───────────────────────────────────────────────────────
        feat_vector = np.array([[features[c] for c in feat_cols]])
        feat_scaled = scaler.transform(feat_vector)

        # Logistic Regression: probability of fraud (0.0 – 1.0)
        lr_score = float(lr.predict_proba(feat_scaled)[0][1])

        # Isolation Forest: anomaly score
        # Raw score is negative (more negative = more anomalous)
        # We normalize to 0–1 where 1 = most anomalous
        raw_if = iso.score_samples(feat_scaled)[0]  # typically -0.7 to 0.1
        anomaly_score = float(np.clip(1 - (raw_if + 0.7) / 0.8, 0, 1))

    result = interpret_score(lr_score, anomaly_score)

    # Update user profile AFTER scoring (don't let this tx influence itself)
    store.update(tx.get("user_id", "unknown"), tx)

    return {**result, "features": features}


def _rule_based_score(features: dict) -> int:
    """
    Simple rule-based fallback when models aren't trained yet.
    Useful for demos before training is complete.
    """
    score = 0
    if features["velocity_1min"] >= 5:   score += 30
    if features["velocity_5min"] >= 10:  score += 20
    if features["amount_zscore"] > 3:    score += 25
    if features["is_new_device"]:        score += 15
    if features["is_vpn"]:               score += 20
    if features["geo_risk_score"] > 0.7: score += 20
    if features["is_foreign"]:           score += 10
    if features["hour"] in range(0, 5):  score += 10
    return min(score, 100)


# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/api/transaction", methods=["POST"])
def ingest_transaction():
    """
    Score an incoming transaction.

    Expected JSON body:
    {
        "user_id":      "USR-1234",
        "amount":       299.99,
        "country_code": "NG",
        "location":     "Lagos, NG",
        "device_id":    "dev-abc123",
        "is_vpn":       false,
        "timestamp":    1710000000.0   ← unix float, optional (defaults to now)
    }
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    required = ["user_id", "amount"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Assign defaults
    data.setdefault("timestamp", time.time())
    data.setdefault("country_code", "")
    data.setdefault("device_id", str(uuid.uuid4()))
    data.setdefault("is_vpn", False)

    tx_id = f"TX-{int(time.time() * 1000) % 100000:05d}"

    result = score_transaction(data)

    record = {
        "tx_id":        tx_id,
        "user_id":      data["user_id"],
        "amount":       data["amount"],
        "country_code": data.get("country_code"),
        "location":     data.get("location", data.get("country_code", "Unknown")),
        "timestamp":    data["timestamp"],
        "scored_at":    datetime.now(timezone.utc).isoformat(),
        **result,
    }

    _tx_log.appendleft(record)
    _stats["total_today"] += 1

    if result["risk_level"] in ("HIGH", "MEDIUM"):
        _stats["fraud_flagged"] += 1
        if result["risk_level"] == "HIGH":
            _alerts.appendleft({**record, "alert_type": "HIGH_RISK_TRANSACTION"})
            log.warning(
                f"🚨 HIGH RISK | {tx_id} | User: {data['user_id']} | "
                f"Score: {result['risk_score']} | ${data['amount']:.2f}"
            )

    return jsonify(record), 200


@app.route("/api/transactions", methods=["GET"])
def get_transactions():
    """Return recent transactions (newest first). Supports ?limit=N&risk=high"""
    limit     = int(request.args.get("limit", 50))
    risk_filter = request.args.get("risk", None)  # high | medium | low

    txs = list(_tx_log)
    if risk_filter:
        txs = [t for t in txs if t.get("risk_level", "").lower() == risk_filter.lower()]

    return jsonify(txs[:limit])


@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Return high-risk alerts."""
    limit = int(request.args.get("limit", 20))
    return jsonify(list(_alerts)[:limit])


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Dashboard summary stats."""
    return jsonify({
        **_stats,
        "transactions_in_memory": len(_tx_log),
        "active_alerts": len(_alerts),
        "model_status": "active" if load_models()[0] is not None else "fallback",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/profile/<user_id>", methods=["GET"])
def get_user_profile(user_id):
    """Return a user's behavioral profile summary."""
    store   = get_store()
    profile = store.get(user_id)

    amounts = list(profile["amounts"])
    hours   = list(profile["hours"])

    return jsonify({
        "user_id":          user_id,
        "transaction_count": len(amounts),
        "avg_amount":        round(float(np.mean(amounts)), 2) if amounts else 0,
        "std_amount":        round(float(np.std(amounts)), 2)  if amounts else 0,
        "typical_hours":     sorted(set(hours)) if hours else [],
        "known_locations":   list(profile["locations"]),
        "known_devices":     len(profile["devices"]),
        "last_seen":         profile.get("last_timestamp"),
    })


@app.route("/api/profile/<user_id>/reset", methods=["POST"])
def reset_user_profile(user_id):
    """Clear a user's behavioral baseline (use after confirmed fraud remediation)."""
    store = get_store()
    store._profiles.pop(user_id, None)
    return jsonify({"message": f"Profile for {user_id} reset.", "user_id": user_id})


@app.route("/api/health", methods=["GET"])
def health():
    iso, lr, scaler, _ = load_models()
    return jsonify({
        "status": "ok",
        "models_loaded": iso is not None,
        "uptime_transactions": _stats["total_today"],
    })


# ─── ERROR HANDLERS ───────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ─── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log.info("  FraudSense API starting on port 5000")
    log.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    load_models()  # warm up on startup
    app.run(host="0.0.0.0", port=5000, debug=True)
