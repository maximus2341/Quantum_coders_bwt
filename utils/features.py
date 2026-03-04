import time
from datetime import datetime, timezone
from collections import defaultdict, deque
import numpy as np

class UserProfileStore:
    def __init__(self):
        self._profiles = defaultdict(lambda: {
            "amounts": deque(maxlen=200),
            "hours": deque(maxlen=200),
            "locations": set(),
            "devices": set(),
            "tx_timestamps": deque(maxlen=500),
            "last_location": None,
            "last_timestamp": None,
        })
    def get(self, user_id):
        return self._profiles[user_id]
    def update(self, user_id, tx):
        p = self._profiles[user_id]
        p["amounts"].append(tx.get("amount", 0))
        p["hours"].append(tx.get("hour", 12))
        p["tx_timestamps"].append(tx.get("timestamp", time.time()))
        if tx.get("location"):
            p["locations"].add(tx["location"])
            p["last_location"] = tx["location"]
        if tx.get("device_id"):
            p["devices"].add(tx["device_id"])
        p["last_timestamp"] = tx.get("timestamp", time.time())

_store = UserProfileStore()

def get_store():
    return _store

GEO_RISK_MAP = {
    "RU": 0.85, "NG": 0.80, "CN": 0.60, "BR": 0.55, "UA": 0.70,
    "US": 0.15, "GB": 0.12, "DE": 0.10, "IN": 0.20, "FR": 0.12,
}

def _geo_risk(country_code):
    return GEO_RISK_MAP.get((country_code or "").upper(), 0.35)

def _velocity(timestamps, window_seconds):
    now = time.time()
    return sum(1 for t in timestamps if now - t <= window_seconds)

def _amount_zscore(amount, history):
    if len(history) < 5:
        return 0.0
    arr = np.array(history)
    std = arr.std()
    return float(np.clip((amount - arr.mean()) / std, -10, 10)) if std > 1e-6 else 0.0

def extract_features(tx, store):
    user_id = tx.get("user_id", "unknown")
    amount = float(tx.get("amount", 0))
    timestamp = float(tx.get("timestamp", time.time()))
    country = tx.get("country_code", "")
    device_id = tx.get("device_id", "")
    is_vpn = int(bool(tx.get("is_vpn", False)))
    location = tx.get("location", "")
    profile = store.get(user_id)
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    last_ts = profile.get("last_timestamp")
    return {
        "amount": amount,
        "hour": dt.hour,
        "velocity_1min": _velocity(profile["tx_timestamps"], 60),
        "velocity_5min": _velocity(profile["tx_timestamps"], 300),
        "velocity_1hr": _velocity(profile["tx_timestamps"], 3600),
        "amount_zscore": _amount_zscore(amount, profile["amounts"]),
        "geo_risk_score": _geo_risk(country),
        "is_new_device": int(bool(device_id) and device_id not in profile["devices"]),
        "is_new_location": int(bool(location) and location not in profile["locations"]),
        "time_since_last_tx": (timestamp - last_ts) if last_ts else 86400.0,
        "is_foreign": int(bool(country) and country.upper() not in {l.split(",")[-1].strip().upper() for l in profile["locations"] if l}),
        "is_vpn": is_vpn,
    }

def interpret_score(risk_score, anomaly_score):
    score_100 = round((risk_score * 0.70) + (anomaly_score * 0.30) * 100)
    level = "HIGH" if score_100 >= 75 else "MEDIUM" if score_100 >= 45 else "LOW"
    action = "BLOCK" if level == "HIGH" else "REVIEW" if level == "MEDIUM" else "ALLOW"
    return {"risk_score": score_100, "risk_level": level, "action": action,
            "lr_score": round(risk_score * 100, 2), "if_score": round(anomaly_score * 100, 2)}