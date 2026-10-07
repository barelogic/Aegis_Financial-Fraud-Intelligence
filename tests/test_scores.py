"""Scores are 0-100 and every row has a non-empty reason."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import compute_features
from score_accounts import score_accounts
from score_transactions import score_transactions

CFG = {"seed": 42, "risk_thresholds": {"medium": 40, "high": 65, "critical": 85},
       "blend": {"isolation_forest": 0.5, "supervised": 0.5},
       "model": {"train_window_frac": 0.6, "n_estimators": 20, "max_depth": 3,
                 "learning_rate": 0.1, "contamination": 0.1, "oof_splits": 2},
       "suppression": {"amount_high_inr": 50000, "zscore_max": 2.0,
                       "fanin_max": 1, "cap_score": 39},
       "account_scoring": {"w_max": 0.6, "w_top3": 0.4, "ring_bump": 20,
                           "ring_txn_bump": 10}}


def _toy(n_accts: int = 8, per: int = 12):
    base = datetime(2025, 1, 1)
    accs = pd.DataFrame([{"account_id": f"a{i}", "created_at": base,
                           "home_city": "Mumbai", "home_country": "India",
                           "segment": "salaried"} for i in range(n_accts)])
    rows = []
    k = 0
    for i in range(n_accts):
        for j in range(per):
            rows.append({"txn_id": f"t{k:04d}",
                         "timestamp": base + timedelta(hours=7 * (j + 1) + i),
                         "account_id": f"a{i}", "amount": 1000 + 50 * j,
                         "currency": "INR", "merchant_id": "m1",
                         "merchant_category": "grocery",
                         "item_id": f"item_{j % 4}", "item_category": "grocery",
                         "channel": "POS", "device_id": f"d{i}",
                         "ip_address": f"10.0.0.{i + 1}", "city": "Mumbai",
                         "country": "India", "txn_type": "purchase",
                         "dest_account_id": ""})
            k += 1
    return pd.DataFrame(rows), accs


def test_scores_bounded_and_explained():
    txns, accs = _toy()
    feats = compute_features(txns, accs, CFG)
    scored = score_transactions(feats, txns, None, CFG)
    assert scored["txn_risk_score"].between(0, 100).all()
    assert (scored["txn_reasons"].str.strip() != "").all()
    assert set(scored["txn_risk_band"]) <= {"low", "medium", "high", "critical"}
    acc_scored = score_accounts(scored, feats, accs, None, CFG)
    assert acc_scored["account_risk_score"].between(0, 100).all()
    assert (acc_scored["account_reasons"].str.strip() != "").all()
