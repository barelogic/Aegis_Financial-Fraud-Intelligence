"""Scores are 0-100 and every row has a non-empty reason."""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import compute_features
from score_accounts import apply_ring_bump, score_accounts
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


def _scored_frame(rows: list[dict]) -> pd.DataFrame:
    base = {"timestamp": datetime(2025, 1, 2), "currency": "INR",
            "merchant_id": "m1", "merchant_category": "grocery",
            "item_id": "item_1", "item_category": "grocery", "channel": "POS",
            "device_id": "d0", "ip_address": "10.0.0.1", "city": "Mumbai",
            "country": "India", "txn_type": "transfer", "txn_risk_band": "high",
            "txn_reasons": "x", "txn_action": "review"}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_pure_receiver_sink_is_scored_from_incoming_flow():
    scored = _scored_frame([
        {"txn_id": f"s{i}", "account_id": f"sender{i}", "amount": 50000,
         "dest_account_id": "sink1", "txn_risk_score": 70.0 + 5 * i}
        for i in range(4)])
    accs = pd.DataFrame([
        {"account_id": f"sender{i}", "created_at": datetime(2025, 1, 1),
         "home_city": "Mumbai", "home_country": "India", "segment": "salaried"}
        for i in range(4)] + [
        {"account_id": "sink1", "created_at": datetime(2025, 1, 1),
         "home_city": "Mumbai", "home_country": "India", "segment": "salaried"}])
    out = score_accounts(scored, None, accs, None, CFG)
    row = out[out["account_id"] == "sink1"].iloc[0]
    assert row["account_risk_score"] >= 65, row["account_risk_score"]
    assert "4 distinct" in row["account_reasons"]
    # ... and attributed to the senders' ring when >=3 senders are members.
    rings = [{"ring_id": "ring_9", "accounts": [f"sender{i}" for i in range(4)],
              "size": 4, "ring_risk_score": 80.0,
              "link_types_found": ["shared_destination"],
              "evidence": {}, "pattern_summary": "p",
              "recommended_actions": []}]
    out2 = apply_ring_bump(out, rings, CFG, scored)
    row2 = out2[out2["account_id"] == "sink1"].iloc[0]
    assert row2["ring_id"] == "ring_9"
    assert "Collector" in row2["account_reasons"]


def test_low_risk_reasons_have_no_zero_value_claims():
    txns, accs = _toy()
    feats = compute_features(txns, accs, CFG)
    scored = score_transactions(feats, txns, None, CFG)
    low = scored[scored["txn_risk_score"] < 40]
    assert len(low) > 0
    for reasons in low["txn_reasons"]:
        assert "0 different accounts" not in reasons, reasons
        assert "Rs 0" not in reasons, reasons
        assert "equals 0%" not in reasons, reasons
