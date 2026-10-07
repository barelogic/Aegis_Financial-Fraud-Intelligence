"""Tests for src/live.py: incremental scoring matches batch semantics.

Fits on a tiny deterministic fixture (fast) and checks: identical engines
score the same arrival identically, bad rows are rejected, unknown accounts
are registered, and rings/refresh finds a planted collector ring.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from live import LiveRings, LiveScorer

CFG = {"seed": 7, "risk_thresholds": {"medium": 40, "high": 65, "critical": 85},
       "blend": {"isolation_forest": 0.5, "supervised": 0.5},
       "model": {"train_window_frac": 0.6, "n_estimators": 10, "max_depth": 3,
                 "learning_rate": 0.1, "contamination": 0.1, "oof_splits": 2},
       "suppression": {"amount_high_inr": 50000, "zscore_max": 2.0,
                       "fanin_max": 1, "cap_score": 39},
       "account_scoring": {"w_max": 0.6, "w_top3": 0.4, "ring_bump": 20,
                           "ring_txn_bump": 10},
       "min_ring_fanin": 4}


def _history():
    base = datetime(2025, 1, 1)
    rows = []
    for a in ("a0", "a1"):
        for j in range(6):
            rows.append({"txn_id": f"h_{a}_{j}",
                         "timestamp": base + timedelta(hours=6 * j),
                         "account_id": a, "amount": 1000.0 + 100 * j,
                         "currency": "INR", "merchant_id": "m1",
                         "merchant_category": "grocery",
                         "item_id": f"item_{j % 3}", "item_category": "grocery",
                         "channel": "POS", "device_id": f"d_{a}",
                         "ip_address": f"10.9.9.{1 if a == 'a0' else 2}",
                         "city": "Mumbai", "country": "India",
                         "txn_type": "purchase", "dest_account_id": ""})
    txns = pd.DataFrame(rows)
    accs = pd.DataFrame([
        {"account_id": a, "created_at": base, "home_city": "Mumbai",
         "home_country": "India", "segment": "salaried"} for a in ("a0", "a1")])
    # Two obvious fraud rows so the supervised leg has both classes.
    fraud_ids = {"h_a1_5", "h_a0_5"}
    labels = pd.DataFrame([
        {"txn_id": t, "is_fraud": int(t in fraud_ids), "ring_id": "",
         "scenario": "legit"}
        for t in txns["txn_id"]])
    return txns, accs, labels


def _arrival(tid="live_1", acct="a0", amt=50000.0, **kw):
    row = {"txn_id": tid, "timestamp": "2025-01-03 10:00:00",
           "account_id": acct, "amount": amt, "currency": "INR",
           "merchant_id": "m9", "merchant_category": "electronics",
           "item_id": "item_x", "item_category": "electronics",
           "channel": "online", "device_id": "d_new",
           "ip_address": "10.9.9.9", "city": "Delhi", "country": "India",
           "txn_type": "purchase", "dest_account_id": ""}
    row.update(kw)
    return row


def _engine():
    txns, accs, labels = _history()
    return LiveScorer(CFG).fit(txns, accs, labels)


def test_live_scores_are_deterministic():
    e1, _ = _engine().score_one(_arrival())
    e2, _ = _engine().score_one(_arrival())
    assert e1["txn_risk_score"] == e2["txn_risk_score"]
    assert e1["txn_risk_band"] == e2["txn_risk_band"]
    assert e1["txn_action"] == e2["txn_action"]
    assert e1["live"] is True
    assert 0 <= e1["txn_risk_score"] <= 100
    assert e1["txn_reasons"].strip() != ""


def test_live_rejects_bad_and_duplicate_rows():
    eng = _engine()
    bad = _arrival()
    del bad["amount"]
    with pytest.raises(ValueError, match="missing required field"):
        eng.score_one(bad)
    eng.score_one(_arrival(tid="dup_1"))
    with pytest.raises(ValueError, match="duplicate txn_id"):
        eng.score_one(_arrival(tid="dup_1"))


def test_live_registers_unknown_account():
    eng = _engine()
    event, _ = eng.score_one(_arrival(tid="n1", acct="brand_new"))
    assert event["account_id"] == "brand_new"
    assert "brand_new" in eng.store.acct_info


def test_rings_refresh_finds_collector():
    eng = _engine()
    rings_engine = LiveRings(CFG)
    for j in range(4):
        raw = _arrival(tid=f"c{j}", acct=f"s{j}", amt=20000.0,
                       txn_type="transfer", dest_account_id="sink_live",
                       device_id="d_c", ip_address=f"10.8.8.{j}",
                       timestamp="2025-01-04 10:00:00")
        event, row = eng.score_one(raw)
        rings_engine.add(row, event)
    cols = ["txn_id", "timestamp", "account_id", "amount", "txn_risk_score",
            "dest_account_id", "device_id", "ip_address", "merchant_category",
            "item_id", "txn_type"]
    empty_scored = pd.DataFrame(columns=cols)
    empty_raw = pd.DataFrame(columns=cols)
    rings = rings_engine.refresh(empty_scored, empty_raw)
    assert len(rings) == 1
    assert sorted(rings[0]["accounts"]) == [f"s{j}" for j in range(4)]
    assert "shared_destination" in rings[0]["link_types_found"]


def test_refresh_handles_mixed_timestamp_types():
    # Batch rows from raw CSV carry STRING timestamps while live rows carry
    # Timestamps. Sorting them together used to raise:
    # TypeError '<' not supported between Timestamp and str.
    eng = _engine()
    rings_engine = LiveRings(CFG)
    for j in range(4):
        raw = _arrival(tid=f"m{j}", acct=f"w{j}", amt=20000.0,
                       txn_type="transfer", dest_account_id="sink_mix",
                       device_id="d_w", ip_address=f"10.7.7.{j}",
                       timestamp="2025-01-04 10:00:00")
        event, row = eng.score_one(raw)
        rings_engine.add(row, event)
    batch_scored = pd.DataFrame([{
        "txn_id": "b0", "timestamp": "2025-01-02 10:00:00",
        "account_id": "a0", "amount": 1000.0, "txn_risk_score": 5.0,
        "dest_account_id": "", "device_id": "d_a0",
        "ip_address": "10.9.9.1", "merchant_category": "grocery",
        "item_id": "item_0", "txn_type": "purchase"}])
    batch_raw = pd.DataFrame([{
        "txn_id": "b0", "timestamp": "2025-01-02 10:00:00",
        "account_id": "a0", "amount": 1000.0,
        "dest_account_id": "", "device_id": "d_a0",
        "ip_address": "10.9.9.1", "merchant_category": "grocery",
        "item_id": "item_0", "txn_type": "purchase"}])
    rings = rings_engine.refresh(batch_scored, batch_raw)
    assert len(rings) == 1
    assert sorted(rings[0]["accounts"]) == [f"w{j}" for j in range(4)]
