"""Synthetic data generator (baseline for teammates to improve).

Scenarios:
  - legit: normal per-account behavior (incl. legit high-value business spend).
  - stolen_card: burst of online purchases on a new device/city.
  - cashout_mule: incoming transfers then rapid cash_out (fan-in destination).
  - ring_A (stealth): 10 accounts with similar purchase sequences and
    occasional shared devices/IPs, but INDIVIDUALLY normal amounts --
    transaction scores alone should NOT catch these (by design).
  - ring_B (obvious): shared device + fan-in to one destination + bursts.

Writes transactions.csv, accounts.csv, ground_truth_txn.csv,
ground_truth_accounts.csv into the raw dir.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


def _rng(seed: int) -> tuple[random.Random, np.random.Generator, Faker]:
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)
    return rnd, rng, fake


def generate_all(raw_dir: str, seed: int = 42, config: dict | None = None,
                 n_accounts: int = 400, n_days: int = 30) -> None:
    """Generate accounts, transactions, and ground-truth files."""
    import os
    os.makedirs(raw_dir, exist_ok=True)
    rnd, rng, fake = _rng(seed)
    segments = ["salaried", "student", "business", "retired"]
    seg_median = {"salaried": 2500, "student": 800, "business": 25000,
                  "retired": 1500}
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Pune", "Jaipur"]
    mccs = ["grocery", "fuel", "electronics", "fashion", "food", "travel"]
    channels = ["POS", "online", "UPI", "ATM", "in_app"]

    start = datetime(2025, 1, 1)
    accounts, txns = [], []
    txn_counter = [0]

    def _add_txn(acct: str, ts: datetime, amt: float, **kw) -> str:
        txn_counter[0] += 1
        tid = f"txn_{txn_counter[0]:06d}"
        txns.append({"txn_id": tid, "timestamp": ts, "account_id": acct,
                     "amount": round(amt, 2), "currency": "INR",
                     "merchant_id": kw.get("merchant_id", f"m_{rnd.randint(1, 200)}"),
                     "merchant_category": kw.get("mcc", rnd.choice(mccs)),
                     "item_id": kw.get("item_id", f"item_{rnd.randint(1, 500)}"),
                     "item_category": kw.get("item_cat", rnd.choice(mccs)),
                     "channel": kw.get("channel", rnd.choice(channels)),
                     "device_id": kw.get("device", f"dev_{acct}"),
                     "ip_address": kw.get("ip", f"10.0.{rnd.randint(0,50)}.{rnd.randint(1,254)}"),
                     "city": kw.get("city", "Mumbai"), "country": "India",
                     "txn_type": kw.get("txn_type", "purchase"),
                     "dest_account_id": kw.get("dest", "")})
        return tid

    gt_txn: list[dict] = []   # txn_id, is_fraud, ring_id, scenario
    gt_acc: dict[str, dict] = {}  # account_id -> ground truth row

    def _mark_fraud(tid: str, ring: str, scenario: str) -> None:
        gt_txn.append({"txn_id": tid, "is_fraud": 1, "ring_id": ring,
                       "scenario": scenario})

    # ---- normal accounts ----
    for i in range(n_accounts):
        aid = f"acc_{i:04d}"
        seg = rnd.choices(segments, weights=[45, 25, 15, 15])[0]
        city = rnd.choice(cities)
        created = start - timedelta(days=rnd.randint(60, 540))
        accounts.append({"account_id": aid, "created_at": created,
                         "home_city": city, "home_country": "India",
                         "segment": seg})
        gt_acc[aid] = {"account_id": aid, "is_fraud_account": 0,
                       "ring_id": "", "scenario": "legit"}
        med = seg_median[seg]
        for d in range(n_days):
            for _ in range(rng.poisson(1.2)):
                ts = start + timedelta(days=d,
                                       hours=rnd.randint(8, 21),
                                       minutes=rnd.randint(0, 59))
                amt = float(rng.lognormal(np.log(med), 0.5))
                _add_txn(aid, ts, amt, city=city)

    # ---- stolen-card bursts (5 accounts) ----
    for aid in rnd.sample([a["account_id"] for a in accounts], 5):
        ts0 = start + timedelta(days=rnd.randint(20, 29),
                                hours=rnd.randint(0, 4))
        for k in range(rnd.randint(6, 10)):
            tid = _add_txn(aid, ts0 + timedelta(minutes=10 * k),
                           float(rng.uniform(8000, 40000)),
                           channel="online", city="UnknownCity",
                           device=f"dev_attack_{aid}", txn_type="purchase")
            _mark_fraud(tid, "", "stolen_card")
        gt_acc[aid].update(is_fraud_account=1, scenario="stolen_card")

    # ---- cash-out mules (5 accounts, fan-in to one dest) ----
    sink = "acc_sink_mule"
    for j in range(5):
        aid = f"acc_mule_{j}"
        accounts.append({"account_id": aid, "created_at": start,
                         "home_city": "Delhi", "home_country": "India",
                         "segment": "student"})
        gt_acc[aid] = {"account_id": aid, "is_fraud_account": 1,
                       "ring_id": "ring_B", "scenario": "cashout_mule"}
        ts0 = start + timedelta(days=rnd.randint(20, 29), hours=2)
        tid_in = _add_txn(aid, ts0, 60000, txn_type="transfer",
                          dest=sink, device="dev_mule_shared")
        _mark_fraud(tid_in, "ring_B", "cashout_mule")
        tid_out = _add_txn(aid, ts0 + timedelta(hours=1), 59000,
                           txn_type="cash_out", channel="ATM",
                           device="dev_mule_shared")
        _mark_fraud(tid_out, "ring_B", "cashout_mule")

    # ---- ring A: stealth (normal-looking amounts, shared sequence) ----
    ring_a_seq = [("electronics", 1800), ("fashion", 2200), ("food", 900),
                  ("grocery", 1500), ("travel", 2600)]
    for j in range(10):
        aid = f"acc_ringA_{j}"
        accounts.append({"account_id": aid,
                         "created_at": start + timedelta(days=rnd.randint(18, 22)),
                         "home_city": "Pune", "home_country": "India",
                         "segment": "salaried"})
        gt_acc[aid] = {"account_id": aid, "is_fraud_account": 1,
                       "ring_id": "ring_A", "scenario": "ring_A_stealth"}
        ts0 = start + timedelta(days=23, hours=10)
        dev = "dev_ringA_shared" if j % 3 == 0 else f"dev_acc_ringA_{j}"
        for k, (cat, base) in enumerate(ring_a_seq):
            tid = _add_txn(
                aid, ts0 + timedelta(hours=5 * k + rnd.randint(0, 1)),
                base + float(rng.uniform(-200, 200)), mcc=cat,
                item_id=f"item_ringA_{k}", item_cat=cat, city="Pune",
                device=dev, ip="10.9.9.9" if j % 2 == 0 else f"10.1.{j}.5")
            _mark_fraud(tid, "ring_A", "ring_A_stealth")

    acc_df = pd.DataFrame(accounts)
    txn_df = pd.DataFrame(txns).sort_values("timestamp").reset_index(drop=True)
    gt_txn_df = pd.DataFrame(gt_txn)
    # Non-fraud txns get explicit ground-truth rows (is_fraud=0).
    fraud_ids = set(gt_txn_df["txn_id"]) if len(gt_txn_df) else set()
    legit_rows = [{"txn_id": t, "is_fraud": 0, "ring_id": "", "scenario": "legit"}
                  for t in txn_df["txn_id"] if t not in fraud_ids]
    gt_txn_df = pd.concat([gt_txn_df, pd.DataFrame(legit_rows)],
                          ignore_index=True)
    gt_acc_df = pd.DataFrame(list(gt_acc.values()))

    acc_df.to_csv(f"{raw_dir}/accounts.csv", index=False)
    txn_df.to_csv(f"{raw_dir}/transactions.csv", index=False)
    gt_txn_df.to_csv(f"{raw_dir}/ground_truth_txn.csv", index=False)
    gt_acc_df.to_csv(f"{raw_dir}/ground_truth_accounts.csv", index=False)
    print(f"[generate_data] {len(acc_df)} accounts, {len(txn_df)} txns "
          f"({int((gt_txn_df['is_fraud'] == 1).sum())} fraud) -> {raw_dir}")
