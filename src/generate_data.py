"""Synthetic fraud-intelligence dataset generator.

Builds a realistic, fully deterministic (fixed seed) dataset over 30 days
with ~600 accounts and ~15,000-25,000 transactions (currency INR), writing
the four CSVs into data/raw/ with these exact schemas:

  transactions.csv:         txn_id,timestamp,account_id,amount,currency,
                            merchant_id,merchant_category,item_id,
                            item_category,channel,device_id,ip_address,
                            city,country,txn_type,dest_account_id
  accounts.csv:             account_id,created_at,home_city,home_country,segment
  ground_truth_txn.csv:     txn_id,is_fraud,ring_id,scenario
  ground_truth_accounts.csv: account_id,is_fraud_account,ring_id,scenario

Normal behavior (~560 accounts):
  Segments salaried / student / small_business / retiree, each with its own
  spend-size lognormal, active hours, merchant-category mix and one home city.
  Recurring items (rent, subscriptions, salary credit), 1-2 devices per
  account, one stable /24 IP range per account. Account ages spread over
  2 months to 5 years.

Planted fraud (ground truth for every row):
  ring_a (core stealth, 10 accounts): different people/cities/devices/IP
      ranges (zero shared device or /24), different stores and channels. The
      ONLY links: same rare items in the same order inside a 48h window, then
      transfers to the same 2 sinks within hours. Every txn individually
      ordinary (segment-normal amount, normal hour, no velocity burst).
  ring_s (stretch, 6 accounts): same pattern but sharing devices + one /24.
  ring_b_ato (5 long-standing accounts): new device + new city, then drain
      to one destination.
  lone_fraud (4 accounts): obvious velocity bursts / amount spikes, no links.
  behavior_change (3 accounts): normal 20 days, then pattern switch.

Hard negatives (is_fraud=0, scenario named per case, >=25 accounts):
  hard_neg_laptop / hard_neg_supplier / hard_neg_travel /
  hard_neg_family_device / hard_neg_shared_ip / hard_neg_festival /
  hard_neg_new_legit

Usage:
  python src/generate_data.py --raw-dir data/raw --seed 42
  python src/generate_data.py --raw-dir /tmp/out --seed 42 --plot out.png
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# --------------------------------------------------------------------------
# Schema constants (must match data/raw/ exactly)
# --------------------------------------------------------------------------

TXN_COLUMNS = ["txn_id", "timestamp", "account_id", "amount", "currency",
               "merchant_id", "merchant_category", "item_id", "item_category",
               "channel", "device_id", "ip_address", "city", "country",
               "txn_type", "dest_account_id"]
ACC_COLUMNS = ["account_id", "created_at", "home_city", "home_country",
               "segment"]
GT_TXN_COLUMNS = ["txn_id", "is_fraud", "ring_id", "scenario"]
GT_ACC_COLUMNS = ["account_id", "is_fraud_account", "ring_id", "scenario"]

# --------------------------------------------------------------------------
# Population / segment configuration
# --------------------------------------------------------------------------

SEGMENTS = ["salaried", "student", "small_business", "retiree"]
SEG_WEIGHTS = [0.40, 0.25, 0.15, 0.20]
# median spend (INR) + lognormal sigma per segment
SEG_MEDIAN = {"salaried": 2500.0, "student": 700.0,
              "small_business": 18000.0, "retiree": 1500.0}
SEG_SIGMA = {"salaried": 0.6, "student": 0.7,
             "small_business": 0.8, "retiree": 0.6}
# active hours (inclusive) per segment
SEG_HOURS = {"salaried": (8, 22), "student": (10, 23),
             "small_business": (9, 20), "retiree": (7, 20)}
# merchant-category mix per segment (weights over MCC below)
MCCS = ["grocery", "fuel", "electronics", "fashion", "food",
        "travel", "health", "utilities"]
SEG_MCC_W = {
    "salaried": [20, 10, 10, 15, 20, 8, 7, 10],
    "student": [15, 5, 12, 20, 30, 8, 5, 5],
    "small_business": [15, 12, 15, 5, 10, 12, 6, 25],
    "retiree": [25, 8, 5, 8, 15, 7, 17, 15],
}
CHANNELS = ["POS", "online", "UPI", "ATM", "in_app"]

CITIES = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Pune",
          "Jaipur", "Hyderabad", "Kolkata", "Ahmedabad", "Lucknow"]
FOREIGN = [("Dubai", "UAE"), ("Singapore", "Singapore"), ("Bangkok", "Thailand")]

# Rare SKUs: never used by normal traffic, only by their ring.
RING_A_ITEMS = [("SKU_GIFTCARD_RARE_7714", "gift_card"),
                ("SKU_ELEC_RARE_3391", "electronics")]
RING_S_ITEMS = [("SKU_GIFTCARD_RARE_5520", "gift_card"),
                ("SKU_ELEC_RARE_8842", "electronics")]

START = datetime(2025, 1, 1)
N_DAYS = 30


def _rng(seed: int):
    rnd = random.Random(seed)
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)
    return rnd, rng, fake


def _subnet(ip: str) -> str:
    return ip.rsplit(".", 1)[0] if "." in ip else ip


# --------------------------------------------------------------------------
# Main generation
# --------------------------------------------------------------------------

def generate_all(raw_dir: str, seed: int = 42, config: dict | None = None,
                 n_accounts: int = 560, n_days: int = 30) -> dict:
    """Generate accounts, transactions and ground truth into raw_dir.

    Returns a dict with the four DataFrames plus scenario counts.
    Fully deterministic from (seed, n_accounts, n_days).
    """
    os.makedirs(raw_dir, exist_ok=True)
    rnd, rng, fake = _rng(seed)
    start = datetime(2025, 1, 1)

    accounts: list[dict] = []
    txns: list[dict] = []
    gt_txn: list[dict] = []
    gt_acc: dict[str, dict] = {}
    txn_counter = [0]

    # per-account runtime state (devices, ip base, segment, city, fraud window)
    state: dict[str, dict] = {}
    # Counts default-subnet assignment so unrelated accounts never share one.
    _subnet_counter: dict[str, int] = {}

    def _add_txn(acct, ts, amt, fraud=0, ring="", scenario="legit", **kw):
        txn_counter[0] += 1
        tid = f"txn_{txn_counter[0]:06d}"
        txns.append({
            "txn_id": tid, "timestamp": ts, "account_id": acct,
            "amount": round(float(amt), 2), "currency": "INR",
            "merchant_id": kw.get("merchant_id", f"m_{rnd.randint(1, 300)}"),
            "merchant_category": kw.get("mcc", rnd.choice(MCCS)),
            "item_id": kw.get("item_id", f"item_{rnd.randint(1, 500)}"),
            "item_category": kw.get("item_cat", kw.get("mcc", "grocery")),
            "channel": kw.get("channel", rnd.choice(CHANNELS)),
            "device_id": kw.get("device", f"dev_{acct}_0"),
            "ip_address": kw.get("ip", "10.0.0.1"),
            "city": kw.get("city", "Mumbai"), "country": kw.get("country", "India"),
            "txn_type": kw.get("txn_type", "purchase"),
            "dest_account_id": kw.get("dest", "")})
        gt_txn.append({"txn_id": tid, "is_fraud": fraud,
                       "ring_id": ring, "scenario": scenario})
        return tid

    def _normal_amount(seg):
        return float(rng.lognormal(np.log(SEG_MEDIAN[seg]), SEG_SIGMA[seg]))

    def _normal_hour(seg):
        lo, hi = SEG_HOURS[seg]
        return rnd.randint(lo, hi)

    def _normal_mcc(seg):
        return rnd.choices(MCCS, weights=SEG_MCC_W[seg])[0]

    def _mk_account(aid, seg, city, created, scenario="legit", fraud=0,
                    ring="", ip_base=None, devices=None):
        if ip_base is None:
            seg_oct = {"salaried": 11, "student": 12,
                       "small_business": 13, "retiree": 14}[seg]
            # Unique /24 per account: unrelated accounts must NEVER share a
            # subnet by chance. Same-/24 + random-host IPs collide on full
            # IPs, fabricating shared-IP ring edges that drag legit accounts
            # into fraud rings (seen: acc_0080 pulled into RING_B's group).
            # Holds while a segment has < 245 generated accounts.
            _subnet_counter[seg] = _subnet_counter.get(seg, 0) + 1
            ip_base = f"10.{seg_oct}.{(5 + _subnet_counter[seg]) % 250}"
        if devices is None:
            devices = [f"dev_{aid}_0"]
            if rnd.random() < 0.35:
                devices.append(f"dev_{aid}_1")
        accounts.append({"account_id": aid, "created_at": created,
                         "home_city": city, "home_country": "India",
                         "segment": seg})
        gt_acc[aid] = {"account_id": aid, "is_fraud_account": fraud,
                       "ring_id": ring, "scenario": scenario}
        state[aid] = {"seg": seg, "city": city, "devices": devices,
                      "ip_base": ip_base, "quiet": []}  # quiet: list of (t0,t1)
        return aid

    def _acct_ip(aid):
        b = state[aid]["ip_base"]
        return f"{b}.{rnd.randint(2, 250)}"

    def _acct_device(aid):
        return rnd.choice(state[aid]["devices"])

    def _in_quiet(aid, ts):
        return any(t0 <= ts <= t1 for t0, t1 in state[aid]["quiet"])

    def _normal_purchase(aid, ts, scenario="legit"):
        st = state[aid]
        mcc = _normal_mcc(st["seg"])
        _add_txn(aid, ts, _normal_amount(st["seg"]), fraud=0, scenario=scenario,
                 mcc=mcc, item_cat=mcc, city=st["city"],
                 device=_acct_device(aid), ip=_acct_ip(aid))

    # ---- 1. normal base population -------------------------------------
    for i in range(n_accounts):
        aid = f"acc_{i:04d}"
        seg = rnd.choices(SEGMENTS, weights=SEG_WEIGHTS)[0]
        city = rnd.choice(CITIES)
        created = start - timedelta(days=rnd.randint(60, 5 * 365))
        _mk_account(aid, seg, city, created)
        st = state[aid]
        # daily purchases ~Poisson(1.0) at active hours
        for d in range(n_days):
            for _ in range(int(rng.poisson(1.0))):
                ts = start + timedelta(days=d, hours=_normal_hour(seg),
                                       minutes=rnd.randint(0, 59))
                _normal_purchase(aid, ts)
        # recurring: salary credit (day 1), rent (day 5), subscriptions
        if seg in ("salaried", "small_business", "retiree"):
            pay = {"salaried": 60000, "small_business": 150000,
                   "retiree": 30000}[seg] * rnd.uniform(0.8, 1.2)
            _add_txn(aid, start + timedelta(days=0, hours=9), pay,
                     txn_type="credit", mcc="salary", item_id="SALARY_CREDIT",
                     item_cat="salary", city=st["city"],
                     device=st["devices"][0], ip=_acct_ip(aid))
            rent = rnd.uniform(8000, 25000) if seg != "small_business" else \
                rnd.uniform(15000, 50000)
            _add_txn(aid, start + timedelta(days=4, hours=10), rent,
                     mcc="utilities", item_id="RENT_MONTHLY",
                     item_cat="utilities", city=st["city"],
                     device=st["devices"][0], ip=_acct_ip(aid))
        for s in range(rnd.randint(1, 2)):
            _add_txn(aid, start + timedelta(days=rnd.randint(0, 27), hours=12),
                     rnd.uniform(199, 999), mcc="utilities",
                     item_id=f"SUBSCRIPTION_{rnd.randint(1, 5)}",
                     item_cat="utilities", city=st["city"],
                     device=st["devices"][0], ip=_acct_ip(aid))

    # ---- festival-week mild boost across many normals (still legit) ----
    fest_days = range(13, 20)
    fest_sample = rnd.sample([a["account_id"] for a in accounts],
                             min(150, len(accounts)))
    for aid in fest_sample:
        for _ in range(rnd.randint(1, 2)):
            d = rnd.choice(list(fest_days))
            ts = start + timedelta(days=d, hours=_normal_hour(state[aid]["seg"]),
                                   minutes=rnd.randint(0, 59))
            mcc = rnd.choice(["fashion", "electronics", "grocery", "food"])
            _add_txn(aid, ts, _normal_amount(state[aid]["seg"]) * rnd.uniform(1.0, 1.8),
                     mcc=mcc, item_cat=mcc, city=state[aid]["city"],
                     device=_acct_device(aid), ip=_acct_ip(aid))

    # ---- 2. RING_A core stealth (10 accounts) ---------------------------
    ring_a_cities = rnd.sample(CITIES, 10)  # all different
    ring_a_segs = (["salaried"] * 4 + ["student"] * 2 +
                   ["small_business"] * 2 + ["retiree"] * 2)
    rnd.shuffle(ring_a_segs)
    ring_a_ids: list[str] = []
    sink_a1, sink_a2 = "acc_sink_A1", "acc_sink_A2"
    win0 = start + timedelta(days=22, hours=9)  # 48h window start
    ring_a_channels = ["POS", "online", "UPI", "ATM", "in_app",
                       "POS", "online", "UPI", "ATM", "in_app"]
    rnd.shuffle(ring_a_channels)
    for j in range(10):
        aid = f"acc_ringA_{j}"
        seg = ring_a_segs[j]
        city = ring_a_cities[j]
        created = start - timedelta(days=rnd.randint(90, 900))
        ip_base = f"172.16.{11 + j}"  # distinct /24 per member
        _mk_account(aid, seg, city, created, scenario="ring_a",
                    fraud=1, ring="RING_A", ip_base=ip_base,
                    devices=[f"dev_ringA_{j}_p"])
        ring_a_ids.append(aid)
        st = state[aid]
        # background normal traffic, kept out of the fraud window
        t0, t1 = win0 - timedelta(hours=6), win0 + timedelta(hours=54)
        st["quiet"].append((t0, t1))
        for d in range(n_days):
            for _ in range(int(rng.poisson(1.0))):
                ts = start + timedelta(days=d, hours=_normal_hour(seg),
                                       minutes=rnd.randint(0, 59))
                if _in_quiet(aid, ts):
                    continue
                _normal_purchase(aid, ts)
        # fraud sequence: rare item 1 -> rare item 2 (same order), then
        # two transfers (one to each sink) within hours. All ordinary-looking.
        off = rnd.randint(0, 36)  # hours into the 48h window
        med = SEG_MEDIAN[seg]

        def _clip(x):
            return float(min(max(x, 0.30 * med), 3.5 * med))

        ch = ring_a_channels[j]
        merch_a = f"m_A_{100 + j}"  # different store per member
        merch_b = f"m_A_{200 + j}"
        p1 = _clip(_normal_amount(seg))
        p2 = _clip(_normal_amount(seg))
        ts1 = win0 + timedelta(hours=off, minutes=rnd.randint(0, 50))
        ts1 = ts1.replace(hour=_normal_hour(seg))
        ts2 = ts1 + timedelta(hours=rnd.randint(5, 9))
        total = p1 + p2
        _add_txn(aid, ts1, p1, fraud=1, ring="RING_A", scenario="ring_a",
                 merchant_id=merch_a, mcc="gift_card",
                 item_id=RING_A_ITEMS[0][0], item_cat="gift_card",
                 channel=ch, city=city, device=st["devices"][0],
                 ip=f"{ip_base}.{rnd.randint(2, 250)}")
        _add_txn(aid, ts2, p2, fraud=1, ring="RING_A", scenario="ring_a",
                 merchant_id=merch_b, mcc="electronics",
                 item_id=RING_A_ITEMS[1][0], item_cat="electronics",
                 channel=ch, city=city, device=st["devices"][0],
                 ip=f"{ip_base}.{rnd.randint(2, 250)}")
        t_amt = total * rnd.uniform(0.75, 0.85)
        t1a = _clip(t_amt / 2)
        t1b = _clip(t_amt - t1a)
        ts3 = ts2 + timedelta(hours=rnd.randint(1, 3))
        ts4 = ts3 + timedelta(hours=rnd.randint(1, 2))
        _add_txn(aid, ts3, t1a, fraud=1, ring="RING_A", scenario="ring_a",
                 txn_type="transfer", dest=sink_a1, mcc="transfer",
                 item_id="TRANSFER_OUT", item_cat="transfer",
                 channel=ch, city=city, device=st["devices"][0],
                 ip=f"{ip_base}.{rnd.randint(2, 250)}")
        _add_txn(aid, ts4, t1b, fraud=1, ring="RING_A", scenario="ring_a",
                 txn_type="transfer", dest=sink_a2, mcc="transfer",
                 item_id="TRANSFER_OUT", item_cat="transfer",
                 channel=ch, city=city, device=st["devices"][0],
                 ip=f"{ip_base}.{rnd.randint(2, 250)}")
    for s in (sink_a1, sink_a2):
        accounts.append({"account_id": s, "created_at": start,
                         "home_city": "Mumbai", "home_country": "India",
                         "segment": "salaried"})
        gt_acc[s] = {"account_id": s, "is_fraud_account": 1,
                     "ring_id": "RING_A", "scenario": "ring_a_sink"}
        state[s] = {"seg": "salaried", "city": "Mumbai",
                    "devices": [f"dev_{s}"], "ip_base": "10.99.99",
                    "quiet": []}

    # ---- 3. RING_S stretch (6 accounts, shared device + /24) ------------
    ring_s_ids: list[str] = []
    sink_s1, sink_s2 = "acc_sink_S1", "acc_sink_S2"
    shared_dev_s = "dev_ringS_shared"
    shared_base_s = "192.168.77"
    winS = start + timedelta(days=24, hours=10)
    for j in range(6):
        aid = f"acc_ringS_{j}"
        seg = rnd.choice(["salaried", "student", "salaried",
                          "retiree", "student", "salaried"])
        city = rnd.choice(CITIES)
        created = start - timedelta(days=rnd.randint(60, 600))
        _mk_account(aid, seg, city, created, scenario="ring_s",
                    fraud=1, ring="RING_S", ip_base=shared_base_s,
                    devices=[shared_dev_s])
        ring_s_ids.append(aid)
        st = state[aid]
        t0, t1 = winS - timedelta(hours=6), winS + timedelta(hours=54)
        st["quiet"].append((t0, t1))
        for d in range(n_days):
            for _ in range(int(rng.poisson(0.8))):
                ts = start + timedelta(days=d, hours=_normal_hour(seg),
                                       minutes=rnd.randint(0, 59))
                if _in_quiet(aid, ts):
                    continue
                _add_txn(aid, ts, _normal_amount(seg), fraud=0,
                         scenario="legit", mcc=_normal_mcc(seg),
                         item_cat="grocery", city=city,
                         device=shared_dev_s,
                         ip=f"{shared_base_s}.{rnd.randint(2, 250)}")
        off = rnd.randint(0, 30)
        med = SEG_MEDIAN[seg]
        p1 = float(min(max(_normal_amount(seg), 0.3 * med), 3.5 * med))
        p2 = float(min(max(_normal_amount(seg), 0.3 * med), 3.5 * med))
        ts1 = (winS + timedelta(hours=off)).replace(hour=_normal_hour(seg))
        ts2 = ts1 + timedelta(hours=rnd.randint(4, 8))
        _add_txn(aid, ts1, p1, fraud=1, ring="RING_S", scenario="ring_s",
                 mcc="gift_card", item_id=RING_S_ITEMS[0][0],
                 item_cat="gift_card", city=city, device=shared_dev_s,
                 ip=f"{shared_base_s}.{20 + j}")
        _add_txn(aid, ts2, p2, fraud=1, ring="RING_S", scenario="ring_s",
                 mcc="electronics", item_id=RING_S_ITEMS[1][0],
                 item_cat="electronics", city=city, device=shared_dev_s,
                 ip=f"{shared_base_s}.{20 + j}")
        tot = (p1 + p2) * rnd.uniform(0.7, 0.85)
        _add_txn(aid, ts2 + timedelta(hours=2), tot / 2, fraud=1,
                 ring="RING_S", scenario="ring_s", txn_type="transfer",
                 dest=sink_s1, mcc="transfer", item_id="TRANSFER_OUT",
                 item_cat="transfer", city=city, device=shared_dev_s,
                 ip=f"{shared_base_s}.{20 + j}")
        _add_txn(aid, ts2 + timedelta(hours=4), tot / 2, fraud=1,
                 ring="RING_S", scenario="ring_s", txn_type="transfer",
                 dest=sink_s2, mcc="transfer", item_id="TRANSFER_OUT",
                 item_cat="transfer", city=city, device=shared_dev_s,
                 ip=f"{shared_base_s}.{20 + j}")
    for s in (sink_s1, sink_s2):
        accounts.append({"account_id": s, "created_at": start,
                         "home_city": "Delhi", "home_country": "India",
                         "segment": "salaried"})
        gt_acc[s] = {"account_id": s, "is_fraud_account": 1,
                     "ring_id": "RING_S", "scenario": "ring_s_sink"}
        state[s] = {"seg": "salaried", "city": "Delhi",
                    "devices": [f"dev_{s}"], "ip_base": "10.98.98",
                    "quiet": []}

    # ---- 4. RING_B account takeover (5 long-standing accounts) ----------
    cands = [a for a in accounts
             if a["account_id"].startswith("acc_")
             and (start - pd_to_dt(a["created_at"])).days > 450
             and gt_acc[a["account_id"]]["scenario"] == "legit"]
    ato_ids = [a["account_id"] for a in rnd.sample(cands, 5)]
    sink_b = "acc_sink_B"
    ato_city = "Kolkata"
    for aid in ato_ids:
        st = state[aid]
        home = st["city"]
        other = rnd.choice([c for c in CITIES if c != home])
        dev_bad = f"dev_attack_{aid}"
        ip_bad = f"45.117.{rnd.randint(10, 200)}.{rnd.randint(2, 250)}"
        ts0 = start + timedelta(days=rnd.randint(25, 28),
                                hours=rnd.randint(1, 4))
        for k in range(rnd.randint(2, 3)):
            _add_txn(aid, ts0 + timedelta(hours=k), rnd.uniform(25000, 70000),
                     fraud=1, ring="RING_B", scenario="ring_b_ato",
                     txn_type="transfer", dest=sink_b, mcc="transfer",
                     item_id="TRANSFER_OUT", item_cat="transfer",
                     channel="online", city=other, device=dev_bad, ip=ip_bad)
        gt_acc[aid].update(is_fraud_account=1, ring_id="RING_B",
                           scenario="ring_b_ato")
    accounts.append({"account_id": sink_b, "created_at": start,
                     "home_city": ato_city, "home_country": "India",
                     "segment": "salaried"})
    gt_acc[sink_b] = {"account_id": sink_b, "is_fraud_account": 1,
                      "ring_id": "RING_B", "scenario": "ring_b_sink"}
    state[sink_b] = {"seg": "salaried", "city": ato_city,
                     "devices": [f"dev_{sink_b}"], "ip_base": "10.97.97",
                     "quiet": []}

    # ---- 5. LONE_FRAUD (4 solo fraudsters, no links) --------------------
    for j in range(4):
        aid = f"acc_lone_{j}"
        seg = rnd.choice(["salaried", "student", "retiree", "salaried"])
        city = rnd.choice(CITIES)
        _mk_account(aid, seg, city, start - timedelta(days=rnd.randint(100, 800)),
                    scenario="lone_fraud", fraud=1, ring="",
                    ip_base=f"172.20.{31 + j}",
                    devices=[f"dev_lone_{j}_uniq"])
        st = state[aid]
        for d in range(n_days):
            if rnd.random() < 0.7:
                ts = start + timedelta(days=d, hours=_normal_hour(seg),
                                       minutes=rnd.randint(0, 59))
                _normal_purchase(aid, ts, scenario="legit")
        ts0 = start + timedelta(days=rnd.randint(24, 28),
                                hours=rnd.randint(1, 4))
        for k in range(rnd.randint(8, 12)):  # velocity burst
            _add_txn(aid, ts0 + timedelta(minutes=7 * k),
                     SEG_MEDIAN[seg] * rnd.uniform(5, 12),  # amount spike
                     fraud=1, scenario="lone_fraud", channel="online",
                     city=city, device=f"dev_lone_{j}_uniq",
                     ip=f"172.20.{31 + j}.{100 + k}")

    # ---- 6. BEHAVIOR_CHANGE (3 accounts, switch after day 20) ----------
    legit_ids = [a["account_id"] for a in accounts
                 if a["account_id"].startswith("acc_")
                 and gt_acc[a["account_id"]]["scenario"] == "legit"]
    bc_ids = rnd.sample(legit_ids, 3)
    for aid in bc_ids:
        st = state[aid]
        for d in range(20, n_days):  # switched pattern
            for _ in range(int(rng.poisson(2.0))):
                ts = start + timedelta(days=d, hours=rnd.randint(0, 5),
                                       minutes=rnd.randint(0, 59))
                mcc = rnd.choice(["electronics", "travel", "fashion"])
                _add_txn(aid, ts, _normal_amount(st["seg"]) * rnd.uniform(3, 5),
                         fraud=1, scenario="behavior_change", mcc=mcc,
                         item_cat=mcc, channel="online", city=st["city"],
                         device=_acct_device(aid), ip=_acct_ip(aid))
        gt_acc[aid].update(is_fraud_account=1, ring_id="",
                           scenario="behavior_change")

    # ---- 7. HARD NEGATIVES (all is_fraud=0) ------------------------------
    # 7a. long-standing salaried buying an 85k laptop
    cands = [a["account_id"] for a in accounts
             if a["account_id"].startswith("acc_")
             and state[a["account_id"]]["seg"] == "salaried"
             and (start - pd_to_dt(a["created_at"])).days > 700
             and gt_acc[a["account_id"]]["scenario"] == "legit"]
    for aid in rnd.sample(cands, 4):
        st = state[aid]
        _add_txn(aid, start + timedelta(days=rnd.randint(10, 25), hours=14),
                 85000, mcc="electronics", item_id="LAPTOP_85000",
                 item_cat="electronics", city=st["city"],
                 device=st["devices"][0], ip=_acct_ip(aid))
        gt_acc[aid].update(scenario="hard_neg_laptop")
    for r in gt_txn:
        pass  # (scenarios for hard-neg txns fixed in post-pass below)

    # 7b. small businesses, large regular supplier transfers
    cands = [a["account_id"] for a in accounts
             if a["account_id"].startswith("acc_")
             and state[a["account_id"]]["seg"] == "small_business"
             and gt_acc[a["account_id"]]["scenario"] == "legit"]
    for k, aid in enumerate(rnd.sample(cands, 4)):
        st = state[aid]
        dest = f"acc_supplier_{k}"
        for wk in range(4):
            _add_txn(aid, start + timedelta(days=2 + 7 * wk, hours=11),
                     rnd.uniform(60000, 150000), txn_type="transfer",
                     dest=dest, mcc="transfer", item_id="SUPPLIER_PAYMENT",
                     item_cat="transfer", city=st["city"],
                     device=st["devices"][0], ip=_acct_ip(aid))
        gt_acc[aid].update(scenario="hard_neg_supplier")

    # 7c. travelers: foreign-country txns + new device
    cands = [a["account_id"] for a in accounts
             if a["account_id"].startswith("acc_")
             and gt_acc[a["account_id"]]["scenario"] == "legit"]
    for aid in rnd.sample(cands, 4):
        st = state[aid]
        fcity, fcountry = rnd.choice(FOREIGN)
        for _ in range(rnd.randint(2, 4)):
            d = rnd.randint(5, 27)
            _add_txn(aid, start + timedelta(days=d, hours=rnd.randint(9, 20)),
                     _normal_amount(st["seg"]), mcc="travel",
                     item_cat="travel", city=fcity, country=fcountry,
                     channel="in_app", device=f"dev_travel_{aid}",
                     ip=f"86.96.{rnd.randint(1, 50)}.{rnd.randint(2, 250)}")
        gt_acc[aid].update(scenario="hard_neg_travel")

    # 7d. 3 families sharing one device each (NOT fraud)
    for f in range(3):
        fam_dev = f"dev_family_{f}"
        fam_base = f"10.50.{10 + f}"
        for m in range(3):
            aid = f"acc_family{f}_{m}"
            seg = rnd.choice(["salaried", "student", "retiree"])
            city = CITIES[f % len(CITIES)]
            _mk_account(aid, seg, city,
                        start - timedelta(days=rnd.randint(200, 1200)),
                        scenario="hard_neg_family_device", ip_base=fam_base,
                        devices=[fam_dev])
            for d in range(n_days):
                for _ in range(int(rng.poisson(0.9))):
                    ts = start + timedelta(days=d, hours=_normal_hour(seg),
                                           minutes=rnd.randint(0, 59))
                    _add_txn(aid, ts, _normal_amount(seg), fraud=0,
                             scenario="hard_neg_family_device",
                             mcc=_normal_mcc(seg), item_cat="grocery",
                             city=city, device=fam_dev,
                             ip=f"{fam_base}.{10 + m}")

    # 7e. office/hostel groups sharing one IP (NOT fraud)
    for g in range(2):
        shared_ip = f"203.0.113.{21 + g}"
        for m in range(4):
            aid = f"acc_office{g}_{m}"
            seg = rnd.choice(["salaried", "student"])
            city = CITIES[(g + 3) % len(CITIES)]
            _mk_account(aid, seg, city,
                        start - timedelta(days=rnd.randint(120, 900)),
                        scenario="hard_neg_shared_ip",
                        devices=[f"dev_office{g}_{m}"])
            for d in range(n_days):
                for _ in range(int(rng.poisson(0.9))):
                    ts = start + timedelta(days=d, hours=_normal_hour(seg),
                                           minutes=rnd.randint(0, 59))
                    _add_txn(aid, ts, _normal_amount(seg), fraud=0,
                             scenario="hard_neg_shared_ip",
                             mcc=_normal_mcc(seg), item_cat="food",
                             city=city, device=f"dev_office{g}_{m}",
                             ip=shared_ip)

    # 7f. festival-week spike accounts (NOT fraud)
    cands = [a["account_id"] for a in accounts
             if a["account_id"].startswith("acc_")
             and gt_acc[a["account_id"]]["scenario"] == "legit"]
    for aid in rnd.sample(cands, 8):
        st = state[aid]
        for _ in range(rnd.randint(6, 10)):
            d = rnd.randint(13, 19)
            ts = start + timedelta(days=d, hours=rnd.randint(10, 21),
                                   minutes=rnd.randint(0, 59))
            mcc = rnd.choice(["fashion", "electronics", "grocery"])
            _add_txn(aid, ts, _normal_amount(st["seg"]) * rnd.uniform(1.2, 2.5),
                     mcc=mcc, item_cat=mcc, city=st["city"],
                     device=_acct_device(aid), ip=_acct_ip(aid))
        gt_acc[aid].update(scenario="hard_neg_festival")

    # 7g. new legit accounts, few normal purchases (NOT fraud)
    for j in range(5):
        aid = f"acc_newlegit_{j}"
        seg = rnd.choice(SEGMENTS)
        city = rnd.choice(CITIES)
        _mk_account(aid, seg, city,
                    start + timedelta(days=rnd.randint(2, 20)),
                    scenario="hard_neg_new_legit")
        for _ in range(rnd.randint(3, 8)):
            d = rnd.randint(0, n_days - 1)
            ts = start + timedelta(days=d, hours=_normal_hour(seg),
                                   minutes=rnd.randint(0, 59))
            _add_txn(aid, ts, _normal_amount(seg), fraud=0,
                     scenario="hard_neg_new_legit", mcc=_normal_mcc(seg),
                     item_cat="food", city=city,
                     device=state[aid]["devices"][0], ip=_acct_ip(aid))

    # ---- post-pass: hard-neg account txns carry their scenario ---------
    hard_neg_acc_scen = {aid: row["scenario"] for aid, row in gt_acc.items()
                         if row["scenario"].startswith("hard_neg")}
    txn_scen_fix = {}  # txn counted per account below via txns list order
    for t, g in zip(txns, gt_txn):
        aid = t["account_id"]
        if aid in hard_neg_acc_scen and g["is_fraud"] == 0 \
                and g["scenario"] == "legit":
            g["scenario"] = hard_neg_acc_scen[aid]
    void = (txn_scen_fix, fake, config)

    # ---- assemble + write ----------------------------------------------
    acc_df = pd.DataFrame(accounts)
    txn_df = pd.DataFrame(txns)
    txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
    acc_df["created_at"] = pd.to_datetime(acc_df["created_at"])
    txn_df = txn_df.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    gt_txn_df = pd.DataFrame(gt_txn)
    gt_acc_df = pd.DataFrame(list(gt_acc.values()))

    # exact column order
    txn_df = txn_df[TXN_COLUMNS]
    acc_df = acc_df[ACC_COLUMNS]
    gt_txn_df = gt_txn_df[GT_TXN_COLUMNS]
    gt_acc_df = gt_acc_df[GT_ACC_COLUMNS]

    acc_df.to_csv(f"{raw_dir}/accounts.csv", index=False)
    txn_df.to_csv(f"{raw_dir}/transactions.csv", index=False)
    gt_txn_df.to_csv(f"{raw_dir}/ground_truth_txn.csv", index=False)
    gt_acc_df.to_csv(f"{raw_dir}/ground_truth_accounts.csv", index=False)

    acc_counts = Counter(gt_acc_df["scenario"])
    txn_counts = Counter(gt_txn_df["scenario"])
    print(f"[generate_data] seed={seed} days={n_days}: "
          f"{len(acc_df)} accounts, {len(txn_df)} txns "
          f"({int((gt_txn_df['is_fraud'] == 1).sum())} fraud) -> {raw_dir}")
    print("  accounts per scenario:")
    for sc, c in sorted(acc_counts.items()):
        print(f"    {sc}: {c}")
    print("  fraud txns per scenario:")
    for sc, c in sorted(txn_counts.items()):
        if (gt_txn_df.loc[gt_txn_df.scenario == sc, "is_fraud"] == 1).any():
            print(f"    {sc}: {c} "
                  f"({int(((gt_txn_df.scenario == sc) & (gt_txn_df.is_fraud == 1)).sum())} fraud)")

    return {"accounts": acc_df, "transactions": txn_df,
            "gt_txn": gt_txn_df, "gt_acc": gt_acc_df,
            "account_scenarios": dict(acc_counts),
            "txn_scenarios": dict(txn_counts)}


def pd_to_dt(x):
    return pd.to_datetime(x)


# --------------------------------------------------------------------------
# Quick eyeball plot: transaction amounts by account type
# --------------------------------------------------------------------------

def save_amount_scatter(transactions=None, ground_truth_txn=None,
                        ground_truth_accounts=None, accounts=None,
                        transactions_csv: str | None = None,
                        out_path: str = "amount_scatter.png",
                        seed: int = 42) -> str:
    """Save a PNG scatter of log(amount) vs time, colored by account type.

    Account type = fraud scenario of the account if any, else its segment.
    Accepts DataFrames or CSV paths. Returns the output path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if transactions is None and transactions_csv is not None:
        transactions = pd.read_csv(transactions_csv)
    if isinstance(transactions, str):
        transactions = pd.read_csv(transactions)
    txns = transactions.copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    if ground_truth_accounts is None and accounts is not None:
        acc = accounts.copy() if not isinstance(accounts, str) else \
            pd.read_csv(accounts)
        label = {r.account_id: f"seg:{r.segment}"
                 for r in acc.itertuples()}
    else:
        gta = ground_truth_accounts
        if isinstance(gta, str):
            gta = pd.read_csv(gta)
        label = {}
        for r in gta.itertuples():
            label[r.account_id] = (r.scenario if r.is_fraud_account == 1
                                   or str(r.scenario).startswith("hard_neg")
                                   else "legit")
    if ground_truth_txn is not None:
        gtt = ground_truth_txn if not isinstance(ground_truth_txn, str) \
            else pd.read_csv(ground_truth_txn)
        fraud_ids = set(gtt.loc[gtt.is_fraud == 1, "txn_id"])
        txns["_fraud"] = txns["txn_id"].isin(fraud_ids)
    else:
        txns["_fraud"] = False
    txns["_grp"] = txns["account_id"].map(label).fillna("legit")

    rng = np.random.default_rng(seed)
    groups = sorted(txns["_grp"].unique())
    cmap = plt.get_cmap("tab20")
    gcolor = {g: cmap(i % 20) for i, g in enumerate(groups)}

    fig, ax = plt.subplots(figsize=(12, 5))
    for g in groups:
        sub = txns[txns["_grp"] == g]
        if len(sub) > 2500:  # thin out for readability (deterministic)
            sub = sub.sample(2500, random_state=seed)
        x = pd.to_datetime(sub["timestamp"]).astype("int64")
        jitter = rng.normal(0, 0.6, size=len(sub))
        ax.scatter(x, np.log10(sub["amount"].clip(lower=1) + jitter * 0 + 1),
                   s=6, alpha=0.5, c=[gcolor[g]], label=f"{g} (n={len(sub)})")
    fraud = txns[txns["_fraud"]]
    if len(fraud):
        ax.scatter(pd.to_datetime(fraud["timestamp"]).astype("int64"),
                   np.log10(fraud["amount"].clip(lower=1) + 1),
                   s=14, alpha=0.9, c="red", marker="x", label="fraud txn")
    ax.set_ylabel("log10(amount INR + 1)")
    ax.set_xlabel("timestamp")
    ax.set_title("Transaction amounts by account type (red x = fraud txn)")
    ax.legend(markerscale=2, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[generate_data] scatter -> {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic fraud data")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-accounts", type=int, default=560)
    p.add_argument("--n-days", type=int, default=30)
    p.add_argument("--plot", default=None,
                   help="optional PNG path for the amount scatter")
    args = p.parse_args()
    out = generate_all(args.raw_dir, seed=args.seed,
                       n_accounts=args.n_accounts, n_days=args.n_days)
    if args.plot:
        save_amount_scatter(out["transactions"], out["gt_txn"], out["gt_acc"],
                            out_path=args.plot, seed=args.seed)


if __name__ == "__main__":
    main()
