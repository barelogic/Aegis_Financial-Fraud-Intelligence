"""Behavior-relative transaction features.

Every feature uses ONLY data strictly before each transaction's timestamp,
so the pipeline can run in real time. Rows are processed in (timestamp,
txn_id) order and all history state is updated AFTER the current row's
features are computed (no future leakage).

Feature columns produced (see FEATURE_COLUMNS):
  amount_zscore, amount_ratio, vel_cnt_1h, vel_amt_1h, vel_cnt_24h,
  vel_amt_24h, is_new_device, is_new_city, is_new_mcc, is_new_destination,
  hour_unusualness, account_age_days, is_transfer_out, burst_transfer_frac,
  dest_fanin, device_sharing, item_rarity
"""
from __future__ import annotations

import random
from collections import defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml

FEATURE_COLUMNS = [
    "amount_zscore",
    "amount_ratio",
    "vel_cnt_1h",
    "vel_amt_1h",
    "vel_cnt_24h",
    "vel_amt_24h",
    "is_new_device",
    "is_new_city",
    "is_new_mcc",
    "is_new_destination",
    "hour_unusualness",
    "account_age_days",
    "is_transfer_out",
    "burst_transfer_frac",
    "dest_fanin",
    "device_sharing",
    "item_rarity",
]

MIN_HISTORY = 5  # need >=5 prior txns before per-account stats are trusted


def set_seed(seed: int) -> None:
    """Fix Python + numpy randomness."""
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str = "config.yaml") -> dict:
    """Load the YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def col(mapping: dict, logical: str) -> str:
    """Resolve a logical schema name to the dataset's column name."""
    return mapping.get(logical, logical)


def load_inputs(transactions_path: str, accounts_path: str,
                config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load transactions/accounts CSVs and parse datetimes.

    Uses config['column_mapping'] so a renamed dataset can be plugged in:
    columns are renamed back to the logical schema names used in code.
    """
    mapping: dict = config.get("column_mapping", {})
    # Invert: dataset column -> logical name.
    inv = {v: k for k, v in mapping.items()}
    txns = pd.read_csv(transactions_path)
    accs = pd.read_csv(accounts_path)
    txns = txns.rename(columns={c: inv.get(c, c) for c in txns.columns})
    accs = accs.rename(columns={c: inv.get(c, c) for c in accs.columns})
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    accs["created_at"] = pd.to_datetime(accs["created_at"])
    return txns, accs


def _is_missing(val) -> bool:
    """True for NaN / None / empty-string values."""
    return val is None or (isinstance(val, float) and np.isnan(val)) or val == ""


def init_state() -> dict:
    """Create an empty causal-history state (all 'so far' containers)."""
    return {
        "amounts": defaultdict(list),
        "hist_times": defaultdict(list),   # per-account timestamps
        "hist_amounts": defaultdict(list),
        "devices": defaultdict(set),
        "cities": defaultdict(set),
        "mccs": defaultdict(set),
        "dests": defaultdict(set),
        "hour_counts": defaultdict(lambda: np.zeros(24, dtype=int)),
        "seg_amounts": defaultdict(list),  # segment fallback
        "global_amounts": [],
        "device_users": defaultdict(set),  # device -> accounts so far
        "dest_senders": defaultdict(set),  # dest -> sender accounts
        "item_buyers": defaultdict(set),   # item -> buyer accounts
    }


def build_account_index(accounts: pd.DataFrame) -> dict:
    """Map account_id -> {created_at, segment} for feature computation."""
    acct_info: dict = {}
    for _, r in accounts.iterrows():
        acct_info[r["account_id"]] = {
            "created_at": pd.to_datetime(r["created_at"])
            if not _is_missing(r.get("created_at")) else None,
            "segment": r.get("segment", "default"),
        }
    return acct_info


def register_account(acct_info: dict, account_id: str, created_at=None,
                     segment: str = "default") -> dict:
    """Add a previously unseen account (first-seen live ingestion).

    created_at defaults to None (= age 0 at first transaction); segment
    comes from the payload when provided, else 'default'.
    """
    if account_id not in acct_info:
        acct_info[account_id] = {"created_at": created_at, "segment": segment}
    return acct_info[account_id]


def features_for_row(r: pd.Series, state: dict, acct_info: dict) -> dict:
    """Compute behavior-relative features for ONE row from prior-only state.

    Pure read of `state` (no mutation): call `update_state` afterwards.
    `r` needs txn_id/account_id/timestamp/amount (+ optional raw fields).
    """
    one_hour = timedelta(hours=1)
    one_day = timedelta(hours=24)
    amounts = state["amounts"]
    hist_times = state["hist_times"]
    hist_amounts = state["hist_amounts"]
    devices = state["devices"]
    cities = state["cities"]
    mccs = state["mccs"]
    dests = state["dests"]
    hour_counts = state["hour_counts"]
    seg_amounts = state["seg_amounts"]
    global_amounts = state["global_amounts"]
    device_users = state["device_users"]
    dest_senders = state["dest_senders"]
    item_buyers = state["item_buyers"]

    acct = r["account_id"]
    ts = r["timestamp"]
    amt = float(r["amount"])
    info = acct_info.get(acct, {})
    created = info.get("created_at", ts)
    if created is None:
        created = ts
    segment = info.get("segment", "default")

    prior = amounts[acct]
    n_prior = len(prior)

    # -- amount vs own history (segment-median fallback if <5 prior) --
    if n_prior >= MIN_HISTORY:
        mean = float(np.mean(prior))
        std = float(np.std(prior, ddof=1)) if n_prior >= 2 else 0.0
        std = max(std, 1.0)  # avoid exploding z on tiny std
        med = float(np.median(prior))
        z = (amt - mean) / std
        ratio = amt / med if med > 0 else amt
    else:
        # Causal fallback: median of prior rows in the same segment
        # (only rows already processed = strictly before this timestamp).
        seg_hist = seg_amounts.get(segment, [])
        if len(seg_hist) >= MIN_HISTORY:
            seg_med = float(np.median(seg_hist))
        elif global_amounts:
            seg_med = float(np.median(global_amounts))
        else:
            seg_med = amt  # very first rows: neutral ratio of 1
        z = 0.0
        ratio = amt / seg_med if seg_med > 0 else 1.0
    z = float(np.clip(z, -10.0, 10.0))
    ratio = float(np.clip(ratio, 0.0, 1000.0))

    # -- velocity from prior rows only --
    times = hist_times[acct]
    amts = hist_amounts[acct]
    c1 = s1 = c24 = s24 = 0
    # Walk backwards; histories are chronological so we can break early.
    for t_prev, a_prev in zip(reversed(times), reversed(amts)):
        delta = ts - t_prev
        if delta > one_day:
            break
        c24 += 1
        s24 += a_prev
        if delta <= one_hour:
            c1 += 1
            s1 += a_prev

    # -- novelty flags (1 if never seen before for this account) --
    dev, city = r.get("device_id"), r.get("city")
    mcc, dest = r.get("merchant_category"), r.get("dest_account_id")
    new_dev = 0 if _is_missing(dev) else int(dev not in devices[acct])
    new_city = 0 if _is_missing(city) else int(city not in cities[acct])
    new_mcc = 0 if _is_missing(mcc) else int(mcc not in mccs[acct])
    new_dest = 0 if _is_missing(dest) else int(dest not in dests[acct])

    # -- hour unusualness vs account norm (0 if <5 prior) --
    hr = int(ts.hour)
    if n_prior >= MIN_HISTORY:
        tot = int(hour_counts[acct].sum())
        hour_unusual = 1.0 - (float(hour_counts[acct][hr]) / tot if tot else 0.0)
    else:
        hour_unusual = 0.0

    # -- account age --
    try:
        age_days = max(0, (ts - pd.to_datetime(created)).days)
    except Exception:
        age_days = 0

    # -- transfer / burst cash-out approximation --
    txn_type = str(r.get("txn_type", "purchase"))
    is_out = int(txn_type in ("transfer", "cash_out"))
    if is_out and s24 > 0:
        burst_frac = amt / (s24 + amt)
    else:
        burst_frac = 0.0

    # -- destination fan-in / device sharing / item rarity (prior only) --
    fanin = 0 if _is_missing(dest) else len(dest_senders.get(dest, set()))
    # Exclude self from device-sharing count (re-login != sharing).
    sharing = 0 if _is_missing(dev) else len(
        {u for u in device_users.get(dev, set()) if u != acct})
    item = r.get("item_id")
    if _is_missing(item):
        rarity = 0.0
    else:
        rarity = 1.0 / (1.0 + len(item_buyers.get(item, set())))

    return {
        "txn_id": r["txn_id"], "account_id": acct, "timestamp": ts,
        "amount_zscore": z, "amount_ratio": ratio,
        "vel_cnt_1h": c1, "vel_amt_1h": float(s1),
        "vel_cnt_24h": c24, "vel_amt_24h": float(s24),
        "is_new_device": new_dev, "is_new_city": new_city,
        "is_new_mcc": new_mcc, "is_new_destination": new_dest,
        "hour_unusualness": float(hour_unusual),
        "account_age_days": int(age_days),
        "is_transfer_out": is_out,
        "burst_transfer_frac": float(np.clip(burst_frac, 0.0, 1.0)),
        "dest_fanin": int(fanin),
        "device_sharing": int(sharing),
        "item_rarity": float(rarity),
    }


def update_state(state: dict, r: pd.Series, segment: str = "default") -> None:
    """Fold one row into history state AFTER its features are computed."""
    acct = r["account_id"]
    ts = r["timestamp"]
    amt = float(r["amount"])
    dev, city = r.get("device_id"), r.get("city")
    mcc, dest = r.get("merchant_category"), r.get("dest_account_id")
    item = r.get("item_id")
    try:
        hr = int(ts.hour)
    except Exception:
        hr = 0
    state["amounts"][acct].append(amt)
    state["hist_times"][acct].append(ts)
    state["hist_amounts"][acct].append(amt)
    if not _is_missing(dev):
        state["devices"][acct].add(dev)
        state["device_users"][dev].add(acct)
    if not _is_missing(city):
        state["cities"][acct].add(city)
    if not _is_missing(mcc):
        state["mccs"][acct].add(mcc)
    if not _is_missing(dest):
        state["dests"][acct].add(dest)
        state["dest_senders"][dest].add(acct)
    if not _is_missing(item):
        state["item_buyers"][item].add(acct)
    state["hour_counts"][acct][hr] += 1
    state["seg_amounts"][segment].append(amt)
    state["global_amounts"].append(amt)


def compute_features(transactions: pd.DataFrame, accounts: pd.DataFrame,
                     config: dict) -> pd.DataFrame:
    """Compute causal behavior-relative features, one row per transaction.

    Args:
        transactions: raw transactions (logical schema names).
        accounts: accounts table with created_at + segment.
        config: config dict (uses config['seed']).

    Returns:
        DataFrame with txn_id, account_id, timestamp + FEATURE_COLUMNS,
        sorted by (timestamp, txn_id).
    """
    set_seed(int(config.get("seed", 42)))
    txns = transactions.copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])
    # Deterministic tie-break so reruns are identical.
    txns = txns.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)

    acct_info = build_account_index(accounts)
    # Running history state (all "so far", i.e. strictly prior rows).
    state = init_state()

    out_rows: list[dict] = []
    for _, r in txns.iterrows():
        # Compute from prior-only state, then fold the row in (causal).
        out_rows.append(features_for_row(r, state, acct_info))
        segment = acct_info.get(
            r["account_id"], {}).get("segment", "default")
        update_state(state, r, segment)

    feats = pd.DataFrame(out_rows)
    return feats[["txn_id", "account_id", "timestamp"] + FEATURE_COLUMNS]


def get_feature_columns() -> list[str]:
    """Return the ordered feature column names."""
    return list(FEATURE_COLUMNS)
