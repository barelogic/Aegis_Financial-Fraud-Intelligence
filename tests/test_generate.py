"""Tests for src/generate_data.py.

All generation happens inside pytest's tmp_path (a separate subdirectory per
test session), so these tests never touch or overwrite data/raw/ and cannot
interfere with the pipeline or other test modules.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_data import (ACC_COLUMNS, GT_ACC_COLUMNS, GT_TXN_COLUMNS,
                           TXN_COLUMNS, generate_all)

SEED = 42


def _subnet(ip: str) -> str:
    return ip.rsplit(".", 1)[0]


@pytest.fixture(scope="module")
def gen(tmp_path_factory):
    """Generate once into an isolated subdirectory; share across tests."""
    out = tmp_path_factory.mktemp("gen_data")
    res = generate_all(str(out), seed=SEED)
    return out, res


@pytest.fixture(scope="module")
def frames(gen):
    out, _ = gen
    txns = pd.read_csv(out / "transactions.csv")
    accs = pd.read_csv(out / "accounts.csv")
    gtt = pd.read_csv(out / "ground_truth_txn.csv")
    gta = pd.read_csv(out / "ground_truth_accounts.csv")
    return txns, accs, gtt, gta


def test_schema_columns_present(frames):
    txns, accs, gtt, gta = frames
    assert list(txns.columns) == TXN_COLUMNS
    assert list(accs.columns) == ACC_COLUMNS
    assert list(gtt.columns) == GT_TXN_COLUMNS
    assert list(gta.columns) == GT_ACC_COLUMNS
    assert (txns["currency"] == "INR").all()


def test_no_duplicate_txn_id(frames):
    txns, _, gtt, _ = frames
    assert txns["txn_id"].is_unique
    assert gtt["txn_id"].is_unique
    assert set(gtt["txn_id"]) == set(txns["txn_id"])


def test_dataset_scale(frames):
    txns, accs, _, _ = frames
    assert 550 <= len(accs) <= 700, f"{len(accs)} accounts"
    assert 15000 <= len(txns) <= 25000, f"{len(txns)} txns"


def _ring_a_accs(frames):
    _, _, _, gta = frames
    return sorted(gta.loc[gta["scenario"] == "ring_a", "account_id"])


def test_ring_a_exactly_10_accounts(frames):
    assert len(_ring_a_accs(frames)) == 10


def test_ring_a_no_shared_device_or_subnet(frames):
    txns, _, gtt, _ = frames
    ring_a = _ring_a_accs(frames)
    fraud_ids = set(gtt.loc[(gtt["scenario"] == "ring_a")
                            & (gtt["is_fraud"] == 1), "txn_id"])
    sub = txns[txns["txn_id"].isin(fraud_ids)]
    assert len(sub) > 0
    dev_by_acc = {a: set(sub.loc[sub.account_id == a, "device_id"])
                  for a in ring_a}
    sub_by_acc = {a: {_subnet(i) for i in
                      sub.loc[sub.account_id == a, "ip_address"]} for a in ring_a}
    for i, a in enumerate(ring_a):
        for b in ring_a[i + 1:]:
            assert dev_by_acc[a].isdisjoint(dev_by_acc[b]), f"{a} x {b} share device"
            assert sub_by_acc[a].isdisjoint(sub_by_acc[b]), f"{a} x {b} share subnet"


def test_ring_a_amounts_inside_segment_normal_range(frames):
    txns, accs, gtt, _ = frames
    seg = dict(zip(accs["account_id"], accs["segment"]))
    legit_ids = set(gtt.loc[(gtt["is_fraud"] == 0)
                            & (gtt["scenario"] == "legit"), "txn_id"])
    legit = txns[txns["txn_id"].isin(legit_ids)]
    legit = legit[legit["txn_type"] == "purchase"]
    bounds = {}
    for s, grp in legit.groupby(legit["account_id"].map(seg)):
        bounds[s] = (grp["amount"].quantile(0.005),
                     grp["amount"].quantile(0.995))
    ring_ids = set(gtt.loc[(gtt["scenario"] == "ring_a")
                           & (gtt["is_fraud"] == 1), "txn_id"])
    ring = txns[(txns["txn_id"].isin(ring_ids))
                & (txns["txn_type"] == "purchase")]
    assert len(ring) >= 20  # 10 accounts x 2 rare purchases
    for r in ring.itertuples():
        lo, hi = bounds[seg[r.account_id]]
        assert lo <= r.amount <= hi, \
            f"{r.txn_id} amount {r.amount} outside {seg[r.account_id]} [{lo:.0f},{hi:.0f}]"


def test_ring_a_shares_rare_items_and_destination(frames):
    txns, _, gtt, gta = frames
    ring_a = _ring_a_accs(frames)
    ring_ids = set(gtt.loc[(gtt["scenario"] == "ring_a")
                           & (gtt["is_fraud"] == 1), "txn_id"])
    ring = txns[txns["txn_id"].isin(ring_ids)]
    # rare = bought by <=2 legit-buyer accounts ...
    legit_ids = set(gtt.loc[(gtt["is_fraud"] == 0)
                            & (gtt["scenario"] == "legit"), "txn_id"])
    legit_buyers = (txns[txns["txn_id"].isin(legit_ids)]
                    .groupby("item_id")["account_id"].nunique())
    buyers_by_item = ring.groupby("item_id")["account_id"].nunique()
    shared_rare = [it for it, n in buyers_by_item.items()
                   if n >= 8 and legit_buyers.get(it, 0) <= 2]
    assert len(shared_rare) >= 2, f"only {shared_rare}"
    # ... and at least one destination shared across the ring
    tr = ring[ring["txn_type"] == "transfer"]
    assert len(tr) > 0
    dest_users = tr.groupby("dest_account_id")["account_id"].nunique()
    assert (dest_users >= 8).any(), dest_users.to_dict()
    # order check: all accounts buy the rare items in the same order
    # (determine the canonical order from the first account, then enforce)
    ordered = ring.sort_values("timestamp").groupby("account_id")["item_id"].apply(list)
    canon = [it for it in ordered.iloc[0] if it in set(shared_rare)]
    assert len(canon) >= 2
    for aid, seq in ordered.items():
        seq_rare = [it for it in seq if it in set(shared_rare)]
        assert seq_rare[:2] == canon[:2], f"{aid}: {seq_rare[:2]} != {canon[:2]}"


def test_family_device_accounts_not_fraud(frames):
    _, _, gtt, gta = frames
    fam = gta[gta["scenario"] == "hard_neg_family_device"]
    assert len(fam) >= 6  # 3 families
    assert (fam["is_fraud_account"] == 0).all()
    # every txn of a family account is labelled is_fraud=0
    txns, _, _, _ = frames
    fam_ids = set(fam["account_id"])
    fam_tids = set(txns.loc[txns["account_id"].isin(fam_ids), "txn_id"])
    sub = gtt[gtt["txn_id"].isin(fam_tids)]
    assert len(sub) == len(fam_tids) and (sub["is_fraud"] == 0).all()
    # they really do share devices (test is meaningful)
    devs = txns.loc[txns["account_id"].isin(fam_ids), ["account_id", "device_id"]]
    assert (devs.groupby("device_id")["account_id"].nunique() >= 2).any()


def test_hard_negatives_present_and_clean(frames):
    _, _, gtt, gta = frames
    hn = gta[gta["scenario"].str.startswith("hard_neg")]
    assert len(hn) >= 25, f"only {len(hn)} hard negatives"
    assert (hn["is_fraud_account"] == 0).all()
    expected = {"hard_neg_laptop", "hard_neg_supplier", "hard_neg_travel",
                "hard_neg_family_device", "hard_neg_shared_ip",
                "hard_neg_festival", "hard_neg_new_legit"}
    assert expected.issubset(set(gta["scenario"])), set(gta["scenario"])
