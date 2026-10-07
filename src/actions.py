"""Recommended actions per flagged item (baseline).

Transaction bands -> allow / review / step_up_auth / block (+ ring review).
Account bands   -> monitor / review / restrict / freeze.
"""
from __future__ import annotations

import pandas as pd

TXN_ACTIONS = {"low": "allow", "medium": "review",
               "high": "step_up_auth", "critical": "block"}
ACC_ACTIONS = {"low": "monitor", "medium": "review",
               "high": "restrict", "critical": "freeze"}


def recommend_txn_action(band: str, in_ring: bool) -> str:
    """Action for one transaction band (ring members get at least review)."""
    action = TXN_ACTIONS.get(band, "review")
    if in_ring and action == "allow":
        return "review"
    return action


def recommend_account_action(band: str, in_ring: bool) -> str:
    """Action for one account band (ring members get at least review)."""
    action = ACC_ACTIONS.get(band, "review")
    if in_ring and action == "monitor":
        return "review"
    return action


def apply_actions(scored_txns: pd.DataFrame, scored_accs: pd.DataFrame,
                  rings: list[dict] | None, config: dict,
                  level: str = "transaction") -> pd.DataFrame:
    """(Re)assign txn_action or account_action using bands + ring membership."""
    ring_members: set[str] = set()
    for r in rings or []:
        ring_members.update(str(a) for a in r.get("accounts", []))
    if level == "transaction":
        out = scored_txns.copy()
        out["txn_action"] = [
            recommend_txn_action(b, str(a) in ring_members)
            for b, a in zip(out["txn_risk_band"], out["account_id"])]
        return out
    out = scored_accs.copy()
    out["account_action"] = [
        recommend_account_action(b, str(a) in ring_members)
        for b, a in zip(out["account_risk_band"], out["account_id"])]
    return out
