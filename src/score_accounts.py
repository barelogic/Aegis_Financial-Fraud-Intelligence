"""Account risk scoring: blend of worst / typical-bad transactions + ring bump.

    base = w_max * max_txn_score + w_top3 * mean(top-3 txn scores)
    final = min(99, base + ring_bump) for ring members, with a reason sentence
    like "Member of ring_1 with 9 other accounts".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _band(score: float, th: dict) -> str:
    if score >= th.get("critical", 85):
        return "critical"
    if score >= th.get("high", 65):
        return "high"
    if score >= th.get("medium", 40):
        return "medium"
    return "low"


def _action(band: str) -> str:
    return {"low": "monitor", "medium": "review",
            "high": "restrict", "critical": "freeze"}[band]


def aggregate_transactions(scored_txns: pd.DataFrame) -> pd.DataFrame:
    """Per-account aggregates: max score, top-3 mean, counts, worst txn id."""
    rows = []
    for acct, g in scored_txns.groupby("account_id"):
        s = g["txn_risk_score"].astype(float).sort_values(ascending=False)
        top3 = s.head(3)
        worst_idx = s.index[0]
        rows.append({
            "account_id": acct,
            "n_txns": len(g),
            "max_score": float(s.max()),
            "top3_mean": float(top3.mean()),
            "n_high": int((s >= 65).sum()),
            "worst_txn_id": str(g.loc[worst_idx, "txn_id"]),
        })
    return pd.DataFrame(rows)


def base_account_score(agg: pd.DataFrame, config: dict) -> pd.Series:
    """Weighted blend of max and top-3 mean (0-100)."""
    w = config.get("account_scoring", {})
    w_max, w_top3 = float(w.get("w_max", 0.6)), float(w.get("w_top3", 0.4))
    return (w_max * agg["max_score"] + w_top3 * agg["top3_mean"]).clip(0, 100)


def build_account_reasons(agg_row: pd.Series, member_txns: pd.DataFrame,
                          feats: pd.DataFrame | None) -> str:
    """Concrete evidence sentences for one account (never empty)."""
    parts: list[str] = []
    parts.append(f"Highest-risk transaction {agg_row['worst_txn_id']} scored "
                 f"{agg_row['max_score']:.0f}/100")
    parts.append(f"Mean of top-3 transaction scores is {agg_row['top3_mean']:.0f}")
    if int(agg_row["n_high"]) > 0:
        parts.append(f"{int(agg_row['n_high'])} transaction(s) scored high/critical")
    if feats is not None and len(member_txns):
        fsub = feats[feats["txn_id"].isin(set(member_txns["txn_id"]))]
        if len(fsub):
            if int(fsub["is_new_device"].sum()) > 0:
                parts.append(f"{int(fsub['is_new_device'].sum())} transaction(s) "
                             f"used a first-time device")
            if int(fsub["is_new_city"].sum()) > 0:
                parts.append(f"{int(fsub['is_new_city'].sum())} transaction(s) "
                             f"from a new city")
            if int(fsub["dest_fanin"].max()) >= 3:
                parts.append(f"Sent to a destination with fan-in "
                             f"{int(fsub['dest_fanin'].max())}")
    if not parts:
        parts.append(f"Scored from {int(agg_row['n_txns'])} transaction(s); "
                     f"no strong risk signals")
    return "; ".join(parts[:6])


def score_accounts(scored_txns: pd.DataFrame, features: pd.DataFrame | None,
                   accounts: pd.DataFrame, rings: list[dict] | None,
                   config: dict) -> pd.DataFrame:
    """Score every account 0-100 with band, ring_id, reasons, action."""
    th = config.get("risk_thresholds", {})
    agg = aggregate_transactions(scored_txns)
    agg["account_risk_score"] = base_account_score(agg, config).round(1)
    agg["ring_id"] = ""

    feat = features  # may be None in tests
    reasons: dict[str, str] = {}
    for _, r in agg.iterrows():
        mem = scored_txns[scored_txns["account_id"] == r["account_id"]]
        reasons[r["account_id"]] = build_account_reasons(r, mem, feat)
    agg["account_reasons"] = agg["account_id"].map(reasons)
    agg["account_risk_band"] = agg["account_risk_score"].map(
        lambda s: _band(float(s), th))
    agg["account_action"] = agg["account_risk_band"].map(_action)

    # Include accounts with zero transactions (score 0, benign reason).
    all_accts = set(accounts["account_id"].astype(str))
    missing = all_accts - set(agg["account_id"].astype(str))
    extra = [{"account_id": a, "account_risk_score": 0.0, "ring_id": "",
              "account_reasons": "No transactions observed for this account",
              "account_risk_band": "low", "account_action": "monitor"}
             for a in missing]
    if extra:
        agg = pd.concat([agg, pd.DataFrame(extra)], ignore_index=True)

    if rings:
        agg = apply_ring_bump(agg, rings, config)
    cols = ["account_id", "account_risk_score", "account_risk_band",
            "ring_id", "account_reasons", "account_action"]
    return agg[cols].sort_values("account_risk_score",
                                 ascending=False).reset_index(drop=True)


def apply_ring_bump(accounts_df: pd.DataFrame, rings: list[dict],
                    config: dict) -> pd.DataFrame:
    """Add ring-membership bump + reason sentence (called after detect_rings).

    Adds a sentence like "Member of ring_1 with 9 other accounts
    (shared devices, shared destination)".
    """
    bump = float(config.get("account_scoring", {}).get("ring_bump", 20))
    th = config.get("risk_thresholds", {})
    member_to_ring: dict[str, dict] = {}
    for r in rings:
        for a in r.get("accounts", []):
            member_to_ring[str(a)] = r
    out = accounts_df.copy()
    for i, (_, row) in enumerate(out.iterrows()):
        ring = member_to_ring.get(str(row["account_id"]))
        if ring:
            others = max(0, int(ring.get("size", len(ring.get("accounts", [])))) - 1)
            links = ", ".join(ring.get("link_types_found", [])[:3]) or "shared links"
            out.loc[out.index[i], "ring_id"] = ring["ring_id"]
            out.loc[out.index[i], "account_risk_score"] = min(
                99.0, float(row["account_risk_score"]) + bump)
            prev = str(row.get("account_reasons", ""))
            extra = (f"Member of {ring['ring_id']} with {others} other "
                     f"account(s) ({links})")
            out.loc[out.index[i], "account_reasons"] = (
                f"{prev}; {extra}" if prev else extra)
    out["account_risk_band"] = out["account_risk_score"].map(
        lambda s: _band(float(s), th))
    out["account_action"] = out["account_risk_band"].map(_action)
    return out.sort_values("account_risk_score",
                           ascending=False).reset_index(drop=True)
