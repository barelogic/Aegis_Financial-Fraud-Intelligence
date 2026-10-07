"""Evaluation against ground truth (baseline).

Reports transaction precision/recall at high/critical, account ranking
quality, rings found, and the alert-budget check (precision guard for
legit high-value transactions).
"""
from __future__ import annotations

import pandas as pd


def evaluate(scored_txns: pd.DataFrame, scored_accs: pd.DataFrame,
             rings: list[dict] | None, gt_txn: pd.DataFrame | None,
             gt_acc: pd.DataFrame | None, config: dict) -> dict:
    """Compute metrics dict (also saved as metrics.json by run_all)."""
    th = config.get("risk_thresholds", {})
    metrics: dict = {}
    if gt_txn is not None and len(gt_txn):
        lab = gt_txn.set_index("txn_id")["is_fraud"].reindex(
            scored_txns["txn_id"]).fillna(0).astype(int)
        pred = (scored_txns.set_index("txn_id")["txn_risk_score"]
                >= th.get("high", 65)).astype(int).values
        tp = int(((pred == 1) & (lab.values == 1)).sum())
        fp = int(((pred == 1) & (lab.values == 0)).sum())
        fn = int(((pred == 0) & (lab.values == 1)).sum())
        metrics["txn_precision"] = round(tp / max(tp + fp, 1), 4)
        metrics["txn_recall"] = round(tp / max(tp + fn, 1), 4)
        metrics["txn_flagged"] = int(pred.sum())
    if gt_acc is not None and len(gt_acc):
        fraud_accts = set(gt_acc[gt_acc["is_fraud_account"] == 1]["account_id"])
        ranked = scored_accs.sort_values("account_risk_score", ascending=False)
        top_k = ranked.head(max(len(fraud_accts), 10))["account_id"].astype(str)
        metrics["account_recall_at_k"] = round(
            len(set(top_k) & {str(a) for a in fraud_accts}) / max(len(fraud_accts), 1), 4)
    gt_rings = set()
    if gt_txn is not None and "ring_id" in gt_txn.columns:
        # NB: read_csv parses empty ring_id as NaN, and bool(nan) is True,
        # so filter NaN explicitly or phantom rings inflate the count.
        gt_rings = {r for r in gt_txn["ring_id"].unique()
                    if pd.notna(r) and r}
    metrics["n_rings_found"] = len(rings or [])
    metrics["n_rings_truth"] = len(gt_rings)
    alert_pct = 100 * (scored_txns["txn_risk_score"] >= th.get("high", 65)).mean()
    metrics["alert_rate_pct"] = round(float(alert_pct), 3)
    metrics["within_budget"] = bool(alert_pct <= config.get("alert_budget_pct", 5.0))
    # Phase 0 frozen evaluation inputs (Phase 4 fills calibration in).
    ts = pd.to_datetime(scored_txns["timestamp"])
    frac = float(config.get("model", {}).get("train_window_frac", 0.6))
    cutoff = ts.min() + (ts.max() - ts.min()) * frac
    metrics["train_cutoff"] = str(cutoff)
    metrics["calibration_window"] = None
    metrics["threshold"] = {"high": th.get("high", 65),
                            "critical": th.get("critical", 85)}
    print(f"[evaluate] {metrics}")
    return metrics
