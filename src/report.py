"""Markdown report builder (baseline)."""
from __future__ import annotations

import pandas as pd


def build_report(scored_txns: pd.DataFrame, scored_accs: pd.DataFrame,
                 rings: list[dict] | None, config: dict,
                 out_dir: str) -> str:
    """Write data/outputs/report.md and return its path."""
    th = config.get("risk_thresholds", {})
    lines = ["# Fraud Intelligence Report", ""]
    lines.append(f"## Transactions ({len(scored_txns)})")
    lines.append(scored_txns["txn_risk_band"].value_counts().to_string())
    n_flag = (scored_txns["txn_risk_score"] >= th.get("high", 65)).sum()
    lines.append(f"\nFlagged high/critical: {n_flag} "
                f"(budget {config.get('alert_budget_pct', 5.0)}%)")
    lines.append(f"\n## Accounts ({len(scored_accs)})")
    lines.append(scored_accs["account_risk_band"].value_counts().to_string())
    lines.append(f"\n## Rings ({len(rings or [])})")
    for r in rings or []:
        lines.append(f"\n### {r['ring_id']} (risk {r['ring_risk_score']})")
        lines.append(f"Pattern: {r['pattern_summary']}")
        lines.append(f"Accounts: {', '.join(r['accounts'][:20])}")
    lines.append("\n## Top-10 riskiest transactions")
    top = scored_txns.nlargest(10, "txn_risk_score")
    for _, row in top.iterrows():
        lines.append(f"- {row['txn_id']} ({row['account_id']}) "
                     f"score={row['txn_risk_score']}: {row['txn_reasons']}")
    path = f"{out_dir}/report.md"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[report] wrote {path}")
    return path
