"""Real-time stream simulator: replay scored transactions in time order.

Baseline: reads data/outputs/transactions_scored.csv and yields rows one by
one (optionally with a delay), printing alerts for high/critical items.
Teammates can wire this to a live source.
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import pandas as pd


def simulate_stream(scored_path: str, delay_s: float = 0.0,
                    alert_threshold: float = 65.0) -> Iterator[dict]:
    """Yield scored rows in timestamp order; print alerts >= threshold."""
    df = pd.read_csv(scored_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    for _, row in df.iterrows():
        rec = row.to_dict()
        if float(rec.get("txn_risk_score", 0)) >= alert_threshold:
            print(f"ALERT {rec['txn_id']} score={rec['txn_risk_score']} "
                  f"action={rec.get('txn_action')}: {rec.get('txn_reasons')}")
        if delay_s > 0:
            time.sleep(delay_s)
        yield rec


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/outputs/transactions_scored.csv"
    n = sum(1 for _ in simulate_stream(path))
    print(f"replayed {n} transactions")
