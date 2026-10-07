"""Real-time stream simulator: replay transactions in time order.

Two modes:
- replay scored rows from data/outputs/transactions_scored.csv (prints
  alerts for high/critical items);
- `--post URL`: POST RAW rows from data/raw/transactions.csv to a live
  server (`python -m src.serve --live`), which scores each AS IT ARRIVES.
  This is the end-to-end real-time demo: no batch rerun, no past logs.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator

import pandas as pd

RAW_COLUMNS = ["txn_id", "timestamp", "account_id", "amount", "currency",
               "merchant_id", "merchant_category", "item_id", "item_category",
               "channel", "device_id", "ip_address", "city", "country",
               "txn_type", "dest_account_id"]


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


def post_live(raw_path: str, server: str, delay_s: float = 0.1,
              max_rows: int = 0) -> int:
    """POST raw transactions to a live server's ingest endpoint.

    Returns the number of accepted rows. Prints each live score as it
    comes back. stdlib only (urllib).
    """
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    df = df.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
    if max_rows > 0:
        df = df.head(max_rows)
    n = 0
    for _, row in df.iterrows():
        payload = {c: (None if pd.isna(row[c]) else row[c])
                   for c in RAW_COLUMNS if c in df.columns}
        payload["timestamp"] = str(row["timestamp"])
        req = urllib.request.Request(
            server.rstrip("/") + "/v1/events/ingest",
            data=json.dumps(payload, default=str).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                event = json.loads(r.read())["event"]
        except urllib.error.HTTPError as e:
            print(f"SKIP {payload['txn_id']}: HTTP {e.code} {e.read()[:120]}")
            continue
        print(f"LIVE {event['txn_id']} score={event['txn_risk_score']} "
              f"band={event['txn_risk_band']} action={event['txn_action']}")
        n += 1
        if delay_s > 0:
            time.sleep(delay_s)
    print(f"posted {n} transactions to {server}")
    return n


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Replay or live-post transactions")
    p.add_argument("path", nargs="?",
                   default="data/outputs/transactions_scored.csv")
    p.add_argument("--post", default=None, metavar="URL",
                   help="live server base URL, e.g. http://127.0.0.1:8000 "
                        "(reads RAW data/raw/transactions.csv instead)")
    p.add_argument("--delay", type=float, default=0.0)
    p.add_argument("--max", type=int, default=0,
                   help="max rows to post (0 = all)")
    args = p.parse_args()
    if args.post:
        raw = "data/raw/transactions.csv"
        post_live(raw, args.post, delay_s=args.delay, max_rows=args.max)
    else:
        n = sum(1 for _ in simulate_stream(args.path, delay_s=args.delay))
        print(f"replayed {n} transactions")
