"""Pipeline entry point: runs all stages in order and prints a summary table.

Order: generate_data -> features -> score_transactions -> score_accounts
       -> detect_rings (+ ring bump back onto accounts/transactions)
       -> actions -> report -> evaluate

Usage:
    python run_all.py [--seed 42] [--skip-generate] [--sample]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "outputs"


def load_config(seed_override: int | None = None) -> dict:
    """Load config.yaml, optionally overriding the seed."""
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    if seed_override is not None:
        cfg["seed"] = seed_override
    return cfg


def maybe_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Uniform random sample of n rows (by time order restored after)."""
    if len(df) <= n:
        return df
    return (
        df.sample(n=n, random_state=seed)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def print_summary(cfg: dict) -> None:
    """Print a short summary table from pipeline outputs."""
    txn_path = OUT / "transactions_scored.csv"
    acc_path = OUT / "accounts_scored.csv"
    rings_path = OUT / "rings.json"
    metrics_path = OUT / "metrics.json"
    print("\n===== SUMMARY =====")
    if txn_path.exists():
        t = pd.read_csv(txn_path)
        print(f"transactions scored: {len(t)}")
        print(t["txn_risk_band"].value_counts().to_string())
        n_flag = (t["txn_risk_score"] >= cfg["risk_thresholds"]["high"]).sum()
        print(f"flagged high/critical txns: {n_flag} ({100*n_flag/max(len(t),1):.2f}%)")
        print(f"suppression helps precision: legit high-value txns stay below "
              f"{cfg['risk_thresholds']['medium']}.")
    if acc_path.exists():
        a = pd.read_csv(acc_path)
        print(f"\naccounts scored: {len(a)}")
        print(a["account_risk_band"].value_counts().to_string())
    if rings_path.exists():
        rings = json.loads(rings_path.read_text())
        print(f"\nrings found: {len(rings)}")
        for r in rings[:10]:
            print(f"  {r['ring_id']}: size={r['size']} "
                  f"risk={r['ring_risk_score']} pattern={r['pattern_summary'][:80]}")
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        print(f"\nmetrics: {json.dumps(m, indent=2)[:800]}")
    print("===================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud-intel pipeline")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--sample", action="store_true",
                        help="run on 2000 rows for fast testing")
    parser.add_argument("--force", action="store_true",
                        help="accept data/raw drift vs manifest.json")
    args = parser.parse_args()

    cfg = load_config(args.seed)
    seed: int = int(cfg["seed"])
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. generate_data
    if not args.skip_generate:
        from generate_data import generate_all
        print("[1/8] generate_data ...")
        generate_all(str(RAW), seed=seed, config=cfg)
    else:
        print("[1/8] generate_data skipped.")

    # 2. features
    from features import compute_features, load_inputs
    print("[2/8] features ...")
    txns, accounts = load_inputs(str(RAW / "transactions.csv"),
                                 str(RAW / "accounts.csv"), cfg)
    if args.sample:
        txns = maybe_sample(txns, int(cfg.get("sample_rows", 2000)), seed)
    feats = compute_features(txns, accounts, cfg)

    # Phase 0 provenance gate: refuse to score drifted inputs.
    from provenance import check_manifest
    ok, msg = check_manifest(str(RAW), OUT / "manifest.json", cfg,
                             force=args.force,
                             just_generated=not args.skip_generate)
    print(msg)
    if not ok:
        sys.exit(2)

    # 3. score_transactions (includes SHAP + rule explanations)
    from score_transactions import score_transactions
    print("[3/8] score_transactions ...")
    gt_path = RAW / "ground_truth_txn.csv"
    labels = pd.read_csv(gt_path) if gt_path.exists() else None
    if labels is not None and args.sample:
        labels = labels[labels["txn_id"].isin(set(txns["txn_id"]))]
    scored_txns = score_transactions(feats, txns, labels, cfg)
    scored_txns.to_csv(OUT / "transactions_scored.csv", index=False)

    # 4. score_accounts (base scores; ring bump applied after detection)
    from score_accounts import score_accounts, apply_ring_bump
    from score_transactions import apply_ring_bump_to_transactions
    print("[4/8] score_accounts ...")
    scored_accs = score_accounts(scored_txns, feats, accounts, None, cfg)
    scored_accs.to_csv(OUT / "accounts_scored.csv", index=False)

    # 5. detect_rings, then push ring risk back onto members
    from detect_rings import detect_rings, save_ring_outputs
    print("[5/8] detect_rings ...")
    rings = detect_rings(scored_txns, feats, txns, cfg)
    save_ring_outputs(rings, scored_txns, txns, str(OUT))
    if rings:
        scored_accs = apply_ring_bump(scored_accs, rings, cfg, scored_txns)
        scored_accs.to_csv(OUT / "accounts_scored.csv", index=False)
        scored_txns = apply_ring_bump_to_transactions(scored_txns, rings, cfg)
        scored_txns.to_csv(OUT / "transactions_scored.csv", index=False)

    # 6. actions
    from actions import apply_actions
    print("[6/8] actions ...")
    scored_txns = apply_actions(scored_txns, scored_accs, rings, cfg,
                                level="transaction")
    scored_accs = apply_actions(scored_txns, scored_accs, rings, cfg,
                                level="account")
    scored_txns.to_csv(OUT / "transactions_scored.csv", index=False)
    scored_accs.to_csv(OUT / "accounts_scored.csv", index=False)

    # 7. report
    from report import build_report
    print("[7/8] report ...")
    build_report(scored_txns, scored_accs, rings, cfg, str(OUT))

    # 8. evaluate
    from evaluate import evaluate
    print("[8/8] evaluate ...")
    try:
        gt_acc = pd.read_csv(RAW / "ground_truth_accounts.csv")
    except FileNotFoundError:
        gt_acc = None
    try:
        gt_txn = pd.read_csv(RAW / "ground_truth_txn.csv")
    except FileNotFoundError:
        gt_txn = None
    metrics = evaluate(scored_txns, scored_accs, rings, gt_txn, gt_acc, cfg)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print_summary(cfg)


if __name__ == "__main__":
    main()
