"""Provenance manifest (Phase 0): binds outputs to exact inputs + config.

`collect_manifest` records seed, config hash, per-file sha256/size/row
counts, data time range, label counts, and git sha. `check_manifest`
refuses to score when `data/raw/*.csv` drifted from the manifest unless
freshly generated or `--force` is passed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

RAW_FILES = ["transactions.csv", "accounts.csv",
             "ground_truth_txn.csv", "ground_truth_accounts.csv"]


def sha256_file(path: str | Path) -> str:
    """SHA-256 hex digest of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(config: dict) -> str:
    """Stable hash of the config dict (key order independent)."""
    blob = yaml.safe_dump(config, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def git_sha(cwd: str | Path) -> str:
    """Current git HEAD sha, or 'unknown' outside a repo."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def collect_manifest(raw_dir: str | Path, config: dict) -> dict:
    """Build the manifest dict for the current raw files + config."""
    raw = Path(raw_dir)
    files: dict[str, dict] = {}
    for name in RAW_FILES:
        p = raw / name
        with open(p, "rb") as f:
            n_rows = sum(1 for _ in f) - 1  # minus header
        files[name] = {"sha256": sha256_file(p),
                       "bytes": p.stat().st_size, "rows": max(n_rows, 0)}
    txns = pd.read_csv(raw / "transactions.csv", usecols=["timestamp"])
    ts = pd.to_datetime(txns["timestamp"])
    gtt = pd.read_csv(raw / "ground_truth_txn.csv", usecols=["is_fraud"])
    gta = pd.read_csv(raw / "ground_truth_accounts.csv",
                      usecols=["is_fraud_account"])
    return {
        "seed": int(config.get("seed", 42)),
        "config_sha256": config_hash(config),
        "files": files,
        "time_range": {"min": str(ts.min()), "max": str(ts.max())},
        "label_counts": {"fraud_txns": int((gtt["is_fraud"] == 1).sum()),
                         "fraud_accounts":
                             int((gta["is_fraud_account"] == 1).sum())},
        "git_sha": git_sha(raw.parent),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(raw_dir: str | Path, out_path: str | Path,
                   config: dict) -> dict:
    """Collect + write manifest.json. Returns the manifest dict."""
    manifest = collect_manifest(raw_dir, config)
    Path(out_path).write_text(json.dumps(manifest, indent=2))
    return manifest


def check_manifest(raw_dir: str | Path, manifest_path: str | Path,
                   config: dict, force: bool = False,
                   just_generated: bool = False) -> tuple[bool, str]:
    """Gate scoring on raw-input provenance.

    Returns (ok, message). Refuses (ok=False) when a manifest exists and
    raw hashes differ, unless `force` is set. Freshly generated (or first
    run) input always (over)writes the manifest and passes.
    """
    manifest_path = Path(manifest_path)
    if just_generated or not manifest_path.exists():
        manifest = write_manifest(raw_dir, manifest_path, config)
        return True, (f"[provenance] wrote {manifest_path} "
                      f"(git {manifest['git_sha'][:8]})")
    old = json.loads(manifest_path.read_text())
    current = collect_manifest(raw_dir, config)
    changed = [n for n in RAW_FILES
               if old.get("files", {}).get(n, {}).get("sha256")
               != current["files"][n]["sha256"]]
    if changed and not force:
        return False, (f"[provenance] REFUSED: data/raw changed since "
                       f"manifest ({', '.join(changed)}). Re-run generation "
                       f"or pass --force to accept new inputs.")
    write_manifest(raw_dir, manifest_path, config)
    extra = f" (overrode drift in {', '.join(changed)})" if changed else ""
    return True, f"[provenance] manifest ok{extra}."
