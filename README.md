# Real-Time Financial Fraud Intelligence

7-hour hackathon project. Laptop-only, no GPU, no paid APIs, no internet at runtime.

## Quickstart

```bash
pip install -r requirements.txt
python run_all.py --sample        # fast test on 2,000 rows
python run_all.py                 # full run
python run_all.py --skip-generate # reuse existing data/raw
python run_all.py --seed 123
python run_all.py --force         # accept data/raw drift vs manifest.json
```

## Provenance (Phase 0)

Every run writes `data/outputs/manifest.json`: seed, config hash, per-file
sha256/size/row counts for `data/raw/*.csv`, data time range, label counts,
and git sha. If raw inputs drift from the manifest (e.g. hand-edited CSVs),
scoring refuses with exit 2 unless the data was freshly generated or
`--force` is passed. `metrics.json` also freezes its evaluation inputs
(`train_cutoff`, `calibration_window`, `threshold`).

## Live replay + UI (Phase 1)

Read-only server over the batch outputs (stdlib `http.server` + pandas,
no Node, works offline via vendored `lib/`):

```bash
.venv/bin/python -m src.serve --outputs data/outputs --port 8000
# open http://127.0.0.1:8000/
```

API: `GET /v1/health`, `GET /v1/events?cursor=&limit=`,
`GET /v1/events/stream` (SSE, 15s heartbeat), `POST /v1/replay/control`
(`start|pause|reset`, speeds 1/5/20), `GET /v1/cases[/id]`,
`GET /v1/entities/:id`, `GET /v1/evaluation`. Replay is strict
`(timestamp, txn_id)` order with decisions precomputed at startup, so
reset → replay yields identical event IDs every time.

## Pipeline

`run_all.py` runs, in order:

1. `generate_data` — synthetic transactions/accounts + ground truth
2. `features` — behavior-relative, strictly causal features (only data
   before each transaction's timestamp)
3. `score_transactions` — IsolationForest + XGBoost blend (0-100) with
   suppression rule + SHAP/rule explanations
4. `score_accounts` — blend of max txn score + top-3 mean (+ ring bump later)
5. `detect_rings` — shared-device/IP/destination graph + community detection;
   ring risk is pushed back onto member accounts/transactions
6. `actions` — recommended action per flagged item
7. `report` — `data/outputs/report.md`
8. `evaluate` — `data/outputs/metrics.json`

## Key design notes

- Ring A transactions look ordinary individually on purpose; transaction
  scores alone will NOT catch ring A. Ring detection pushes risk back down
  onto member accounts/transactions afterwards.
- Suppression rule protects legit high-value transactions: high amount but
  normal z-score + known device/city + no destination fan-in caps the score
  below the medium threshold (every firing is logged).
- Every score ships with plain-English reasons (SHAP top-3 + rules).
  Low-risk rows get a reason too (e.g. "Consistent with ... normal spending").
- `config.yaml:column_mapping` maps logical schema names to dataset columns
  so a different dataset can be plugged in without code changes.

## Layout

See repo tree. Outputs land in `data/outputs/` with the fixed schemas.
