# Real-Time Financial Fraud Intelligence

7-hour hackathon project. Laptop-only, no GPU, no paid APIs, no internet at runtime.

## Quickstart

One command runs everything (deps → sample data → backend → React UI):

```bash
python run.py                 # backend :8000 + frontend :5173, opens browser
python run.py --live          # also enable live scoring (/ingest, /rings/refresh)
python run.py --build         # prod: build dist/, backend serves it, no vite server
python run.py --backend-only  # skip the frontend (needs no Node)
python run.py --help          # --full, --port, --skip-pipeline, --skip-install, ...
```

Manual steps (what `run.py` automates):

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

## Live replay + UI

Read-only server over the batch outputs (stdlib `http.server` + pandas).
The UI is now **React + TypeScript** (`frontend/`, Vite build). The legacy
vanilla page (`ui/`) is kept as an offline fallback:

```bash
.venv/bin/python -m src.serve --outputs data/outputs --port 8000
# open http://127.0.0.1:8000/          -> frontend/dist/ when built, else ui/
# open http://127.0.0.1:8000/legacy    -> legacy vanilla UI always
```

```bash
cd frontend
npm install
npm run dev     # http://127.0.0.1:5173, /v1 proxied to 127.0.0.1:8000
npm run build   # emits frontend/dist/, served by src/serve.py at /
```

API: `GET /v1/health`, `GET /v1/events?cursor=&limit=`,
`GET /v1/events/stream` (SSE, 15s heartbeat), `POST /v1/replay/control`
(`start|pause|reset`, speeds 1/5/20), `GET /v1/cases[/id]`,
`GET /v1/entities/:id`, `GET /v1/evaluation`. Replay is strict
`(timestamp, txn_id)` order with decisions precomputed at startup, so
reset → replay yields identical event IDs every time.

## Real-time scoring (no batch rerun, no past logs)

The batch pipeline replays history; live mode scores transactions **as
they arrive**, with the identical models, causal features, suppression
rule, bands, and reason builders (single source of truth — `src/live.py`
reuses `features_for_row`/`update_state` and the batch scorers):

```bash
.venv/bin/python -m src.serve --live --port 8000   # warms up from data/raw, fits models
# score one arrival (same schema as transactions.csv, no score columns):
curl -X POST localhost:8000/v1/events/ingest -d '{"txn_id":"live_1",
  "timestamp":"2025-02-01 10:00:00", "account_id":"acc_0001", "amount":42000,
  "merchant_category":"electronics", "channel":"online", "device_id":"dev_new",
  "city":"Delhi", "txn_type":"purchase"}'
# -> {"ok": true, "event": {"txn_risk_score": ..., "txn_reasons": ..., ...}}
curl -X POST localhost:8000/v1/rings/refresh   # re-run ring detection incl. live rows
# demo: stream raw history into the live server as if it were happening now:
.venv/bin/python src/simulate_stream.py --post http://127.0.0.1:8000 --max 50 --delay 0.2
```

Live rows append to the same ordered log with a `live: true` flag (badged
LIVE in the UI); the cursor reveals them once caught up, and reset replays
them in arrival order. Ring refresh recomputes over batch + live rows and
refreshes the account view (including sink attribution). Unknown accounts
are registered on first sight; duplicates are rejected (409).

The React UI exposes both live endpoints directly: a **Live ingest** form
(`POST /v1/events/ingest`) and a **⟳ Rings** button
(`POST /v1/rings/refresh`). Both need `--live`; otherwise the server
returns `400 live mode is off`. `/v1/*` sends
`Access-Control-Allow-Origin: *` so `npm run dev` (port 5173) can call the
backend (port 8000) cross-origin.

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
