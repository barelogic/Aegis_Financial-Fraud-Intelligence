# Real-Time Financial Fraud Intelligence

Batch + real-time fraud detection over synthetic transaction streams, with
explainable risk scores (0–100), account-level rollups, mule-ring graph
detection, a replayable live API, and a React risk-ops dashboard.

Built as a 7-hour hackathon project. No GPU, no paid APIs,
no internet required at runtime (frontend deps are vendored via
`frontend/package-lock.json`; Python deps install from `requirements.txt`).

Repository: https://github.com/barelogic/Financial-fraud-Intelligent-Checker
— public, contains the complete source code. Clone it and follow
**Setup** + **How to run** below; no other steps or private submodules
are needed.

## What it does

- **Synthetic data generator** (`src/generate_data.py`): accounts,
  transactions, and ground-truth labels, including an intentionally
  stealthy fraud ring (Ring A) whose single transactions look ordinary.
- **Causal feature engine** (`src/features.py`): behavior-relative
  features computed strictly from data *before* each transaction's
  timestamp (no look-ahead).
- **Transaction scoring** (`src/score_transactions.py`):
  IsolationForest + XGBoost blend (0–100) with a suppression rule for
  legit high-value spend and SHAP/rule plain-English reasons.
- **Account scoring** (`src/score_accounts.py`): blend of max txn score
  + top-3 mean, plus a ring bump.
- **Ring detection** (`src/detect_rings.py`): shared-device/IP/
  destination/merchant graph + community detection; ring risk is pushed
  back onto member accounts/transactions.
- **Actions + report + evaluation** (`src/actions.py`, `src/report.py`,
  `src/evaluate.py`): recommended action per flagged item,
  `data/outputs/report.md`, `data/outputs/metrics.json`.
- **Replay server** (`src/serve.py`, stdlib `http.server` + pandas):
  deterministic `(timestamp, txn_id)` replay with SSE streaming,
  case/entity/evaluation APIs.
- **Live scoring** (`src/live.py`): scores arrivals as they come in with
  the identical models, causal features, suppression rule, bands, and
  reason builders — no batch rerun.
- **React + TypeScript UI** (`frontend/`, Vite): event feed, transfer
  graph, case queue, detail panel, evaluation panel, live-ingest form,
  ring-refresh button. Legacy vanilla page (`ui/`) kept as offline
  fallback.

## Prerequisites

- Python 3.10+ (tested on 3.10–3.14, CPU only)
- Node 18+ and npm (only for the React UI; backend works without it)
- `pip`, `venv` (standard library), `git`
- ~500 MB free for `.venv/` + `frontend/node_modules/` + generated CSVs

## Setup

```bash
git clone https://github.com/barelogic/Financial-fraud-Intelligent-Checker.git
cd Financial-fraud-Intelligent-Checker

# 1. Python environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Frontend deps (skip if you only want backend + legacy ui/)
cd frontend && npm ci && cd ..

# 3. Generate + score data (fast sample: 2,000 rows)
python run_all.py --sample
# full run:
# python run_all.py
```

What `run_all.py` accepts:

```bash
python run_all.py --sample        # fast test on 2,000 rows
python run_all.py                 # full run
python run_all.py --skip-generate # reuse existing data/raw
python run_all.py --seed 123
python run_all.py --force         # accept data/raw drift vs manifest.json
```

Sample `data/raw/*.csv` is committed so evaluators can run without
regenerating. Fresh runs write `data/outputs/` (`transactions_scored.csv`,
`accounts_scored.csv`, `rings.json`, `metrics.json`, `report.md`, `manifest.json`).

## How to run the solution

One command runs everything (deps → sample data → backend → React UI):

```bash
python run.py                 # backend :8000 + frontend :5173, opens browser
python run.py --live          # also enable live scoring (/ingest, /rings/refresh)
python run.py --build         # prod: build dist/, backend serves it, no vite server
python run.py --backend-only  # skip the frontend (needs no Node)
python run.py --help          # --full, --port, --skip-pipeline, --skip-install, ...
```

Then open:

- React UI: http://127.0.0.1:5173/ (dev) or http://127.0.0.1:8000/ (`--build`)
- Backend health: http://127.0.0.1:8000/v1/health
- Legacy fallback UI: http://127.0.0.1:8000/legacy

Manual equivalent (what `run.py` automates):

```bash
# backend over batch outputs
.venv/bin/python -m src.serve --outputs data/outputs --port 8000
# open http://127.0.0.1:8000/          -> frontend/dist/ when built, else ui/
# open http://127.0.0.1:8000/legacy    -> legacy vanilla UI always

# frontend dev server
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

## Tests

```bash
.venv/bin/python -m pytest -q
```

Covers generation, temporal causality (no look-ahead), score bands,
live/batch parity (`tests/test_live.py`), the HTTP API
(`tests/test_serve.py`), and an end-to-end smoke run
(`tests/test_smoke.py`).

## Provenance

Every run writes `data/outputs/manifest.json`: seed, config hash, per-file
sha256/size/row counts for `data/raw/*.csv`, data time range, label counts,
and git sha. If raw inputs drift from the manifest (e.g. hand-edited CSVs),
scoring refuses with exit 2 unless the data was freshly generated or
`--force` is passed. `metrics.json` also freezes its evaluation inputs
(`train_cutoff`, `calibration_window`, `threshold`).

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

## Project layout

```text
run.py                  one-command runner (backend + frontend)
run_all.py              batch pipeline driver
config.yaml             thresholds, blend weights, column mapping
requirements.txt        Python deps
src/                    generate_data, features, scoring, rings, live, serve
frontend/               React + TypeScript UI (Vite)
  src/api.ts, types.ts, hooks/useFraudIntel.ts, components/, App.tsx
ui/                     legacy vanilla fallback UI (offline)
lib/                    vendored vis-network / tom-select for offline ui/
synthgen/               synthetic-data helper notes
tests/                  pytest suite
data/raw/               committed sample input CSVs
data/outputs/           generated on run (gitignored except .gitkeep)
```
