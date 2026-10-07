# Improvements — port the good parts of Talon into this repo

> Agent instructions: implement the phases below in order. Work in this repo only.
> Reference design: https://github.com/praneettigga/Talon (do NOT add it as a
> dependency, do NOT copy its IBM/Kaggle data pipeline, do NOT add Node/npm,
> torch, or GPU code).

## Context

This repo (`run_all.py` + `src/`) is a Python-only batch pipeline:
`generate_data -> features -> score_transactions -> score_accounts ->
detect_rings -> actions -> report -> evaluate`, outputs in `data/outputs/`.
Constraints: laptop-only, no GPU, no paid APIs, no internet at runtime.
Keep it that way. Python stdlib + current `requirements.txt` only
(pandas/numpy/sklearn/xgboost/shap/networkx/python-louvain/pyvis/matplotlib).
Vendored UI assets already exist under `lib/vis-9.1.2/`, `lib/tom-select/`.

What to steal from Talon (adapted, not copied):
live replay + UI, deterministic ordering + immutable decisions, structured
cases with evidence/timeline/severity, explicit typology rules, calibrated
threshold + review gate, time-decayed entity risk, what-if intervention
simulator, provenance manifest, separate txn vs case metrics, reproducibility tests.

## Phase 0 — Provenance + frozen config (do first, ~1h)

1. Add `src/provenance.py`: `write_manifest(raw_dir, out_path, config)` recording
   `{seed, config hash, file sha256/sizes/row counts, time range, label counts,
   git sha}` to `data/outputs/manifest.json`. Add `--manifest` check in
   `run_all.py`: refuse to score if `data/raw/*.csv` hashes differ from manifest
   unless `--skip-generate` is passed with fresh generation.
2. Freeze evaluation inputs: record `train_cutoff, calibration_window, threshold`
   in `metrics.json` (see Phase 4). Never tune on test after freezing.
3. Acceptance: `python run_all.py --sample` twice byte-identical outputs
   (sort by `(timestamp, txn_id)` everywhere); `manifest.json` present.

## Phase 1 — Read-only UI + replay API (biggest win)

No Node/React. Pure Python stdlib `http.server` + vendored `lib/vis-9.1.2/vis-network.min.js`.

1. New `src/serve.py` (~250 lines): loads `data/outputs/*.csv + rings.json/cases.json`,
   serves:
   - `GET /` → static `ui/index.html` (new `ui/` dir, vanilla JS + vis-network).
   - `GET /v1/health`, `GET /v1/events?cursor=&limit=` (time-ordered),
     `GET /v1/cases`, `GET /v1/cases/:id`, `GET /v1/entities/:id`,
     `GET /v1/evaluation` (serves `metrics.json`).
   - `GET /v1/events/stream` (SSE, heartbeat 15s). State lives in server memory;
     `POST /v1/replay/control` with `start/pause/reset/speed`.
2. Replay semantics (copy Talon's): strict `(timestamp, txn_id)` order, duplicate-ID
   error, pause waits for in-flight event, reset clears + recreates identical
   sequence. Precompute decisions once at startup; streaming only reveals prefix.
3. UI (`ui/index.html + ui/app.js`, no build step): feed (latest 50), graph
   (latest 40 transfers, toggle all), case queue, click node/edge → detail panel,
   pause/reset/speed 1-5-20. Reference Talon's `frontend/src/Graph.tsx` layout
   but implement in ~300 lines vanilla JS.
4. `python -m src.serve --outputs data/outputs --port 8000` documents in README.
5. Acceptance: start server, `curl /v1/health`, reset → replay twice → identical
   event IDs; UI renders with no CDN/internet (all assets local under `lib/`).

## Phase 2 — Cases (upgrade rings.json, don't break it)

Talon's `backend/python/engine.py:correlate` merges findings into cases; do same.

1. New `src/cases.py`: `build_cases(rings, scored_txns, feats) -> cases.json` list of
   `{case_id, finding_ids, transaction_ids, entities[{id, roles}], typologies,
   evidence[{ring_id, link_types, shared_devices/ips/destinations, window}],
   severity, severity_inputs, severity_reason, timeline[{timestamp, stage, facts}]}`.
2. Merge rule v1 (simple): two rings sharing ≥1 transaction OR ≥1 anchor
   (shared device/ip/destination) within 7 days merge. Stable `case_XXXX` IDs,
   merged IDs kept as `aliases`. Keep writing legacy `rings.json` unchanged.
3. Severity v1: `LOW` default; `MEDIUM` if `max_member_score >= high_threshold`;
   `HIGH` if also `>=80 and entities>=4`; `CRITICAL` if `>=95 and entities>=6`.
   (Talon additionally requires model corroboration — wire that in Phase 4.)
4. Acceptance: `cases.json` validates (1:1 txn overlap matching in tests);
   `report.py` gains a Cases section; old consumers of `rings.json` unaffected.

## Phase 3 — Explicit typology detectors (alongside Louvain)

Talon `engine.py:detect` has FAN-IN/FAN-OUT/CYCLE/STACK/etc. with 7-day window.
Port a minimal pure-pandas version into `src/typologies.py`:

- `FAN-IN/FAN-OUT`: ≥2 distinct senders/receivers on one hub in 7d.
- `CYCLE`: time-ordered 2-6 hop path returning to start (bounded 10k expansions).
- `STACK`: 3 time-ordered links through 4 distinct accounts.
Self-transfers excluded. Each returns `{typology, anchors, transaction_ids,
strength, facts[]}`. Feed these as extra `link_types_found` into `detect_rings.py`
and as `evidence` into `cases.py`. Do NOT replace Louvain yet.
Acceptance: unit tests on synthetic 6-row fixtures per typology + no new rings
on `hard_neg_family_device` / `hard_neg_shared_ip` fixtures.

## Phase 4 — Calibration + review gate + decayed entity risk

Kills the current FP problem (suppression rule alone is ad-hoc).

1. In `score_transactions.py`: split calibration window (e.g. days 20-24 fit
   Platt logistic on OOF margins, days 24-27 pick threshold at ≤1% FPR), freeze
   threshold in `metrics.json`. Rows before train cutoff get `risk=None,
   status=historical warmup` instead of a fake score.
2. Review gate: `HIGH+` requires `(typology_match OR dest_fanin>=min_ring_fanin)
   AND fused_risk>=threshold`. Report both raw-threshold and gated P/R/FPR.
3. In `score_accounts.py`: replace static blend with 7-day time-decayed max
   (`score * exp(-dt/24h)`), store `riskAsOf + sourceTxnId`. Keep old columns.
4. Acceptance: `metrics.json` contains `{threshold, calibration{...}, gated vs
   ungated P/R/FPR, PR-AUC, Brier}`; warmup rows have null scores; gate reduces
   FPR on `--sample` run without collapsing recall.

## Phase 5 — What-if intervention simulator (pure Python)

Port of Talon's `interventions.py` minus time-ordering shortcuts.

1. New `src/interventions.py`: `compare_hold(cases, case_id, held[], compare_held[])`
   on a copy of the case graph: delete outgoing edges of held accounts, return
   `{interrupted_links, unreachable_downstream, alternate_witnesses (≤50),
   touched_accounts}`. No state mutation, no "money saved" claim. Label output
   `observed-route disruption; assumes similar routes recur`.
2. Expose `POST /v1/interventions/simulate` in `serve.py` + minimal UI panel
   (checkbox accounts → Compare button → red highlight).
3. Acceptance: pure-function tests (cycle graph: holding midpoint disconnects
   sink; holding non-member → 400); replay state unchanged after simulate.

## Phase 6 — Evaluation: txn vs case metrics

Extend `src/evaluate.py` (keep old keys): add `gated_{precision,recall,fpr}`,
`pr_auc, brier`, and `case_{precision,recall,mean_jaccard}` via max-cardinality
1:1 assignment on txn-overlap (any nonempty overlap may match; Jaccard =
   |∩|/|∪| averaged over matches). Exclude boundary-truncated attempts.
   Serve via `/v1/evaluation`. Acceptance: metrics recompute deterministically;
   README documents that case metrics include LOW cases and differ from gated txn metrics.

## Phase 7 — Tests + docs

Mirror Talon's `backend/python/tests/` + `backend/tests/` discipline:

- `tests/test_temporal.py` (extend): strict ordering, duplicate-ID rejection,
  prior-only features (no future leakage), reset reproducibility.
- `tests/test_cases.py` (new): merge/expiry fixtures, severity gates.
- `tests/test_interventions.py` (new): simulator fixtures.
- Update `README.md` (serve + cases + metrics) and `synthgen/README.md` if schemas change.
- Full `pytest tests/ -q` green.

## Non-goals (do NOT do)

- No npm/Node/TypeScript, no torch/GIN, no Kaggle/IBM download, no `.env`/credentials,
  no DB, no "portable model interchange" (joblib artifacts stay local), no internet
  at runtime, no claiming risk scores are fraud probabilities.

## Order + checkpoints

Land Phase 0+1 first (reviewable UI), then 2+3, then 4+6, then 5+7.
Each phase: code + tests + README line + `python run_all.py --sample` green.
