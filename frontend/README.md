# Fraud Intel Frontend (React + TypeScript)

Live-replay UI for the Real-Time Financial Fraud Intelligence backend (`src/serve.py`).

## Prereqs

- Python backend working: `python run_all.py --sample` then `python -m src.serve --port 8000`
- Node 18+ and npm

## Dev (hot reload, backend on :8000)

```bash
cd frontend
npm install
npm run dev      # http://127.0.0.1:5173, /v1 + /lib proxied to 127.0.0.1:8000
```

Vite proxy is configured in `vite.config.ts`, so no CORS setup is needed in dev,
but the backend also sends `Access-Control-Allow-Origin: *` on `/v1/*`.

## Prod (served by the Python backend, still offline-safe)

```bash
cd frontend
npm install
npm run build    # emits frontend/dist/
python -m src.serve --port 8000   # serves dist/ at /, falls back to ui/ if missing
```

`src/serve.py` resolution order for `/`:

1. `frontend/dist/index.html` when it exists (React build)
2. `ui/index.html` legacy fallback (kept for offline use + old tests)

Static `frontend/dist/assets/*` is served with long cache headers.
Legacy routes `/app.js` and `/lib/*` are still served for backwards compatibility.

## Live mode

```bash
python -m src.serve --live --port 8000
```

The React UI adds two controls the legacy `ui/` page lacks:

- **Live ingest** form → `POST /v1/events/ingest` (same schema as
  `transactions.csv`, no score columns)
- **⟳ Rings** button → `POST /v1/rings/refresh`

Without `--live`, both return `400 live mode is off` and the UI shows the error.

## Structure

- `src/api.ts` — typed fetch wrappers for every `/v1/*` endpoint
- `src/types.ts` — backend JSON shapes
- `src/hooks/useFraudIntel.ts` — health/feed/cases/evaluation state + SSE stream
  with 5s polling fallback (parity with legacy `ui/app.js`)
- `src/components/` — `Controls`, `EventFeed`, `TransferGraph` (vis-network),
  `CaseQueue`, `DetailPanel`, `IngestForm`, `EvaluationPanel`
