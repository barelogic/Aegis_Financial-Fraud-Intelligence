"""Read-only replay server + UI backend (Phase 1).

Python stdlib (`http.server`) + pandas only. Serves the batch outputs with
a live-replay API:

  GET /                        -> frontend/dist/index.html (React build) or
                                  ui/index.html legacy fallback
  GET /assets/...              -> frontend/dist/assets/... (Vite build)
  GET /app.js                  -> ui/app.js (legacy, kept for tests/offline)
  GET /lib/...                 -> vendored assets (vis-network, offline-safe)
  GET /v1/health               -> {ok, n_events, cursor, playing, speed}
  GET /v1/events?cursor=&limit -> {events, next_cursor} in time order
  GET /v1/cases                -> cases.json, or rings-derived fallback
  GET /v1/cases/:id
  GET /v1/entities/:id         -> account or transaction detail
  GET /v1/evaluation           -> metrics.json
  GET /v1/events/stream        -> SSE (heartbeat every 15s when idle)
  POST /v1/replay/control      -> {op: start|pause|reset, speed?}

Live mode (`--live`): additionally --
  POST /v1/events/ingest       -> score one raw transaction AS IT ARRIVES
  POST /v1/rings/refresh       -> re-run ring detection over batch + live rows

Replay semantics: strict (timestamp, txn_id) order; all decisions are
precomputed at startup (streaming only reveals a prefix); duplicate txn_id
is a startup error; reset restores cursor 0 so replays are identical;
pause freezes the cursor (in-flight SSE write finishes first).

/v1/* responses carry `Access-Control-Allow-Origin: *` so the React
Vite dev server (http://127.0.0.1:5173) can call the backend directly.

Usage:
    python -m src.serve --outputs data/outputs --port 8000
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import yaml

# Sibling imports must work whether this runs as `python -m src.serve`
# (repo root on sys.path, `src/` is NOT) or as `import serve` with `src/`
# already on the path (tests). So anchor src/ explicitly. No new deps.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from live import LiveRings, LiveScorer, rescore_accounts

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEED = 5


def _clean(value):
    """Make a pandas scalar JSON-safe (NaN/NaT -> None, Timestamp -> str)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _row_dict(row: pd.Series) -> dict:
    return {k: _clean(v) for k, v in row.items()}


def _severity(score: float) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 65:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


class Store:
    """All precomputed decisions, loaded once at startup."""

    def __init__(self, outputs_dir: str | Path, live: bool = False,
                 config: dict | None = None):
        out = Path(outputs_dir)
        txns = pd.read_csv(out / "transactions_scored.csv",
                           parse_dates=["timestamp"])
        if txns["txn_id"].duplicated().any():
            dupes = txns.loc[txns["txn_id"].duplicated(), "txn_id"].unique()
            raise ValueError(f"duplicate txn_id in outputs: {list(dupes)[:5]}")
        txns = txns.sort_values(["timestamp", "txn_id"]).reset_index(drop=True)
        self.events: list[dict] = [_row_dict(r) for _, r in txns.iterrows()]
        self.by_txn: dict[str, dict] = {e["txn_id"]: e for e in self.events}
        accs = pd.read_csv(out / "accounts_scored.csv")
        self.batch_accounts_df = accs
        self.accounts: dict[str, dict] = {
            str(r["account_id"]): _row_dict(r) for _, r in accs.iterrows()}
        self.txns_by_account: dict[str, list[str]] = {}
        for e in self.events:
            self.txns_by_account.setdefault(str(e["account_id"]), []).append(
                e["txn_id"])
        rings_path = out / "rings.json"
        self.rings: list[dict] = (json.loads(rings_path.read_text())
                                  if rings_path.exists() else [])
        cases_path = out / "cases.json"
        self.cases: list[dict] | None = (
            json.loads(cases_path.read_text()) if cases_path.exists() else None)
        metrics_path = out / "metrics.json"
        self.metrics: dict = (json.loads(metrics_path.read_text())
                              if metrics_path.exists() else {})
        self.lock = threading.Lock()
        self.cursor = 0
        self.playing = False
        self.speed = DEFAULT_SPEED
        # -- live state (only with --live) --
        self.live = live
        self.live_rows: list[dict] = []
        self.live_flushed = 0
        self.live_rings: list[dict] | None = None
        self.live_accounts: dict[str, dict] | None = None
        self.scorer: LiveScorer | None = None
        self.rings_engine: LiveRings | None = None
        self.raw_txns_df: pd.DataFrame | None = None
        if live:
            if config is None:
                with open(ROOT / "config.yaml") as f:
                    config = yaml.safe_load(f)
            self.config = config
            raw_dir = out.parent / "raw"
            # Parse datetimes: live rows carry Timestamps, so batch strings
            # must match (mixed str/Timestamp columns crash sorts).
            raw_txns = pd.read_csv(raw_dir / "transactions.csv",
                                   parse_dates=["timestamp"])
            raw_accs = pd.read_csv(raw_dir / "accounts.csv",
                                   parse_dates=["created_at"])
            gt_path = raw_dir / "ground_truth_txn.csv"
            labels = pd.read_csv(gt_path) if gt_path.exists() else None
            print("[serve] live warm-up: replaying history + fitting models ...")
            self.scorer = LiveScorer(config).fit(raw_txns, raw_accs, labels)
            self.rings_engine = LiveRings(config)
            self.raw_txns_df = raw_txns
            self.raw_accs_df = raw_accs
            print("[serve] live ready: send raw txns to POST /v1/events/ingest")

    @property
    def log(self) -> list[dict]:
        """Full ordered log: batch events followed by live arrivals."""
        return self.events + self.live_rows

    @property
    def effective_rings(self) -> list[dict]:
        return self.live_rings if self.live_rings is not None else self.rings

    @property
    def effective_accounts(self) -> dict[str, dict]:
        if self.live_accounts is not None:
            return self.live_accounts
        return self.accounts

    # -- replay state (all under lock) --
    def snapshot(self) -> dict:
        with self.lock:
            return {"cursor": self.cursor, "playing": self.playing,
                    "speed": self.speed}

    def control(self, op: str, speed: int | None = None) -> dict:
        with self.lock:
            if op == "start":
                self.playing = True
            elif op == "pause":
                self.playing = False
            elif op == "reset":
                self.playing = False
                self.cursor = 0
                self.live_flushed = 0
            else:
                raise ValueError(f"unknown op: {op!r}")
            if speed is not None:
                if speed not in (1, 5, 20):
                    raise ValueError("speed must be one of 1, 5, 20")
                self.speed = speed
            return {"cursor": self.cursor, "playing": self.playing,
                    "speed": self.speed}

    def advance(self, n: int) -> list[dict]:
        """Pop up to n UNFLUSHED events from the cursor (SSE tick).

        Live rows revealed via drain_new_live are skipped here so no row
        is ever delivered twice on the stream.
        """
        with self.lock:
            out: list[dict] = []
            base = len(self.events)
            i = self.cursor
            while len(out) < n and i < base + len(self.live_rows):
                if i < base:
                    out.append(self.events[i])
                else:
                    j = i - base
                    if j >= self.live_flushed:
                        out.append(self.live_rows[j])
                        self.live_flushed = j + 1
                i += 1
            self.cursor = i
            return out

    def drain_new_live(self) -> list[dict]:
        """Live rows the cursor has caught up to but not yet delivered."""
        with self.lock:
            base = len(self.events)
            out = []
            while (self.live_flushed < len(self.live_rows)
                   and base + self.live_flushed < self.cursor):
                out.append(self.live_rows[self.live_flushed])
                self.live_flushed += 1
            return out

    # -- live ingestion (only with --live) --
    def ingest_live(self, raw: dict) -> dict:
        """Score one arriving raw transaction and append it to the log."""
        if not self.live or self.scorer is None:
            raise ValueError("live mode is off (restart with --live)")
        event, row = self.scorer.score_one(raw)
        with self.lock:
            self.live_rows.append(event)
            self.by_txn[event["txn_id"]] = event
            self.txns_by_account.setdefault(str(event["account_id"]),
                                            []).append(event["txn_id"])
            assert self.rings_engine is not None
            self.rings_engine.add(row, event)
        return event

    def refresh_rings(self) -> list[dict]:
        """Re-run ring detection over batch + live rows; refresh accounts."""
        if not self.live or self.rings_engine is None:
            raise ValueError("live mode is off (restart with --live)")
        with self.lock:
            batch_scored = pd.DataFrame(self.events)
        assert self.raw_txns_df is not None
        rings = self.rings_engine.refresh(batch_scored, self.raw_txns_df)
        acct_df = self.rings_engine.live_accounts_frame(
            self.scorer.store.acct_info)
        full_accs = pd.concat([self.batch_accounts_df, acct_df],
                              ignore_index=True)
        full_accs = full_accs.drop_duplicates("account_id", keep="first")
        accs = rescore_accounts(batch_scored,
                                list(self.live_rows), full_accs, rings,
                                self.config)
        with self.lock:
            self.live_rings = rings
            self.live_accounts = {
                str(r["account_id"]): _row_dict(r)
                for _, r in accs.iterrows()}
            for aid, tids in self.txns_by_account.items():
                pass  # txns_by_account already tracks live rows on ingest
        return rings

    # -- read views --
    def cases_view(self) -> list[dict]:
        """cases.json when present (Phase 2+), else a rings-derived fallback."""
        if self.cases is not None:
            return self.cases
        out = []
        for r in self.effective_rings:
            members = [str(a) for a in r.get("accounts", [])]
            tids = [t for a in members
                    for t in self.txns_by_account.get(a, [])]
            sev = _severity(float(r.get("ring_risk_score", 0)))
            out.append({
                "case_id": str(r["ring_id"]), "finding_ids": [r["ring_id"]],
                "transaction_ids": sorted(tids),
                "entities": [{"id": a, "roles": ["member"]} for a in members],
                "typologies": r.get("link_types_found", []),
                "evidence": [r.get("evidence", {})],
                "severity": sev, "severity_inputs": {
                    "ring_risk_score": r.get("ring_risk_score")},
                "severity_reason": f"ring risk {r.get('ring_risk_score')} "
                                   f"-> {sev}",
                "timeline": [{"stage": "detected",
                              "facts": r.get("pattern_summary", "")}],
            })
        return out

    def entity(self, entity_id: str) -> dict | None:
        accounts = self.effective_accounts
        if entity_id in accounts:
            acct = dict(accounts[entity_id])
            acct["kind"] = "account"
            acct["transactions"] = self.txns_by_account.get(entity_id, [])
            return acct
        if entity_id in self.by_txn:
            txn = dict(self.by_txn[entity_id])
            txn["kind"] = "transaction"
            return txn
        return None


def make_server(host: str, port: int, store: Store) -> ThreadingHTTPServer:
    """Build (but do not start) the HTTP server. Port 0 picks a free port."""
    ui_dir = ROOT / "ui"
    lib_dir = ROOT / "lib"
    dist_dir = ROOT / "frontend" / "dist"
    dist_index = dist_dir / "index.html"

    class Handler(BaseHTTPRequestHandler):
        server_version = "FraudIntel/1"

        def log_message(self, fmt, *args):  # quieter logs
            print(f"[serve] {self.address_string()} {fmt % args}")

        def _cors(self) -> None:
            # Vite dev server (5173) calls the backend (8000) cross-origin.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods",
                             "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers",
                             "Content-Type")

        def _send_json(self, obj, status: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, ctype: str,
                       max_age: int = 0) -> None:
            try:
                body = path.read_bytes()
            except FileNotFoundError:
                self._send_json({"error": "not found"}, 404)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            if max_age:
                self.send_header("Cache-Control",
                                 f"public, max-age={max_age}")
            self.end_headers()
            self.wfile.write(body)

        def _send_index(self) -> None:
            # Prefer the React build; fall back to the legacy vanilla UI
            # so offline use and old bookmarks keep working when dist/
            # has not been built yet.
            if dist_index.exists():
                self._send_file(dist_index, "text/html")
            else:
                self._send_file(ui_dir / "index.html", "text/html")

        def do_OPTIONS(self):  # noqa: N802 (stdlib naming)
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802 (stdlib naming)
            parsed = urllib.parse.urlparse(self.path)
            parts = parsed.path.strip("/").split("/")
            query = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path in ("/", "/index.html"):
                    self._send_index()
                elif parsed.path == "/legacy":
                    self._send_file(ui_dir / "index.html", "text/html")
                elif parsed.path == "/app.js":
                    self._send_file(ui_dir / "app.js",
                                    "application/javascript")
                elif parts[0] == "assets" and dist_index.exists():
                    safe = (dist_dir / "/".join(parts)).resolve()
                    if not str(safe).startswith(str(dist_dir.resolve())):
                        self._send_json({"error": "forbidden"}, 403)
                        return
                    ctype = ("application/javascript"
                             if safe.suffix == ".js"
                             else "text/css" if safe.suffix == ".css"
                             else "application/octet-stream")
                    self._send_file(safe, ctype, max_age=31536000)
                elif parts[0] == "lib":
                    safe = (lib_dir / "/".join(parts[1:])).resolve()
                    if not str(safe).startswith(str(lib_dir.resolve())):
                        self._send_json({"error": "forbidden"}, 403)
                        return
                    ctype = ("text/css" if safe.suffix == ".css"
                             else "application/javascript"
                             if safe.suffix == ".js" else
                             "application/octet-stream")
                    self._send_file(safe, ctype)
                elif parsed.path == "/v1/health":
                    snap = store.snapshot()
                    self._send_json({"ok": True,
                                     "n_events": len(store.log),
                                     "live": store.live,
                                     "n_live": len(store.live_rows),
                                     **snap})
                elif parsed.path == "/v1/events":
                    cursor = int(query.get("cursor", ["0"])[0])
                    limit = min(int(query.get("limit", ["50"])[0]), 500)
                    log = store.log
                    if cursor < 0 or cursor > len(log):
                        self._send_json({"error": "cursor out of range"}, 400)
                        return
                    batch = log[cursor:cursor + limit]
                    self._send_json({"events": batch,
                                     "next_cursor": cursor + len(batch)})
                elif parsed.path == "/v1/cases":
                    self._send_json({"cases": store.cases_view()})
                elif len(parts) == 3 and parts[:2] == ["v1", "cases"]:
                    match = [c for c in store.cases_view()
                             if c["case_id"] == parts[2]]
                    if not match:
                        self._send_json({"error": "case not found"}, 404)
                    else:
                        self._send_json(match[0])
                elif len(parts) == 3 and parts[:2] == ["v1", "entities"]:
                    ent = store.entity(parts[2])
                    if ent is None:
                        self._send_json({"error": "entity not found"}, 404)
                    else:
                        self._send_json(ent)
                elif parsed.path == "/v1/evaluation":
                    self._send_json(store.metrics)
                elif parsed.path == "/v1/events/stream":
                    self._sse_stream()
                elif dist_index.exists() and "." in parts[-1]:
                    # Any other dotted path (vite.svg, favicon, …) resolves
                    # inside frontend/dist when the React build exists.
                    safe = (dist_dir / parsed.path.lstrip("/")).resolve()
                    if (str(safe).startswith(str(dist_dir.resolve()))
                            and safe.is_file()):
                        ctype = ("application/javascript"
                                 if safe.suffix == ".js"
                                 else "text/css" if safe.suffix == ".css"
                                 else "image/svg+xml"
                                 if safe.suffix == ".svg"
                                 else "text/html" if safe.suffix == ".html"
                                 else "application/octet-stream")
                        self._send_file(safe, ctype)
                        return
                    self._send_json({"error": "not found"}, 404)
                else:
                    self._send_json({"error": "not found"}, 404)
            except (ValueError, KeyError) as e:
                self._send_json({"error": str(e)}, 400)

        def do_POST(self):  # noqa: N802 (stdlib naming)
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/v1/replay/control":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    state = store.control(payload.get("op", ""),
                                          payload.get("speed"))
                    self._send_json({"ok": True, **state})
                except (ValueError, json.JSONDecodeError) as e:
                    self._send_json({"error": str(e)}, 400)
                return
            if parsed.path == "/v1/events/ingest":
                try:
                    length = int(self.headers.get("Content-Length", 0))
                    raw = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(raw, dict):
                        raise ValueError("ingest body must be a JSON object")
                    event = store.ingest_live(raw)
                    self._send_json({"ok": True, "event": event})
                except ValueError as e:
                    msg = str(e)
                    status = 409 if "duplicate" in msg else 400
                    self._send_json({"error": msg}, status)
                except json.JSONDecodeError as e:
                    self._send_json({"error": str(e)}, 400)
                return
            if parsed.path == "/v1/rings/refresh":
                try:
                    rings = store.refresh_rings()
                    self._send_json({"ok": True, "n_rings": len(rings),
                                     "rings": rings})
                except ValueError as e:
                    self._send_json({"error": str(e)}, 400)
                return
            self._send_json({"error": "not found"}, 404)

        def _sse_stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            last_send = time.monotonic()
            try:
                while True:
                    snap = store.snapshot()
                    # Live arrivals the cursor has caught up to go out first.
                    for e in store.drain_new_live():
                        self.wfile.write(
                            f"data: {json.dumps(e)}\n\n".encode())
                        self.wfile.flush()
                        last_send = time.monotonic()
                    if snap["playing"]:
                        batch = store.advance(snap["speed"])
                        for e in batch:
                            self.wfile.write(
                                f"data: {json.dumps(e)}\n\n".encode())
                        if batch:
                            self.wfile.flush()
                            last_send = time.monotonic()
                    if time.monotonic() - last_send >= 15:
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        last_send = time.monotonic()
                    time.sleep(1.0)
            except (BrokenPipeError, ConnectionResetError):
                return

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve fraud-intel outputs")
    parser.add_argument("--outputs", default=str(ROOT / "data" / "outputs"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--live", action="store_true",
                        help="enable live scoring: warm up from data/raw, fit "
                             "models, accept POST /v1/events/ingest")
    args = parser.parse_args()
    store = Store(args.outputs, live=args.live)
    server = make_server(args.host, args.port, store)
    print(f"[serve] {len(store.events)} events, "
          f"{len(store.cases_view())} cases -> "
          f"http://{args.host}:{server.server_address[1]}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
