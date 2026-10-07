"""Tests for src/serve.py against hermetic fixture outputs.

Builds a tiny outputs dir in tmp_path (5 txns, 2 accounts, 1 ring) and
drives a real HTTP server on 127.0.0.1:0 in a daemon thread. Fast, no
dependency on data/outputs/.
"""
import json
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import serve


def _fixture_outputs(path: Path) -> None:
    txns = pd.DataFrame([
        {"txn_id": f"t{i}", "timestamp": f"2025-01-0{d} 10:00:00",
         "account_id": "a0" if i < 3 else "a1", "amount": 1000 + i * 100,
         "currency": "INR", "merchant_id": "m1", "merchant_category": "grocery",
         "item_id": "item_1", "item_category": "grocery", "channel": "POS",
         "device_id": "d0", "ip_address": "10.0.0.1", "city": "Mumbai",
         "country": "India",
         "txn_type": "transfer" if i == 4 else "purchase",
         "dest_account_id": "a0" if i == 4 else "",
         "txn_risk_score": 10.0 * i, "txn_risk_band": "low",
         "txn_reasons": "ok", "txn_action": "allow"}
        for i, d in [(0, 1), (1, 1), (2, 2), (3, 2), (4, 3)]
    ])
    txns.to_csv(path / "transactions_scored.csv", index=False)
    accs = pd.DataFrame([
        {"account_id": "a0", "account_risk_score": 20.0,
         "account_risk_band": "low", "ring_id": "ring_1",
         "account_reasons": "r", "account_action": "monitor"},
        {"account_id": "a1", "account_risk_score": 70.0,
         "account_risk_band": "high", "ring_id": "",
         "account_reasons": "r", "account_action": "restrict"},
    ])
    accs.to_csv(path / "accounts_scored.csv", index=False)
    (path / "rings.json").write_text(json.dumps([{
        "ring_id": "ring_1", "accounts": ["a0"], "size": 1,
        "ring_risk_score": 50.0, "link_types_found": ["shared_device"],
        "evidence": {"shared_devices": ["d0"], "shared_ips": [],
                     "shared_destinations": [],
                     "purchase_sequence_similarity": 0.0,
                     "max_destination_fanin": 0,
                     "account_open_window": "w"},
        "pattern_summary": "p", "recommended_actions": []}]))
    (path / "metrics.json").write_text(json.dumps({"txn_precision": 1.0}))


@pytest.fixture()
def server(tmp_path):
    """Yield (base_url, store) for a live server on a free port."""
    _fixture_outputs(tmp_path)
    store = serve.Store(tmp_path)
    srv = serve.make_server("127.0.0.1", 0, store)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", store
    srv.shutdown()
    srv.server_close()


def _get(base, path, method="GET", payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_health_and_pagination(server):
    base, _ = server
    status, body = _get(base, "/v1/health")
    assert status == 200 and body["ok"] and body["n_events"] == 5
    status, body = _get(base, "/v1/events?cursor=0&limit=2")
    assert [e["txn_id"] for e in body["events"]] == ["t0", "t1"]
    assert body["next_cursor"] == 2
    status, body = _get(base, "/v1/events?cursor=4&limit=50")
    assert [e["txn_id"] for e in body["events"]] == ["t4"]
    status, _ = _get(base, "/v1/events?cursor=99&limit=10")
    assert status == 400


def test_replay_reset_is_deterministic(server):
    base, _ = server
    _get(base, "/v1/replay/control", "POST", {"op": "start", "speed": 20})
    first = _get(base, "/v1/events?cursor=0&limit=50")[1]["events"]
    _get(base, "/v1/replay/control", "POST", {"op": "reset"})
    second = _get(base, "/v1/events?cursor=0&limit=50")[1]["events"]
    assert [e["txn_id"] for e in first] == [e["txn_id"] for e in second]
    status, _ = _get(base, "/v1/replay/control", "POST", {"op": "explode"})
    assert status == 400
    status, _ = _get(base, "/v1/replay/control", "POST", {"speed": 7})
    assert status == 400  # invalid speed rejected


def test_cases_entities_evaluation(server):
    base, _ = server
    _, body = _get(base, "/v1/cases")
    assert body["cases"][0]["case_id"] == "ring_1"
    _, body = _get(base, "/v1/cases/ring_1")
    assert body["severity"] == "MEDIUM"  # ring risk 50
    status, _ = _get(base, "/v1/cases/nope")
    assert status == 404
    status, body = _get(base, "/v1/entities/a0")
    assert status == 200 and body["kind"] == "account"
    assert body["transactions"] == ["t0", "t1", "t2"]
    status, body = _get(base, "/v1/entities/t4")
    assert status == 200 and body["kind"] == "transaction"
    status, _ = _get(base, "/v1/entities/ghost")
    assert status == 404
    status, body = _get(base, "/v1/evaluation")
    assert status == 200 and body["txn_precision"] == 1.0


def test_static_assets_served(server):
    base, _ = server
    for path in ["/", "/app.js", "/lib/vis-9.1.2/vis-network.min.js"]:
        req = urllib.request.Request(base + path)
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200 and len(r.read()) > 100


def test_duplicate_txn_id_rejected(tmp_path):
    _fixture_outputs(tmp_path)
    df = pd.read_csv(tmp_path / "transactions_scored.csv")
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    df.to_csv(tmp_path / "transactions_scored.csv", index=False)
    with pytest.raises(ValueError, match="duplicate txn_id"):
        serve.Store(tmp_path)
