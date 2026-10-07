"""Ring detection: accounts acting suspiciously TOGETHER.

Builds an account graph with weighted link types (shared device / IP /
destination / merchant, purchase-sequence similarity), runs community
detection, and returns ring dicts in the rings.json schema. Also writes
per-ring HTML (pyvis) and PNG (matplotlib) diagrams.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd


def _seq_signature(transactions: pd.DataFrame, acct: str) -> tuple:
    """Ordered merchant-category sequence for one account (baseline)."""
    sub = transactions[transactions["account_id"] == acct].sort_values("timestamp")
    return tuple(sub["merchant_category"].tolist()[:8])


def build_link_graph(scored_txns: pd.DataFrame, transactions: pd.DataFrame,
                     config: dict):
    """Build a weighted account graph from shared-entity links."""
    import networkx as nx
    w = config.get("link_weights", {})
    G = nx.Graph()
    for a in transactions["account_id"].unique():
        G.add_node(a)

    def _add_link(col: str, weight: float, min_share: int = 2) -> None:
        groups = transactions.groupby(col)["account_id"].apply(set)
        for _, users in groups.items():
            users = {u for u in users if u == u and u != ""}
            if len(users) < min_share:
                continue
            for u, v in itertools.combinations(sorted(users), 2):
                prev = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
                G.add_edge(u, v, weight=prev + weight)

    # Families legitimately share devices, so device links alone are weak.
    _add_link("device_id", float(w.get("shared_device", 1.0)), min_share=2)
    _add_link("ip_address", float(w.get("shared_ip", 1.0)), min_share=2)
    _add_link("dest_account_id", float(w.get("shared_destination", 1.5)),
              min_share=2)

    # Purchase-sequence similarity: same ordered category pattern.
    seq_to_accts: dict[tuple, set] = {}
    for a in transactions["account_id"].unique():
        seq_to_accts.setdefault(_seq_signature(transactions, a), set()).add(a)
    for seq, users in seq_to_accts.items():
        if len(seq) >= 3 and len(users) >= 3:
            for u, v in itertools.combinations(sorted(users), 2):
                prev = G[u][v]["weight"] if G.has_edge(u, v) else 0.0
                G.add_edge(u, v,
                           weight=prev + float(w.get("purchase_sequence", 0.8)))
    # Drop trivial single-weak-link edges to limit false rings.
    weak = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 1.5]
    G.remove_edges_from(weak)
    return G


def find_communities(G, seed: int) -> list[set]:
    """Community detection: Louvain if available, else greedy modularity."""
    import networkx as nx
    if G.number_of_edges() == 0:
        return []
    try:
        import community as community_louvain  # python-louvain
        part = community_louvain.best_partition(G, weight="weight",
                                                random_state=seed)
        groups: dict[int, set] = {}
        for node, cid in part.items():
            groups.setdefault(cid, set()).add(node)
        return [g for g in groups.values() if len(g) >= 3]
    except ImportError:
        return [set(c) for c in
                nx.algorithms.community.greedy_modularity_communities(
                    G, weight="weight") if len(c) >= 3]


def detect_rings(scored_txns: pd.DataFrame, features: pd.DataFrame,
                 transactions: pd.DataFrame, config: dict) -> list[dict]:
    """Detect fraud rings; return rings.json-style list of dicts."""
    seed = int(config.get("seed", 42))
    G = build_link_graph(scored_txns, transactions, config)
    comms = find_communities(G, seed)
    score_by_acct = scored_txns.groupby("account_id")["txn_risk_score"].mean()
    rings: list[dict] = []
    for i, members in enumerate(sorted(comms, key=len, reverse=True)):
        members = sorted(members)
        sub = transactions[transactions["account_id"].isin(members)]
        shared_dev = sub.groupby("device_id")["account_id"].nunique()
        shared_dev = shared_dev[shared_dev >= 2].index.tolist()
        shared_ip = sub.groupby("ip_address")["account_id"].nunique()
        shared_ip = shared_ip[shared_ip >= 2].index.tolist()
        dests = sub[sub["dest_account_id"].notna() & (sub["dest_account_id"] != "")]
        shared_dst = dests.groupby("dest_account_id")["account_id"].nunique()
        shared_dst = shared_dst[shared_dst >= 2].index.tolist()
        links = []
        if shared_dev:
            links.append("shared_device")
        if shared_ip:
            links.append("shared_ip")
        if shared_dst:
            links.append("shared_destination")
        seqs = {_seq_signature(transactions, a) for a in members}
        seq_sim = round(1.0 - len(seqs) / max(len(members), 1), 3)
        if seq_sim >= 0.5:
            links.append("purchase_sequence")
        try:
            span = (f"{pd.to_datetime(sub['timestamp']).min().date()} to "
                    f"{pd.to_datetime(sub['timestamp']).max().date()}")
        except Exception:
            span = "unknown"
        window = f"clustered account activity window ({span})"
        avg_score = float(score_by_acct.reindex(members).fillna(0).mean())
        ring_score = round(min(99.0, avg_score + 5 * len(members)), 1)
        rings.append({
            "ring_id": f"ring_{i + 1}",
            "accounts": members, "size": len(members),
            "ring_risk_score": ring_score,
            "link_types_found": links or ["shared_device"],
            "evidence": {
                "shared_devices": [str(d) for d in shared_dev[:10]],
                "shared_ips": [str(x) for x in shared_ip[:10]],
                "shared_destinations": [str(d) for d in shared_dst[:10]],
                "purchase_sequence_similarity": seq_sim,
                "account_open_window": window,
            },
            "pattern_summary": (f"{len(members)} accounts linked by "
                                f"{', '.join(links) or 'shared entities'}; "
                                f"avg txn risk {avg_score:.0f}"),
            "recommended_actions": ["restrict", "manual_review", "file_sar"],
        })
    print(f"[detect_rings] found {len(rings)} ring(s).")
    return rings


def save_ring_outputs(rings: list[dict], scored_txns: pd.DataFrame,
                      transactions: pd.DataFrame, out_dir: str) -> None:
    """Write rings.json + per-ring HTML/PNG diagrams."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "rings.json").write_text(json.dumps(rings, indent=2))
    for r in rings:
        rid = r["ring_id"]
        members = set(r["accounts"])
        sub = transactions[transactions["account_id"].isin(members)]
        # PNG via matplotlib.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import networkx as nx
            G = nx.Graph()
            for _, row in sub.iterrows():
                G.add_edge(str(row["account_id"]),
                           f"dev:{row['device_id']}")
                if row.get("dest_account_id"):
                    G.add_edge(str(row["account_id"]),
                               f"dst:{row['dest_account_id']}")
            plt.figure(figsize=(8, 6))
            nx.draw_spring(G, with_labels=False, node_size=60)
            plt.title(f"{rid} (n={len(members)})")
            plt.savefig(out / f"{rid}.png", dpi=80)
            plt.close()
        except Exception as e:
            print(f"[detect_rings] PNG failed for {rid}: {e}")
        # HTML via pyvis.
        try:
            from pyvis.network import Network
            net = Network(height="600px", width="100%")
            for a in members:
                net.add_node(str(a), label=str(a), color="red")
            for _, row in sub.iterrows():
                d = f"dev:{row['device_id']}"
                net.add_node(d, label=d, color="lightblue")
                net.add_edge(str(row["account_id"]), d)
            net.save_graph(str(out / f"{rid}.html"))
        except Exception as e:
            print(f"[detect_rings] HTML failed for {rid}: {e}")
            (out / f"{rid}.html").write_text(
                f"<html><body><h1>{rid}</h1><p>diagram unavailable</p></body></html>")
