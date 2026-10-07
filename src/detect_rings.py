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


def _item_order_pairs(transactions: pd.DataFrame, min_shared: int = 2,
                      max_buyers: int = 12) -> set[frozenset]:
    """Account pairs that bought >=min_shared uncommon items in the same order.

    'Uncommon' = bought by 2..max_buyers distinct accounts. Bulk/placeholder
    items (TRANSFER_OUT, popular goods) and one-off coincidences are excluded:
    a single shared item is never enough, which protects hard negatives like
    the laptop buyers (1 shared item) and supplier accounts (1 shared item).
    """
    tx = transactions[transactions["item_id"].notna()
                      & (transactions["item_id"] != "")]
    buyers = tx.groupby("item_id")["account_id"].apply(lambda s: sorted(set(s)))
    buyers = buyers[buyers.apply(len).between(2, max_buyers)]
    if buyers.empty:
        return set()
    # Chronological item list per account.
    ordered = (tx.sort_values("timestamp").groupby("account_id")["item_id"]
                 .apply(list))
    pos: dict[str, dict] = {a: {it: i for i, it in enumerate(seq)}
                            for a, seq in ordered.items()}
    shared: dict[frozenset, list] = {}
    for _item, accts in buyers.items():
        for u, v in itertools.combinations(accts, 2):
            shared.setdefault(frozenset((u, v)), []).append(_item)
    out: set[frozenset] = set()
    for pair, items in shared.items():
        if len(items) < min_shared:
            continue
        u, v = tuple(pair)
        pu, pv = pos.get(u, {}), pos.get(v, {})
        # Same relative order in both purchase histories?
        if (sorted(items, key=lambda it: pu.get(it, 10 ** 9))
                == sorted(items, key=lambda it: pv.get(it, 10 ** 9))):
            out.add(pair)
    return out


def build_link_graph(scored_txns: pd.DataFrame, transactions: pd.DataFrame,
                     config: dict,
                     item_order_pairs: set[frozenset] | None = None):
    """Build a weighted account graph from shared-entity links."""
    import networkx as nx
    w = config.get("link_weights", {})
    G = nx.Graph()
    for a in transactions["account_id"].unique():
        G.add_node(a)

    def _add_link(col: str, weight: float, min_share: int = 2) -> None:
        groups = transactions.groupby(col)["account_id"].apply(set)
        for key, users in groups.items():
            # Never link on empty/NaN keys: every purchase-only account
            # shares dest "" / NaN, which would hairball the whole graph.
            if key is None or key == "" or (
                    isinstance(key, float) and pd.isna(key)):
                continue
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
    # Same-order rare-item purchases (coordinated buying trips): pairs that
    # bought >=2 uncommon items in the same chronological order.
    if item_order_pairs is None:
        item_order_pairs = _item_order_pairs(transactions)
    # sorted(): set iteration order varies across processes (hash seed);
    # edge insertion order must be deterministic for byte-identical outputs.
    for pair in sorted(item_order_pairs, key=lambda p: sorted(tuple(p))):
        u, v = sorted(tuple(pair))
        if G.has_node(u) and G.has_node(v):
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
    min_fanin = int(config.get("min_ring_fanin", 4))
    # Item-order pairs are computed on GLOBAL buyer pools (an item that is
    # rare globally is meaningful; rarity inside one community is not).
    io_pairs = _item_order_pairs(transactions)
    G = build_link_graph(scored_txns, transactions, config,
                         item_order_pairs=io_pairs)
    comms = find_communities(G, seed)
    score_by_acct = scored_txns.groupby("account_id")["txn_risk_score"].mean()
    score_max_by_acct = scored_txns.groupby("account_id")["txn_risk_score"].max()
    rings: list[dict] = []
    kept = 0
    for members in sorted(comms, key=len, reverse=True):
        members = sorted(members)
        sub = transactions[transactions["account_id"].isin(members)]
        dev_counts = sub.groupby("device_id")["account_id"].nunique()
        shared_dev = dev_counts[dev_counts >= 2].index.tolist()
        ip_counts = sub.groupby("ip_address")["account_id"].nunique()
        shared_ip = ip_counts[ip_counts >= 2].index.tolist()
        dests = sub[sub["dest_account_id"].notna() & (sub["dest_account_id"] != "")]
        dst_counts = dests.groupby("dest_account_id")["account_id"].nunique()
        shared_dst = dst_counts[dst_counts >= 2].index.tolist()
        biggest = max([int(dev_counts.max()) if len(dev_counts) else 0,
                       int(ip_counts.max()) if len(ip_counts) else 0,
                       int(dst_counts.max()) if len(dst_counts) else 0])
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
        # Same-order rare-item buying: what fraction of member pairs share it?
        all_pairs = [frozenset(p) for p in
                     itertools.combinations(members, 2)]
        io_frac = (sum(1 for p in all_pairs if p in io_pairs)
                   / max(len(all_pairs), 1))
        if io_frac >= 0.5 and "purchase_sequence" not in links:
            links.append("purchase_sequence")
        avg_score = float(score_by_acct.reindex(members).fillna(0).mean())
        max_fanin = int(dst_counts[dst_counts >= 2].max()) if shared_dst else 0
        # Minimum-evidence filter: one weak link type alone (e.g. a single
        # shared IP) is not a ring. Exceptions: a high fan-in collector
        # destination (>= min_ring_fanin senders) is a classic money-mule
        # signal on its own, and high behavioral scores corroborate the link.
        single_dest_case = (links == ["shared_destination"]
                            and max_fanin >= min_fanin)
        if len(links) < 2 and not single_dest_case and avg_score < 65:
            continue
        # Evidence-based score: link diversity + sequence similarity + the
        # biggest shared-entity group. Size alone must NOT inflate risk.
        # Take the max with the members' worst transaction score: a ring
        # containing behaviorally hot transactions is hot even when its
        # average is diluted by legitimate background traffic. (The
        # minimum-evidence filter above already keeps this from inflating
        # weak groups.)
        evidence_score = (35 + 15 * len(links) + 10 * seq_sim
                          + min(10, biggest))
        member_max = float(score_max_by_acct.reindex(members).fillna(0).max())
        ring_score = round(min(99.0, max(evidence_score, member_max)), 1)
        try:
            span = (f"{pd.to_datetime(sub['timestamp']).min().date()} to "
                    f"{pd.to_datetime(sub['timestamp']).max().date()}")
        except Exception:
            span = "unknown"
        window = f"clustered account activity window ({span})"
        kept += 1
        rings.append({
            "ring_id": f"ring_{kept}",
            "accounts": members, "size": len(members),
            "ring_risk_score": ring_score,
            "link_types_found": links or ["shared_device"],
            "evidence": {
                "shared_devices": [str(d) for d in shared_dev[:10]],
                "shared_ips": [str(x) for x in shared_ip[:10]],
                "shared_destinations": [str(d) for d in shared_dst[:10]],
                "purchase_sequence_similarity": seq_sim,
                "max_destination_fanin": max_fanin,
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
