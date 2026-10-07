import {useEffect, useRef} from "react";
import type {ReactElement} from "react";
import { DataSet } from "vis-data";
import { Network } from "vis-network/standalone";
import type { FraudEvent } from "../types";
import { NODE_COLORS, bandOf, formatAmount, isTransfer } from "../utils";

interface Props {
  events: FraudEvent[];
  showAll: boolean;
  onSelect: (id: string) => void;
}

/**
 * Transfer graph (vis-network, same library as the legacy ui/).
 * Shows the last 40 transfers (or all events when showAll is on).
 */
export function TransferGraph({ events, showAll, onSelect }: Props): ReactElement {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const networkRef = useRef<Network | null>(null);
  const selectRef = useRef(onSelect);
  selectRef.current = onSelect;

  const selected = events.filter((e) => showAll || isTransfer(e)).slice(-40);

  useEffect(() => {
    if (!containerRef.current) return;
    if (!networkRef.current) {
      networkRef.current = new Network(containerRef.current, { nodes: [], edges: [] }, {
        physics: { barnesHut: { gravitationalConstant: -4000 }, stabilization: true },
        interaction: { hover: true, tooltipDelay: 100 },
      });
      networkRef.current.on("click", (p: { nodes: string[] }) => {
        if (p.nodes.length) selectRef.current(p.nodes[0]);
      });
    }
    return () => {
      networkRef.current?.destroy();
      networkRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!networkRef.current) return;
    const nodes = new Map<string, any>();
    const edges: any[] = [];
    for (const e of selected) {
      const a = String(e.account_id);
      if (!nodes.has(a)) {
        nodes.set(a, {
          id: a,
          label: a.length > 10 ? a.slice(-8) : a,
          color: { background: "#1a2130", border: NODE_COLORS[bandOf(e)] ?? "#888" },
          font: { color: "#dbe2ef" },
          shape: "dot",
          size: 14,
          title: `${a} (risk ${String(e.txn_risk_score ?? "?")})`,
        });
      }
      const d = e.dest_account_id ? String(e.dest_account_id) : null;
      if (d) {
        if (!nodes.has(d)) {
          nodes.set(d, {
            id: d,
            label: d.length > 10 ? d.slice(-8) : d,
            color: { background: "#12283a", border: "#48c" },
            font: { color: "#dbe2ef" },
            shape: "diamond",
            size: 14,
            title: d,
          });
        }
        edges.push({
          from: a,
          to: d,
          color: { color: "#3a4a6b" },
          title: `${e.txn_id}: ${formatAmount(e)} (risk ${String(e.txn_risk_score ?? "?")})`,
        });
      }
    }
    networkRef.current.setData({
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      nodes: new DataSet([...nodes.values()]),
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      edges: new DataSet(edges),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events, showAll]);

  return (
    <>
      <div id="graph" ref={containerRef} />
      <div className="legend">
        <span>
          <i style={{ background: "#f44" }} />
          critical
        </span>
        <span>
          <i style={{ background: "#f80" }} />
          high
        </span>
        <span>
          <i style={{ background: "#dd4" }} />
          medium
        </span>
        <span>
          <i style={{ background: "#4a4" }} />
          low
        </span>
        <span>
          <i style={{ background: "#48c" }} />
          destination
        </span>
        <span>
          <i style={{ background: "#5b8cff" }} />
          scored live (not replayed)
        </span>
      </div>
      {!selected.length && (
        <div className="empty">0 — no transfers in window</div>
      )}
    </>
  );
}
