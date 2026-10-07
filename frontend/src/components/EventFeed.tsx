import type { ReactElement } from "react";
import type { FraudEvent } from "../types";
import { bandOf, formatAmount, shortTime } from "../utils";
import { Pill, ScoreBar } from "./ui";

interface Props {
  events: FraudEvent[];
  loading: boolean;
  onSelect: (txnId: string) => void;
}

export function EventFeed({ events, loading, onSelect }: Props): ReactElement {
  if (!events.length) {
    return (
      <div className="empty">
        {loading ? "loading…" : "no events yet — press Start"}
      </div>
    );
  }
  return (
    <div style={{ maxHeight: 480, overflow: "auto" }}>
      <table>
        <thead>
          <tr>
            <th>time</th>
            <th>txn</th>
            <th>account</th>
            <th>amount</th>
            <th>score</th>
            <th>band</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e) => (
            <tr key={e.txn_id} onClick={() => onSelect(e.txn_id)}>
              <td className="mono">{shortTime(e.timestamp)}</td>
              <td className="mono">
                {e.txn_id} {e.live ? <span className="chip live">LIVE</span> : null}
              </td>
              <td className="mono">{String(e.account_id)}</td>
              <td>{formatAmount(e)}</td>
              <td className="mono">
                {String(e.txn_risk_score ?? "?")}
                <ScoreBar score={e.txn_risk_score} band={bandOf(e)} />
              </td>
              <td>
                <Pill band={String(e.txn_risk_band ?? "?")} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
