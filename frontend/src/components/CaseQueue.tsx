import type { ReactElement } from "react";
import type { FraudCase } from "../types";
import { Pill } from "./ui";

interface Props {
  cases: FraudCase[];
  onSelect: (caseId: string) => void;
}

export function CaseQueue({ cases, onSelect }: Props): ReactElement {
  if (!cases.length) return <div className="empty">no cases</div>;
  return (
    <div>
      {cases.map((c) => {
        const nEnt = c.entities?.length ?? 0;
        const nTxn = c.transaction_ids?.length ?? 0;
        return (
          <div key={c.case_id} className="case" onClick={() => onSelect(c.case_id)}>
            <div className="row1">
              <span className="cid">{c.case_id}</span>
              <Pill band={String(c.severity ?? "?")} />
            </div>
            <div className="meta">
              {nEnt} entities · {nTxn} txns
              <br />
              {(c.typologies ?? []).map((t) => (
                <span key={t} className="chip">
                  {t}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
