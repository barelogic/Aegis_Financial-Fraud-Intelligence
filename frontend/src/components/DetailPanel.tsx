import { Fragment } from "react";
import type { ReactElement } from "react";
import type { EntityDetail } from "../types";
import { bandOf, formatAmount, splitReasons } from "../utils";
import { Pill } from "./ui";

function KvRows({ obj, keys }: { obj: Record<string, unknown>; keys: string[] }): ReactElement {
  return (
    <>
      {keys
        .filter((k) => obj[k] != null && obj[k] !== "")
        .map((k) => (
          <Fragment key={k}>
            <dt>{k}</dt>
            <dd>{String(obj[k])}</dd>
          </Fragment>
        ))}
    </>
  );
}

export function DetailPanel({ entity }: { entity: EntityDetail | null }): ReactElement {
  if (!entity) {
    return <div className="empty">Click a row, node, or case…</div>;
  }
  return (
    <div id="detail">
      {entity.kind === "account" && (
        <>
          <div className="dhead">
            <span className="did">{entity.account_id}</span>
            <Pill band={String(entity.account_risk_band ?? "?")} />
            <span className="mono">score {String(entity.account_risk_score ?? "?")}</span>
            {entity.ring_id ? <span className="chip">ring: {entity.ring_id}</span> : null}
            <span className="chip">action: {String(entity.account_action ?? "")}</span>
          </div>
          {entity.account_reasons && (
            <ul className="reasons">
              {splitReasons(entity.account_reasons).map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
          <dl>
            <KvRows obj={entity} keys={["n_txns", "segment", "home_city"]} />
          </dl>
          {entity.transactions?.length ? (
            <div className="meta" style={{ color: "var(--muted)" }}>
              {entity.transactions.length} transactions (latest:{" "}
              <span className="mono">{entity.transactions.slice(-3).join(", ")}</span>)
            </div>
          ) : null}
        </>
      )}
      {entity.kind === "transaction" && (
        <>
          <div className="dhead">
            <span className="did">{entity.txn_id}</span>
            <Pill band={String(entity.txn_risk_band ?? "?")} />
            <span className="mono">score {String(entity.txn_risk_score ?? "?")}</span>
            <span className="chip">{formatAmount(entity)}</span>
            <span className="chip">action: {String(entity.txn_action ?? "")}</span>
            {entity.live ? <span className="chip live">LIVE</span> : null}
          </div>
          {entity.txn_reasons && (
            <ul className="reasons">
              {splitReasons(entity.txn_reasons).map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
          <dl>
            <KvRows
              obj={entity}
              keys={[
                "account_id",
                "timestamp",
                "txn_type",
                "channel",
                "merchant_category",
                "city",
                "device_id",
                "ip_address",
                "dest_account_id",
              ]}
            />
          </dl>
          <div className="meta" style={{ color: "var(--muted)" }}>
            band via <span className="mono">{bandOf(entity)}</span>
          </div>
        </>
      )}
      {entity.kind === "case" && (
        <>
          <div className="dhead">
            <span className="did">{entity.case_id ?? "?"}</span>
            <Pill band={String(entity.severity ?? "?")} />
          </div>
          {entity.severity_reason && <div>{entity.severity_reason}</div>}
          {(entity.transaction_ids?.length ?? 0) > 0 && (
            <div className="meta" style={{ color: "var(--muted)" }}>
              {entity.transaction_ids!.length} transactions
            </div>
          )}
        </>
      )}
      <details>
        <summary>raw JSON</summary>
        <pre>{JSON.stringify(entity, null, 1)}</pre>
      </details>
    </div>
  );
}
