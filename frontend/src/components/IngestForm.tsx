import {useState} from "react";
import type {FormEvent, ReactElement} from "react";
import { ingestEvent } from "../api";
import type { FraudEvent } from "../types";

const FIELDS: Array<{ key: string; label: string; placeholder: string; required?: boolean }> = [
  { key: "txn_id", label: "txn_id", placeholder: "live_1", required: true },
  { key: "timestamp", label: "timestamp", placeholder: "2025-02-01 10:00:00", required: true },
  { key: "account_id", label: "account_id", placeholder: "acc_0001", required: true },
  { key: "amount", label: "amount", placeholder: "42000", required: true },
  { key: "merchant_category", label: "merchant_category", placeholder: "electronics" },
  { key: "channel", label: "channel", placeholder: "online" },
  { key: "device_id", label: "device_id", placeholder: "dev_new" },
  { key: "city", label: "city", placeholder: "Delhi" },
  { key: "txn_type", label: "txn_type", placeholder: "purchase" },
  { key: "dest_account_id", label: "dest_account_id", placeholder: "(transfer only)" },
];

const DEFAULTS: Record<string, string> = {
  timestamp: "2025-02-01 10:00:00",
  merchant_category: "electronics",
  channel: "online",
  city: "Delhi",
  txn_type: "purchase",
};

/**
 * Live ingest form. Only works when the backend runs with --live;
 * otherwise the server returns 400 "live mode is off".
 */
export function IngestForm({ onIngested }: { onIngested: () => void }): ReactElement {
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<Record<string, string>>({ ...DEFAULTS });
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const set = (k: string, v: string) => setForm((f) => ({ ...f, [k]: v }));

  async function submit(e: FormEvent): Promise<void> {
    e.preventDefault();
    setBusy(true);
    setResult(null);
    try {
      const raw: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(form)) {
        if (v !== "") raw[k] = k === "amount" ? Number(v) : v;
      }
      const res = await ingestEvent(raw);
      const ev: FraudEvent = res.event;
      setResult(`ok: ${String(ev.txn_id)} → ${String(ev.txn_risk_score)} (${String(ev.txn_risk_band)})`);
      onIngested();
    } catch (err) {
      setResult(err instanceof Error ? `error: ${err.message}` : "error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h3>
        Live ingest{" "}
        <span className="count" title="Requires --live backend">
          --live
        </span>
        <button style={{ marginLeft: "auto" }} onClick={() => setOpen((o) => !o)}>
          {open ? "Hide" : "Score a transaction"}
        </button>
      </h3>
      {open && (
        <form onSubmit={submit} className="ingest-form">
          {FIELDS.map((f) => (
            <label key={f.key}>
              {f.label}
              {f.required ? " *" : ""}
              <input
                value={form[f.key] ?? ""}
                placeholder={f.placeholder}
                onChange={(e) => set(f.key, e.target.value)}
                required={f.required}
              />
            </label>
          ))}
          <div>
            <button type="submit" className="primary" disabled={busy}>
              {busy ? "Scoring…" : "POST /v1/events/ingest"}
            </button>
            {result && <span className="mono" style={{ marginLeft: 10 }}>{result}</span>}
          </div>
          <div className="meta">
            Same schema as transactions.csv, no score columns. Unknown accounts are
            registered on first sight; duplicates get 409.
          </div>
        </form>
      )}
    </section>
  );
}
