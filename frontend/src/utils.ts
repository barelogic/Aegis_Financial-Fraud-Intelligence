export function bandOf(e: { txn_risk_band?: unknown; severity?: unknown }): string {
  return String(e.txn_risk_band ?? e.severity ?? "low").toLowerCase();
}

export function formatAmount(e: { amount?: unknown }): string {
  if (e.amount == null || e.amount === "") return "?";
  const n = Number(e.amount);
  if (Number.isNaN(n)) return String(e.amount);
  return `Rs ${n.toLocaleString("en-IN")}`;
}

export function shortTime(ts?: unknown): string {
  return String(ts ?? "").slice(0, 16).replace("T", " ");
}

export function isTransfer(e: { txn_type?: unknown; dest_account_id?: unknown }): boolean {
  return (
    (e.txn_type === "transfer" || e.txn_type === "cash_out") &&
    Boolean(e.dest_account_id)
  );
}

export const NODE_COLORS: Record<string, string> = {
  critical: "#f44",
  high: "#f80",
  medium: "#dd4",
  low: "#4a4",
};

export function splitReasons(reasons?: string | null): string[] {
  if (!reasons) return [];
  return String(reasons)
    .split(";")
    .map((s) => s.trim())
    .filter(Boolean);
}
