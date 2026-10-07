import type {
  EntityDetail,
  FraudCase,
  FraudEvent,
  Health,
  ReplayOp,
  ReplaySpeed,
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = "";
    try {
      const body = (await res.json()) as { error?: string };
      if (body?.error) detail = `: ${body.error}`;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${path}: ${res.status}${detail}`);
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<Health> {
  return request<Health>("/v1/health");
}

export function getEvents(
  cursor: number,
  limit = 50,
): Promise<{ events: FraudEvent[]; next_cursor: number }> {
  return request(`/v1/events?cursor=${cursor}&limit=${limit}`);
}

export function replayControl(
  op: ReplayOp,
  speed: ReplaySpeed,
): Promise<Health> {
  return request("/v1/replay/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ op, speed }),
  });
}

export function getCases(): Promise<{ cases: FraudCase[] }> {
  return request<{ cases: FraudCase[] }>("/v1/cases");
}

export function getCase(id: string): Promise<FraudCase> {
  return request<FraudCase>(`/v1/cases/${encodeURIComponent(id)}`);
}

export function getEntity(id: string): Promise<EntityDetail> {
  return request<EntityDetail>(`/v1/entities/${encodeURIComponent(id)}`);
}

export function getEvaluation(): Promise<Record<string, unknown>> {
  return request<Record<string, unknown>>("/v1/evaluation");
}

export function ingestEvent(
  raw: Record<string, unknown>,
): Promise<{ ok: boolean; event: FraudEvent }> {
  return request("/v1/events/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(raw),
  });
}

export function refreshRings(): Promise<{
  ok: boolean;
  n_rings: number;
  rings: unknown[];
}> {
  return request("/v1/rings/refresh", { method: "POST" });
}
