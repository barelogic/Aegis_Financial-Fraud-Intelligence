import { useCallback, useEffect, useRef, useState } from "react";
import {
  getCase,
  getCases,
  getEntity,
  getEvaluation,
  getEvents,
  getHealth,
  replayControl,
} from "../api";
import type {
  EntityDetail,
  FraudCase,
  FraudEvent,
  Health,
  ReplayOp,
  ReplaySpeed,
} from "../types";

export interface FraudIntelState {
  health: Health | null;
  events: FraudEvent[];
  cases: FraudCase[];
  evaluation: Record<string, unknown> | null;
  selected: EntityDetail | null;
  loading: boolean;
  error: string | null;
  speed: ReplaySpeed;
  showAll: boolean;
  refreshAll: () => Promise<void>;
  control: (op: ReplayOp) => Promise<void>;
  setSpeed: (s: ReplaySpeed) => void;
  setShowAll: (v: boolean) => void;
  selectEntity: (id: string) => Promise<void>;
  selectCase: (id: string) => Promise<void>;
  clearSelection: () => void;
}

/**
 * Central data hook. Mirrors the old ui/app.js behaviour:
 * health polling + SSE stream (falls back to polling) driving
 * feed + cases refreshes.
 */
export function useFraudIntel(): FraudIntelState {
  const [health, setHealth] = useState<Health | null>(null);
  const [events, setEvents] = useState<FraudEvent[]>([]);
  const [cases, setCases] = useState<FraudCase[]>([]);
  const [evaluation, setEvaluation] = useState<Record<string, unknown> | null>(
    null,
  );
  const [selected, setSelected] = useState<EntityDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [speed, setSpeed] = useState<ReplaySpeed>(5);
  const [showAll, setShowAll] = useState(false);
  const inFlight = useRef(false);

  const refreshAll = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    try {
      const h = await getHealth();
      setHealth(h);
      setError(null);
      // Feed follows the replay cursor so rows visibly stream during
      // playback. Cursor 0 (fresh/reset/paused-at-start) shows the tail,
      // where the fraud clusters; otherwise show the 50 rows ending at
      // the cursor.
      const end = h.cursor > 0 ? Math.min(h.cursor, h.n_events) : h.n_events;
      const { events: evts } = await getEvents(Math.max(0, end - 50), 50);
      setEvents(evts);
      const { cases: cs } = await getCases();
      setCases(cs);
      try {
        setEvaluation(await getEvaluation());
      } catch {
        setEvaluation(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "server unreachable");
    } finally {
      setLoading(false);
      inFlight.current = false;
    }
  }, []);

  const control = useCallback(
    async (op: ReplayOp) => {
      await replayControl(op, speed);
      await refreshAll();
    },
    [refreshAll, speed],
  );

  const selectEntity = useCallback(async (id: string) => {
    try {
      setSelected(await getEntity(id));
    } catch (e) {
      setError(e instanceof Error ? e.message : `entity not found: ${id}`);
    }
  }, []);

  const selectCase = useCallback(async (id: string) => {
    try {
      const c = await getCase(id);
      setSelected({ kind: "case", ...c });
    } catch (e) {
      setError(e instanceof Error ? e.message : `case not found: ${id}`);
    }
  }, []);

  const clearSelection = useCallback(() => setSelected(null), []);

  // Initial load + 5s health-driven refresh (matches old setInterval).
  useEffect(() => {
    void refreshAll();
    const t = window.setInterval(() => void refreshAll(), 5000);
    return () => window.clearInterval(t);
  }, [refreshAll]);

  // Live updates via SSE; reconnect after 5s on error (old behaviour).
  useEffect(() => {
    let src: EventSource | null = null;
    let closed = false;
    const connect = () => {
      try {
        src = new EventSource("/v1/events/stream");
        src.onmessage = () => void refreshAll();
        src.onerror = () => {
          src?.close();
          src = null;
          if (!closed) window.setTimeout(connect, 5000);
        };
      } catch {
        /* EventSource unavailable: polling above already covers us */
      }
    };
    connect();
    return () => {
      closed = true;
      src?.close();
    };
  }, [refreshAll]);

  return {
    health,
    events,
    cases,
    evaluation,
    selected,
    loading,
    error,
    speed,
    showAll,
    refreshAll,
    control,
    setSpeed,
    setShowAll,
    selectEntity,
    selectCase,
    clearSelection,
  };
}
