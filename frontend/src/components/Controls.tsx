import type { ReactElement } from "react";
import type { Health, ReplayOp, ReplaySpeed } from "../types";

interface Props {
  health: Health | null;
  speed: ReplaySpeed;
  showAll: boolean;
  onControl: (op: ReplayOp) => void;
  onSpeed: (s: ReplaySpeed) => void;
  onShowAll: (v: boolean) => void;
  onRefreshRings: () => void;
  ringsBusy: boolean;
}

export function Controls({
  health,
  speed,
  showAll,
  onControl,
  onSpeed,
  onShowAll,
  onRefreshRings,
  ringsBusy,
}: Props): ReactElement {
  const statusText = health
    ? `${health.playing ? "LIVE" : "paused"} · cursor ${health.cursor}/${health.n_events} · speed ${health.speed}×${health.live ? ` · live rows ${health.n_live ?? 0}` : ""}`
    : "connecting…";
  return (
    <header>
      <div className="brand">
        <span className="dot" />
        Fraud Intel <small>live replay · React</small>
      </div>
      <div className="controls">
        <button className="primary" onClick={() => onControl("start")}>
          ▶ Start
        </button>
        <button onClick={() => onControl("pause")}>⏸ Pause</button>
        <button onClick={() => onControl("reset")}>↺ Reset</button>
        <label className="toggle">
          Speed
          <select
            value={String(speed)}
            onChange={(e) => onSpeed(Number(e.target.value) as ReplaySpeed)}
          >
            <option value="1">1</option>
            <option value="5">5</option>
            <option value="20">20</option>
          </select>
        </label>
        <label className="toggle">
          <input
            type="checkbox"
            checked={showAll}
            onChange={(e) => onShowAll(e.target.checked)}
          />{" "}
          all events in graph
        </label>
        <button
          onClick={onRefreshRings}
          disabled={ringsBusy}
          title="POST /v1/rings/refresh (live mode)"
        >
          {ringsBusy ? "Refreshing…" : "⟳ Rings"}
        </button>
      </div>
      <span id="status" className={health?.playing ? "live" : ""}>
        <span className="pulse" />
        <span id="status-text">{statusText}</span>
      </span>
    </header>
  );
}
