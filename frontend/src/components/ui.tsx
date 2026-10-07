import type { ReactElement, ReactNode } from "react";
import { bandOf } from "../utils";

export function Pill({ band }: { band: string }): ReactElement {
  const b = bandOf({ txn_risk_band: band });
  return <span className={`pill ${b}`}>{band}</span>;
}

export function ScoreBar({
  score,
  band,
}: {
  score?: number | null;
  band?: string;
}): ReactElement {
  const colors: Record<string, string> = {
    critical: "#f44",
    high: "#f80",
    medium: "#dd4",
    low: "#4a4",
  };
  const v = Math.max(0, Math.min(100, Number(score) || 0));
  const color = colors[bandOf({ txn_risk_band: band })] ?? "#888";
  return (
    <span className="scorebar">
      <span style={{ width: `${v}%`, background: color }} />
    </span>
  );
}

export function Section({
  title,
  count,
  children,
}: {
  title: string;
  count?: ReactNode;
  children: ReactNode;
}): ReactElement {
  return (
    <section>
      <h3>
        {title} <span className="count">{count ?? "–"}</span>
      </h3>
      {children}
    </section>
  );
}
