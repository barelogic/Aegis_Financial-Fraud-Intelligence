import type { ReactElement } from "react";
export function EvaluationPanel({
  evaluation,
}: {
  evaluation: Record<string, unknown> | null;
}): ReactElement {
  if (!evaluation) return <div className="empty">no metrics yet</div>;
  const entries = Object.entries(evaluation);
  return (
    <div style={{ maxHeight: 220, overflow: "auto" }}>
      <table>
        <thead>
          <tr>
            <th>metric</th>
            <th>value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k}>
              <td className="mono">{k}</td>
              <td className="mono">
                {typeof v === "object" ? JSON.stringify(v) : String(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
