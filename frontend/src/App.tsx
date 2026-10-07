import {useState} from "react";
import type {ReactElement} from "react";
import { refreshRings } from "./api";
import { CaseQueue } from "./components/CaseQueue";
import { Controls } from "./components/Controls";
import { DetailPanel } from "./components/DetailPanel";
import { EvaluationPanel } from "./components/EvaluationPanel";
import { EventFeed } from "./components/EventFeed";
import { IngestForm } from "./components/IngestForm";
import { TransferGraph } from "./components/TransferGraph";
import { Section } from "./components/ui";
import { useFraudIntel } from "./hooks/useFraudIntel";
import "./styles.css";

export default function App(): ReactElement {
  const intel = useFraudIntel();
  const [ringsBusy, setRingsBusy] = useState(false);
  const [ringsMsg, setRingsMsg] = useState<string | null>(null);

  async function handleRefreshRings(): Promise<void> {
    setRingsBusy(true);
    setRingsMsg(null);
    try {
      const r = await refreshRings();
      setRingsMsg(`ok: ${r.n_rings} rings`);
      await intel.refreshAll();
    } catch (e) {
      setRingsMsg(e instanceof Error ? e.message : "refresh failed");
    } finally {
      setRingsBusy(false);
    }
  }

  return (
    <>
      <Controls
        health={intel.health}
        speed={intel.speed}
        showAll={intel.showAll}
        onControl={(op) => void intel.control(op)}
        onSpeed={intel.setSpeed}
        onShowAll={intel.setShowAll}
        onRefreshRings={() => void handleRefreshRings()}
        ringsBusy={ringsBusy}
      />
      {intel.error && (
        <div className="banner-error" role="alert">
          {intel.error}{" "}
          <button onClick={() => void intel.refreshAll()}>retry</button>
        </div>
      )}
      {ringsMsg && <div className="banner-info">{ringsMsg}</div>}
      <main>
        <Section title="Event feed" count={intel.events.length || "–"}>
          <EventFeed
            events={intel.events}
            loading={intel.loading}
            onSelect={(id) => void intel.selectEntity(id)}
          />
        </Section>
        <div style={{ display: "flex", flexDirection: "column", gap: 12, minWidth: 0 }}>
          <Section title="Transfer graph" count="live">
            <TransferGraph
              events={intel.events}
              showAll={intel.showAll}
              onSelect={(id) => void intel.selectEntity(id)}
            />
          </Section>
          <Section title="Case queue" count={intel.cases.length}>
            <CaseQueue cases={intel.cases} onSelect={(id) => void intel.selectCase(id)} />
          </Section>
        </div>
        <div style={{ gridColumn: "1/-1", display: "grid", gap: 12 }}>
          <Section title="Detail">
            <DetailPanel entity={intel.selected} />
          </Section>
          <IngestForm onIngested={() => void intel.refreshAll()} />
          <Section title="Evaluation" count={intel.evaluation ? Object.keys(intel.evaluation).length : "–"}>
            <EvaluationPanel evaluation={intel.evaluation} />
          </Section>
        </div>
      </main>
      <footer>
        Decisions precomputed by the batch pipeline · streaming only reveals them in
        order · React + TypeScript UI proxied to the Python backend
      </footer>
    </>
  );
}
