import { GitCompareArrows, LoaderCircle } from "lucide-react";
import { useHarmonize } from "../../hooks/useHarmonize";
import { useStudioStore } from "../../store";
import { EngineSlot } from "./EngineSlot";
import { ViolationScore } from "./ViolationScore";

export function ComparisonPanel() {
  const engines = useStudioStore((state) => state.engines);
  const engineStatus = useStudioStore((state) => state.enginesStatus);
  const engineError = useStudioStore((state) => state.enginesError);
  const slots = useStudioStore((state) => state.slots);
  const viewMode = useStudioStore((state) => state.viewMode);
  const setViewMode = useStudioStore((state) => state.setViewMode);
  const melodyRevision = useStudioStore((state) => state.melodyRevision);
  const melodyCount = useStudioStore((state) => state.melody.notes.length);
  const { compareBoth } = useHarmonize();
  const loading =
    slots.A.status === "loading" || slots.B.status === "loading";
  const engineName = (id: string) =>
    engines.find((engine) => engine.id === id)?.name ?? id;

  return (
    <aside className="comparison-panel" aria-label="Engine comparison">
      <div className="comparison-heading">
        <div className="eyebrow">
          <GitCompareArrows size={13} />
          ENGINE LAB
        </div>
        <h2>Same melody. Different decisions.</h2>
        <p>
          Compare voiced parts and their defects—not just a final MIDI file.
        </p>
      </div>

      <div className="view-switcher" aria-label="Comparison view">
        {(["A", "B"] as const).map((view) => (
          <button
            type="button"
            key={view}
            className={viewMode === view ? "active" : ""}
            onClick={() => setViewMode(view)}
            disabled={!slots[view].result}
            aria-pressed={viewMode === view}
            title={`Show result ${view} (${view === "A" ? "1" : "2"})`}
          >
            Result {view}
            <kbd>{view === "A" ? "1" : "2"}</kbd>
          </button>
        ))}
      </div>

      {engineStatus === "error" && (
        <div className="panel-error">{engineError}</div>
      )}

      <div className="score-comparison">
        {(["A", "B"] as const).map((slot) => (
          <ViolationScore
            key={slot}
            slot={slot}
            engineName={engineName(slots[slot].engineId)}
            violations={slots[slot].result?.violations}
            latency={slots[slot].result?.latencyMs}
            stale={
              slots[slot].result !== undefined &&
              slots[slot].requestRevision !== melodyRevision
            }
          />
        ))}
      </div>

      <div className="engine-slots">
        <EngineSlot slot="A" />
        <EngineSlot slot="B" />
      </div>

      <button
        type="button"
        className="compare-button"
        onClick={() => void compareBoth()}
        disabled={
          loading ||
          melodyCount === 0 ||
          engineStatus === "loading" ||
          engines.length === 0
        }
      >
        {loading ? (
          <LoaderCircle size={15} className="spin" />
        ) : (
          <GitCompareArrows size={15} />
        )}
        {loading ? "Generating both…" : "Compare both engines"}
      </button>
    </aside>
  );
}
