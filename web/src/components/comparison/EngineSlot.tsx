import { BrainCircuit, Cpu, LoaderCircle, WandSparkles } from "lucide-react";
import { useHarmonize } from "../../hooks/useHarmonize";
import { useStudioStore, type ComparisonSlotId } from "../../store";

interface EngineSlotProps {
  slot: ComparisonSlotId;
}

export function EngineSlot({ slot }: EngineSlotProps) {
  const engines = useStudioStore((state) => state.engines);
  const config = useStudioStore((state) => state.slots[slot]);
  const setEngine = useStudioStore((state) => state.setSlotEngine);
  const melodyCount = useStudioStore((state) => state.melody.notes.length);
  const { harmonizeSlot } = useHarmonize();
  const selected = engines.find((engine) => engine.id === config.engineId);

  return (
    <article className="engine-slot">
      <div className="engine-slot-heading">
        <span className={`slot-tag slot-${slot.toLowerCase()}`}>{slot}</span>
        <div>
          <strong>Engine {slot}</strong>
          <small>{selected?.learned ? "LEARNED MODEL" : "RULE SYSTEM"}</small>
        </div>
        {selected?.learned ? (
          <BrainCircuit size={16} />
        ) : (
          <Cpu size={16} />
        )}
      </div>
      <label className="engine-select">
        <span>HARMONIZER</span>
        <select
          value={config.engineId}
          onChange={(event) => setEngine(slot, event.currentTarget.value)}
          disabled={engines.length === 0}
        >
          {engines.map((engine) => (
            <option
              value={engine.id}
              key={engine.id}
              disabled={!engine.available}
            >
              {engine.name}
              {!engine.available ? " — unavailable" : ""}
            </option>
          ))}
        </select>
      </label>
      <p>{selected?.description ?? "Loading available engines…"}</p>
      {config.error && <div className="engine-error">{config.error}</div>}
      <button
        type="button"
        className="harmonize-slot-button"
        onClick={() => void harmonizeSlot(slot)}
        disabled={
          !config.engineId ||
          melodyCount === 0 ||
          config.status === "loading" ||
          selected?.available === false
        }
      >
        {config.status === "loading" ? (
          <LoaderCircle size={14} className="spin" />
        ) : (
          <WandSparkles size={14} />
        )}
        {config.status === "loading" ? "Voicing…" : `Harmonize ${slot}`}
      </button>
    </article>
  );
}
