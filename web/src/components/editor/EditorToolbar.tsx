import {
  Minus,
  MousePointer2,
  Plus,
  RefreshCw,
  SlidersHorizontal,
} from "lucide-react";
import { useHarmonize } from "../../hooks/useHarmonize";
import { useStudioStore } from "../../store";
import { SelectedNoteInspector } from "./SelectedNoteInspector";

export function EditorToolbar() {
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const setZoom = useStudioStore((state) => state.setZoom);
  const snap = useStudioStore((state) => state.snap);
  const setSnap = useStudioStore((state) => state.setSnap);
  const viewMode = useStudioStore((state) => state.viewMode);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const setActiveSlot = useStudioStore((state) => state.setActiveSlot);
  const slots = useStudioStore((state) => state.slots);
  const melodyRevision = useStudioStore((state) => state.melodyRevision);
  const { harmonizeSlot } = useHarmonize();
  const slot = slots[activeSlot];
  const stale =
    slot.result !== undefined && slot.requestRevision !== melodyRevision;

  return (
    <div className="editor-toolbar">
      <div className="editor-toolbar-top">
        <div className="editor-title">
          <MousePointer2 size={15} />
          <div>
            <strong>Voicing editor</strong>
            <span>drag · resize edge · right-click delete</span>
          </div>
        </div>
        {viewMode === "overlay" && (
          <div className="edit-layer-toggle" aria-label="Editable overlay">
            <span>Edit</span>
            {(["A", "B"] as const).map((value) => (
              <button
                type="button"
                key={value}
                className={activeSlot === value ? "active" : ""}
                onClick={() => setActiveSlot(value)}
                aria-pressed={activeSlot === value}
              >
                {value}
              </button>
            ))}
          </div>
        )}
        <div className="roll-tool-group">
          <label className="snap-control">
            <SlidersHorizontal size={13} />
            <span>Snap</span>
            <select
              value={snap}
              onChange={(event) => setSnap(Number(event.currentTarget.value))}
            >
              <option value={1}>1/4</option>
              <option value={0.5}>1/8</option>
              <option value={0.25}>1/16</option>
              <option value={0.125}>1/32</option>
            </select>
          </label>
          <button
            type="button"
            className="icon-button"
            onClick={() => setZoom(pxPerBeat - 12)}
            aria-label="Zoom out"
          >
            <Minus size={14} />
          </button>
          <span className="zoom-readout">{Math.round(pxPerBeat)} px</span>
          <button
            type="button"
            className="icon-button"
            onClick={() => setZoom(pxPerBeat + 12)}
            aria-label="Zoom in"
          >
            <Plus size={14} />
          </button>
        </div>
        {stale && (
          <button
            type="button"
            className="reharmonize-button"
            onClick={() => void harmonizeSlot(activeSlot)}
            disabled={slot.status === "loading"}
          >
            <RefreshCw
              size={13}
              className={slot.status === "loading" ? "spin" : ""}
            />
            Re-harmonize {activeSlot}
          </button>
        )}
      </div>
      <SelectedNoteInspector />
    </div>
  );
}
