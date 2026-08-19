import {
  Maximize2,
  Minus,
  MousePointer2,
  Plus,
  RefreshCw,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { useHarmonize } from "../../hooks/useHarmonize";
import { useStudioStore } from "../../store";
import { voiceLabel } from "../../utils/music";
import { MelodyOctaveControl } from "./MelodyOctaveControl";
import { SelectedNoteInspector } from "./SelectedNoteInspector";

export function EditorToolbar() {
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const setZoom = useStudioStore((state) => state.setZoom);
  const snap = useStudioStore((state) => state.snap);
  const setSnap = useStudioStore((state) => state.setSnap);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const focusedLane = useStudioStore((state) => state.focusedLane);
  const setFocusedLane = useStudioStore((state) => state.setFocusedLane);
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
            <span>marquee select · ruler cycle · arrows nudge</span>
          </div>
        </div>
        {focusedLane && (
          <div className="focus-mode-pill">
            <Maximize2 size={12} />
            <span>
              {focusedLane === "melody"
                ? "Melody"
                : voiceLabel(focusedLane)}{" "}
              focus
            </span>
            <button
              type="button"
              onClick={() => setFocusedLane(undefined)}
              aria-label="Exit lane focus"
              title="Exit focus (Escape)"
            >
              <X size={11} />
            </button>
          </div>
        )}
        <MelodyOctaveControl />
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
