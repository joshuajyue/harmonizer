import { Circle, Square } from "lucide-react";
import type {
  CountInBars,
  RecordingState,
} from "../../store";

interface RecordControlsProps {
  state: RecordingState;
  countInBars: CountInBars;
  countInRemaining: number;
  onArm: () => void;
  onToggleRecording: () => void;
  onCountInBarsChange: (bars: CountInBars) => void;
}

export function RecordControls({
  state,
  countInBars,
  countInRemaining,
  onArm,
  onToggleRecording,
  onCountInBarsChange,
}: RecordControlsProps) {
  const recording = state === "recording";
  const counting = state === "counting";
  const busy = recording || counting;
  return (
    <>
      <button
        type="button"
        className={`record-arm-button ${state !== "idle" ? "active" : ""}`}
        onClick={onArm}
        disabled={busy}
        aria-label={state === "idle" ? "Arm recording" : "Disarm recording"}
        aria-pressed={state !== "idle"}
        title="Arm performance capture"
      >
        <Circle size={10} fill="currentColor" />
        {state === "idle" ? "Arm" : "Armed"}
      </button>
      <button
        type="button"
        className={`record-take-button ${recording ? "active" : ""} ${counting ? "counting" : ""}`}
        onClick={onToggleRecording}
        disabled={state === "idle"}
        aria-label={
          counting
            ? "Cancel count-in"
            : recording
              ? "Stop recording"
              : "Start recording"
        }
        aria-pressed={busy}
        title="Record/stop (R) · raw timing · same-pitch retriggers close the prior note"
      >
        {busy ? (
          <Square size={10} fill="currentColor" />
        ) : (
          <Circle size={10} fill="currentColor" />
        )}
        {counting
          ? `Count ${countInRemaining || "…"}`
          : recording
            ? "Stop"
            : "Record"}
      </button>
      <label className="count-in-select">
        <span>Count-in</span>
        <select
          value={countInBars}
          onChange={(event) =>
            onCountInBarsChange(
              Number(event.currentTarget.value) as CountInBars,
            )
          }
          disabled={busy}
          aria-label="Recording count-in length"
        >
          <option value={0}>Off</option>
          <option value={1}>1 bar</option>
          <option value={2}>2 bars</option>
        </select>
      </label>
    </>
  );
}
