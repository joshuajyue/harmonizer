import { Circle, Square } from "lucide-react";
import type { RecordingState } from "../../store";

interface RecordControlsProps {
  state: RecordingState;
  onArm: () => void;
  onToggleRecording: () => void;
}

export function RecordControls({
  state,
  onArm,
  onToggleRecording,
}: RecordControlsProps) {
  const recording = state === "recording";
  return (
    <>
      <button
        type="button"
        className={`record-arm-button ${state !== "idle" ? "active" : ""}`}
        onClick={onArm}
        disabled={recording}
        aria-label={state === "idle" ? "Arm recording" : "Disarm recording"}
        aria-pressed={state !== "idle"}
        title="Arm performance capture"
      >
        <Circle size={10} fill="currentColor" />
        {state === "idle" ? "Arm" : "Armed"}
      </button>
      <button
        type="button"
        className={`record-take-button ${recording ? "active" : ""}`}
        onClick={onToggleRecording}
        disabled={state === "idle"}
        aria-label={recording ? "Stop recording" : "Start recording"}
        aria-pressed={recording}
        title="Raw-timing overdub; same-pitch retriggers close the prior note"
      >
        {recording ? (
          <Square size={10} fill="currentColor" />
        ) : (
          <Circle size={10} fill="currentColor" />
        )}
        {recording ? "Stop" : "Record"}
      </button>
    </>
  );
}
