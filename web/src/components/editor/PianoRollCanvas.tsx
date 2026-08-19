import { useEffect, useMemo, useRef } from "react";
import { useStudioStore } from "../../store";
import { drawRoll } from "./drawRoll";
import { ROLL_HEIGHT } from "./rollGeometry";
import type { DrawModel } from "./rollTypes";
import { useRollInteraction } from "./useRollInteraction";

interface PianoRollCanvasProps {
  width: number;
  duration: number;
}

export function PianoRollCanvas({
  width,
  duration,
}: PianoRollCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawResultRef = useRef<ReturnType<typeof drawRoll> | undefined>(
    undefined,
  );
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const viewMode = useStudioStore((state) => state.viewMode);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const selectedNote = useStudioStore((state) => state.selectedNote);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const loopEnabled = useStudioStore((state) => state.loopEnabled);
  const loopStart = useStudioStore((state) => state.loopStart);
  const loopEnd = useStudioStore((state) => state.loopEnd);
  const voiceVisibility = useStudioStore((state) => state.voiceVisibility);
  const interaction = useRollInteraction(drawResultRef, duration);

  const model = useMemo<DrawModel>(
    () => ({
      melody,
      resultA: slots.A.result,
      resultB: slots.B.result,
      viewMode,
      activeSlot,
      selectedNote,
      pxPerBeat,
      duration,
      loopEnabled,
      loopStart,
      loopEnd,
      voiceVisibility,
    }),
    [
      activeSlot,
      duration,
      loopEnabled,
      loopEnd,
      loopStart,
      melody,
      pxPerBeat,
      selectedNote,
      slots.A.result,
      slots.B.result,
      viewMode,
      voiceVisibility,
    ],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    const pixelWidth = Math.ceil(width * ratio);
    const pixelHeight = Math.ceil(ROLL_HEIGHT * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawResultRef.current = drawRoll(context, model);
  }, [model, width]);

  return (
    <div className="roll-canvas-wrap" style={{ width, height: ROLL_HEIGHT }}>
      <canvas
        ref={canvasRef}
        className="roll-canvas"
        style={{ width, height: ROLL_HEIGHT }}
        aria-label="Editable multi-voice piano roll. Double-click the melody lane to add a note."
        role="application"
        tabIndex={0}
        onPointerDown={interaction.onPointerDown}
        onPointerMove={interaction.onPointerMove}
        onPointerLeave={interaction.onPointerLeave}
        onPointerUp={interaction.onPointerUp}
        onPointerCancel={interaction.onPointerUp}
        onDoubleClick={interaction.onDoubleClick}
        onContextMenu={interaction.onContextMenu}
      />
      <div
        className="roll-playhead"
        aria-hidden="true"
        style={{ transform: `translateX(${currentBeat * pxPerBeat}px)` }}
      />
      {interaction.hoveredViolation && (
        <div
          className={`violation-tooltip severity-${interaction.hoveredViolation.violation.severity}`}
          style={{
            left: Math.min(
              width - 260,
              interaction.hoveredViolation.rect.x + 12,
            ),
            top: Math.max(34, interaction.hoveredViolation.rect.y - 10),
          }}
          role="tooltip"
        >
          <strong>
            {interaction.hoveredViolation.slot} ·{" "}
            {interaction.hoveredViolation.violation.kind.replaceAll("_", " ")}
          </strong>
          <span>{interaction.hoveredViolation.violation.message}</span>
          <small>
            {interaction.hoveredViolation.violation.voices.join(" ↔ ")}
          </small>
        </div>
      )}
    </div>
  );
}
