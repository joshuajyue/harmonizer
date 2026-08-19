import { useEffect, useMemo, useRef } from "react";
import { useStudioStore } from "../../store";
import { drawRoll } from "./drawRoll";
import type { RollLayout } from "./rollGeometry";
import type { DrawModel } from "./rollTypes";
import { useRollInteraction } from "./useRollInteraction";

interface PianoRollCanvasProps {
  width: number;
  duration: number;
  layout: RollLayout;
}

export function PianoRollCanvas({
  width,
  duration,
  layout,
}: PianoRollCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawResultRef = useRef<ReturnType<typeof drawRoll> | undefined>(
    undefined,
  );
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const viewMode = useStudioStore((state) => state.viewMode);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const selectedNotes = useStudioStore((state) => state.selectedNotes);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const loopEnabled = useStudioStore((state) => state.loopEnabled);
  const loopStart = useStudioStore((state) => state.loopStart);
  const loopEnd = useStudioStore((state) => state.loopEnd);
  const voiceVisibility = useStudioStore((state) => state.voiceVisibility);
  const interaction = useRollInteraction(drawResultRef, duration, layout);

  const model = useMemo<DrawModel>(
    () => ({
      melody,
      resultA: slots.A.result,
      resultB: slots.B.result,
      viewMode,
      activeSlot,
      selectedNotes,
      pxPerBeat,
      duration,
      loopEnabled,
      loopStart,
      loopEnd,
      voiceVisibility,
      layout,
    }),
    [
      activeSlot,
      duration,
      loopEnabled,
      loopEnd,
      loopStart,
      layout,
      melody,
      pxPerBeat,
      selectedNotes,
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
    const pixelHeight = Math.ceil(layout.rollHeight * ratio);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    drawResultRef.current = drawRoll(context, model);
  }, [layout.rollHeight, model, width]);

  return (
    <div
      className="roll-canvas-wrap"
      style={{ width, height: layout.rollHeight }}
    >
      <canvas
        ref={canvasRef}
        className="roll-canvas"
        style={{ width, height: layout.rollHeight }}
        aria-label="Editable multi-voice piano roll. Drag empty space to select notes; drag the ruler to set the cycle range."
        role="application"
        tabIndex={0}
        onPointerDown={interaction.onPointerDown}
        onPointerMove={interaction.onPointerMove}
        onPointerLeave={interaction.onPointerLeave}
        onPointerUp={interaction.onPointerUp}
        onPointerCancel={interaction.onPointerCancel}
        onDoubleClick={interaction.onDoubleClick}
        onContextMenu={interaction.onContextMenu}
      />
      <div
        className="roll-playhead"
        aria-hidden="true"
        style={{ transform: `translateX(${currentBeat * pxPerBeat}px)` }}
      />
      {interaction.marquee && (
        <div
          className="roll-marquee"
          aria-hidden="true"
          style={{
            left: interaction.marquee.x,
            top: interaction.marquee.y,
            width: interaction.marquee.width,
            height: interaction.marquee.height,
          }}
        />
      )}
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
