import type {
  Chord,
  HarmonizeResponse,
} from "../../../../contracts/types";
import {
  CHORD_HEIGHT,
  LANES,
  LANE_HEIGHT,
  laneCenter,
  RULER_HEIGHT,
} from "./rollGeometry";
import type {
  DrawModel,
  ViolationHit,
} from "./rollTypes";

export function drawChords(
  context: CanvasRenderingContext2D,
  chords: Chord[],
  model: DrawModel,
) {
  const top = RULER_HEIGHT + LANES.length * LANE_HEIGHT;
  context.fillStyle = "#0d1117";
  context.fillRect(0, top, model.duration * model.pxPerBeat, CHORD_HEIGHT);
  context.fillStyle = "#5d6675";
  context.font = '9px "DM Mono", monospace';
  context.fillText("HARMONIC ANALYSIS", 10, top + 14);
  for (const chord of chords) {
    const x = chord.start * model.pxPerBeat;
    const width = chord.duration * model.pxPerBeat;
    context.fillStyle = "rgba(199, 255, 94, .075)";
    context.beginPath();
    context.roundRect(x + 2, top + 21, Math.max(8, width - 4), 24, 5);
    context.fill();
    context.fillStyle = "#dfffa3";
    context.font = '500 12px "DM Mono", monospace';
    context.textAlign = "center";
    context.fillText(
      chord.roman,
      x + width / 2,
      top + 37,
      Math.max(6, width - 10),
    );
    context.textAlign = "start";
  }
}

export function drawViolations(
  context: CanvasRenderingContext2D,
  result: HarmonizeResponse | undefined,
  slot: "A" | "B",
  model: DrawModel,
  hits: ViolationHit[],
) {
  for (const violation of result?.violations ?? []) {
    const x = violation.start * model.pxPerBeat;
    const centers = violation.voices.map((voice) => laneCenter(voice));
    const top = Math.min(...centers);
    const bottom = Math.max(...centers);
    const color =
      violation.severity === "error"
        ? "#ff5e72"
        : violation.severity === "warning"
          ? "#ffbd59"
          : "#61b8ff";
    context.strokeStyle = color;
    context.fillStyle = color;
    context.lineWidth = slot === "A" ? 1.5 : 1;
    context.setLineDash(slot === "B" ? [3, 3] : []);
    context.beginPath();
    context.moveTo(x, top);
    context.lineTo(x, bottom);
    context.stroke();
    context.setLineDash([]);
    context.save();
    context.translate(x, (top + bottom) / 2);
    context.rotate(Math.PI / 4);
    context.fillRect(-4.5, -4.5, 9, 9);
    context.restore();
    context.fillStyle = "#0c1016";
    context.font = '700 7px "DM Mono", monospace';
    context.textAlign = "center";
    context.fillText(slot, x, (top + bottom) / 2 + 2.5);
    context.textAlign = "start";
    hits.push({
      rect: { x: x - 9, y: top - 8, width: 18, height: bottom - top + 16 },
      violation,
      slot,
    });
  }
}
