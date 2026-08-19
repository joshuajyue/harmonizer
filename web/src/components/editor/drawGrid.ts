import {
  laneTop,
  pitchRange,
  pitchRowRect,
  pitchToY,
  RULER_HEIGHT,
} from "./rollGeometry";
import type { DrawModel } from "./rollTypes";

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

export function drawGrid(
  context: CanvasRenderingContext2D,
  model: DrawModel,
) {
  const signature = model.melody.timeSignature ?? {
    numerator: 4,
    denominator: 4,
  };
  const barLength = signature.numerator * (4 / signature.denominator);
  context.fillStyle = "#10141b";
  context.fillRect(
    0,
    0,
    model.duration * model.pxPerBeat,
    model.layout.rollHeight,
  );
  if (model.empty) return;

  model.layout.lanes.forEach((lane, laneIndex) => {
    const top = laneTop(lane, model.layout);
    context.fillStyle = laneIndex % 2 === 0 ? "#11161e" : "#0f141b";
    context.fillRect(
      0,
      top,
      model.duration * model.pxPerBeat,
      model.layout.laneHeight,
    );
    const range = pitchRange(lane, model.layout);
    for (let pitch = range.min; pitch <= range.max; pitch += 1) {
      if (model.layout.focusedLane) {
        const row = pitchRowRect(lane, pitch, model.layout);
        if (BLACK_KEYS.has(pitch % 12)) {
          context.fillStyle = "rgba(0, 0, 0, 0.21)";
          context.fillRect(
            0,
            row.y,
            model.duration * model.pxPerBeat,
            row.height,
          );
        }
        context.strokeStyle =
          pitch % 12 === 0 ? "#384252" : "rgba(54, 63, 77, .42)";
        context.lineWidth = pitch % 12 === 0 ? 1.2 : 1;
        context.beginPath();
        context.moveTo(0, row.y + 0.5);
        context.lineTo(model.duration * model.pxPerBeat, row.y + 0.5);
        context.stroke();
      } else if (BLACK_KEYS.has(pitch % 12)) {
        const y = pitchToY(lane, pitch, model.layout);
        context.fillStyle = "rgba(0, 0, 0, 0.13)";
        context.fillRect(0, y - 2, model.duration * model.pxPerBeat, 4);
      }
    }
  });

  const smallestDivision = model.pxPerBeat >= 80 ? 0.25 : 0.5;
  for (let beat = 0; beat <= model.duration; beat += smallestDivision) {
    const x = beat * model.pxPerBeat;
    const isBar = Math.abs(beat % barLength) < 0.001;
    const isBeat = Math.abs(beat % 1) < 0.001;
    context.strokeStyle = isBar
      ? "#3a4252"
      : isBeat
        ? "#29313e"
        : "#1d2430";
    context.lineWidth = isBar ? 1.4 : 1;
    context.beginPath();
    context.moveTo(x + 0.5, RULER_HEIGHT);
    context.lineTo(x + 0.5, model.layout.rollHeight);
    context.stroke();
    if (isBeat && beat < model.duration - 0.001) {
      context.fillStyle = isBar ? "#c2c8d3" : "#737d8e";
      context.font = '10px "DM Mono", monospace';
      context.fillText(
        isBar
          ? `${Math.floor(beat / barLength) + 1}`
          : `${(beat % barLength) + 1}`,
        x + 6,
        19,
      );
    }
  }

  for (let index = 0; index <= model.layout.lanes.length; index += 1) {
    const y = RULER_HEIGHT + index * model.layout.laneHeight;
    context.strokeStyle = "#252c38";
    context.beginPath();
    context.moveTo(0, y + 0.5);
    context.lineTo(model.duration * model.pxPerBeat, y + 0.5);
    context.stroke();
  }
}
