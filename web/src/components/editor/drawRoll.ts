import { selectionKey } from "../../store/selection";
import { drawChords, drawViolations } from "./drawAnnotations";
import { drawGrid } from "./drawGrid";
import { drawMelodyNotes, drawVoiceNotes } from "./drawNotes";
import { RULER_HEIGHT } from "./rollGeometry";
import type {
  DrawModel,
  DrawResult,
  NoteHit,
  ViolationHit,
} from "./rollTypes";

export function drawRoll(
  context: CanvasRenderingContext2D,
  model: DrawModel,
): DrawResult {
  const noteHits: NoteHit[] = [];
  const violationHits: ViolationHit[] = [];
  const selectedKeys = new Set(model.selectedNotes.map(selectionKey));
  context.clearRect(
    0,
    0,
    model.duration * model.pxPerBeat,
    model.layout.rollHeight,
  );
  drawGrid(context, model);
  const melodyVisible =
    model.layout.focusedLane === undefined ||
    model.layout.focusedLane === "melody";

  if (melodyVisible) {
    drawMelodyNotes(context, model, noteHits, selectedKeys);
  }

  if (model.viewMode === "overlay") {
    drawVoiceNotes(
      model.resultA,
      "A",
      model,
      noteHits,
      context,
      model.activeSlot === "A" ? "solid" : "outline",
      selectedKeys,
    );
    drawVoiceNotes(
      model.resultB,
      "B",
      model,
      noteHits,
      context,
      model.activeSlot === "B" ? "solid" : "outline",
      selectedKeys,
    );
    drawViolations(context, model.resultA, "A", model, violationHits);
    drawViolations(context, model.resultB, "B", model, violationHits);
  } else {
    const slot = model.viewMode;
    const result = slot === "A" ? model.resultA : model.resultB;
    drawVoiceNotes(
      result,
      slot,
      model,
      noteHits,
      context,
      "solid",
      selectedKeys,
    );
    drawViolations(context, result, slot, model, violationHits);
  }

  const chordResult =
    model.activeSlot === "A" ? model.resultA : model.resultB;
  drawChords(context, chordResult?.chords ?? [], model);

  if (model.loopEnabled) {
    const x = model.loopStart * model.pxPerBeat;
    const width = (model.loopEnd - model.loopStart) * model.pxPerBeat;
    context.fillStyle = "rgba(199,255,94,.035)";
    context.fillRect(x, 0, width, model.layout.rollHeight);
    context.fillStyle = "rgba(199,255,94,.12)";
    context.fillRect(x, 0, width, RULER_HEIGHT);
    context.strokeStyle = "rgba(199,255,94,.46)";
    context.strokeRect(
      x + 0.5,
      0.5,
      width - 1,
      model.layout.rollHeight - 1,
    );
  }

  return { noteHits, violationHits };
}
