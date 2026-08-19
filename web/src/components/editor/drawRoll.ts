import type {
  HarmonizeResponse,
  Note,
  VoiceName,
} from "../../../../contracts/types";
import { midiToName, VOICE_COLORS } from "../../utils/music";
import { drawChords, drawViolations } from "./drawAnnotations";
import { drawGrid } from "./drawGrid";
import {
  laneCenter,
  pitchToY,
  ROLL_HEIGHT,
  RULER_HEIGHT,
} from "./rollGeometry";
import type {
  DrawModel,
  DrawResult,
  NoteHit,
  ViolationHit,
} from "./rollTypes";

function roundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius = 4,
) {
  context.beginPath();
  context.roundRect(x, y, width, height, radius);
}

function isSelected(hit: NoteHit, model: DrawModel) {
  const selected = model.selectedNote;
  return (
    selected?.source === hit.source &&
    selected.index === hit.index &&
    selected.slot === hit.slot &&
    selected.voice === hit.voice
  );
}

function drawNote(
  context: CanvasRenderingContext2D,
  hit: NoteHit,
  color: string,
  model: DrawModel,
  style: "solid" | "outline" | "melody",
) {
  const { rect } = hit;
  roundedRect(context, rect.x, rect.y, rect.width, rect.height);
  if (style === "outline") {
    context.setLineDash([5, 3]);
    context.strokeStyle = color;
    context.lineWidth = 1.7;
    context.stroke();
    context.setLineDash([]);
  } else {
    context.fillStyle = style === "melody" ? "#e9edf5" : color;
    context.globalAlpha = style === "melody" ? 0.92 : 0.78;
    context.fill();
    context.globalAlpha = 1;
    context.strokeStyle =
      style === "melody" ? "#ffffff" : "rgba(255,255,255,.28)";
    context.lineWidth = 1;
    context.stroke();
  }
  if (isSelected(hit, model)) {
    roundedRect(
      context,
      rect.x - 2,
      rect.y - 2,
      rect.width + 4,
      rect.height + 4,
      5,
    );
    context.strokeStyle = "#c7ff5e";
    context.lineWidth = 2;
    context.stroke();
  }
  if (rect.width > 34 && style !== "outline") {
    context.fillStyle = style === "melody" ? "#151922" : "#11151b";
    context.font = '500 9px "DM Mono", monospace';
    context.fillText(
      midiToName(hit.note.pitch),
      rect.x + 5,
      rect.y + rect.height - 3,
    );
  }
  if (hit.editable && isSelected(hit, model)) {
    context.fillStyle = "#c7ff5e";
    context.fillRect(rect.x + rect.width - 3, rect.y + 2, 2, rect.height - 4);
  }
}

function collectVoiceNotes(
  result: HarmonizeResponse | undefined,
  slot: "A" | "B",
  model: DrawModel,
  hits: NoteHit[],
  context: CanvasRenderingContext2D,
  style: "solid" | "outline",
) {
  if (!result) return;
  for (const voice of result.voices) {
    if (!model.voiceVisibility[voice.name]) continue;
    voice.notes.forEach((note, index) => {
      const rect = noteRect(note, voice.name, model.pxPerBeat, style === "outline");
      const hit: NoteHit = {
        rect,
        note,
        source: "voice",
        index,
        slot,
        voice: voice.name,
        editable: slot === model.activeSlot,
      };
      hits.push(hit);
      drawNote(context, hit, VOICE_COLORS[voice.name], model, style);
    });
  }
}

function noteRect(
  note: Note,
  lane: "melody" | VoiceName,
  pxPerBeat: number,
  offset = false,
) {
  const height = lane === "melody" ? 11 : 10;
  return {
    x: note.start * pxPerBeat + 2 + (offset ? 2 : 0),
    y: pitchToY(lane, note.pitch) - height / 2 + (offset ? 2 : 0),
    width: Math.max(5, note.duration * pxPerBeat - 4),
    height,
  };
}

export function drawRoll(
  context: CanvasRenderingContext2D,
  model: DrawModel,
): DrawResult {
  const noteHits: NoteHit[] = [];
  const violationHits: ViolationHit[] = [];
  context.clearRect(0, 0, model.duration * model.pxPerBeat, ROLL_HEIGHT);
  drawGrid(context, model);
  if (model.melody.notes.length === 0) {
    context.fillStyle = "#687283";
    context.font = '10px "DM Mono", monospace';
    context.fillText(
      "Double-click to add a melody note, or choose an input below",
      18,
      laneCenter("melody") + 3,
    );
  }

  model.melody.notes.forEach((note, index) => {
    const hit: NoteHit = {
      rect: noteRect(note, "melody", model.pxPerBeat),
      note,
      source: "melody",
      index,
      editable: true,
    };
    noteHits.push(hit);
    drawNote(context, hit, "#ffffff", model, "melody");
  });

  if (model.viewMode === "overlay") {
    collectVoiceNotes(
      model.resultA,
      "A",
      model,
      noteHits,
      context,
      model.activeSlot === "A" ? "solid" : "outline",
    );
    collectVoiceNotes(
      model.resultB,
      "B",
      model,
      noteHits,
      context,
      model.activeSlot === "B" ? "solid" : "outline",
    );
    drawViolations(context, model.resultA, "A", model, violationHits);
    drawViolations(context, model.resultB, "B", model, violationHits);
  } else {
    const slot = model.viewMode;
    const result = slot === "A" ? model.resultA : model.resultB;
    collectVoiceNotes(
      result,
      slot,
      model,
      noteHits,
      context,
      "solid",
    );
    drawViolations(context, result, slot, model, violationHits);
  }

  const chordResult =
    model.activeSlot === "A" ? model.resultA : model.resultB;
  if (!chordResult) {
    context.fillStyle = "#596273";
    context.font = '10px "DM Mono", monospace';
    context.fillText(
      `Harmonize engine ${model.activeSlot} to reveal independent SATB parts`,
      18,
      laneCenter("alto") + 3,
    );
  }
  drawChords(context, chordResult?.chords ?? [], model);

  if (model.loopEnabled) {
    const x = model.loopStart * model.pxPerBeat;
    const width = (model.loopEnd - model.loopStart) * model.pxPerBeat;
    context.fillStyle = "rgba(199,255,94,.035)";
    context.fillRect(x, RULER_HEIGHT, width, ROLL_HEIGHT - RULER_HEIGHT);
    context.strokeStyle = "rgba(199,255,94,.46)";
    context.strokeRect(x + 0.5, RULER_HEIGHT, width - 1, ROLL_HEIGHT - RULER_HEIGHT);
  }

  return { noteHits, violationHits };
}
