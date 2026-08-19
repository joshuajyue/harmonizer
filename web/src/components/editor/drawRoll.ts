import type {
  HarmonizeResponse,
  Note,
  VoiceName,
} from "../../../../contracts/types";
import { midiToName, VOICE_COLORS } from "../../utils/music";
import { selectionKey } from "../../store/selection";
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

function hitKey(hit: NoteHit) {
  return selectionKey({
    source: hit.source,
    index: hit.index,
    slot: hit.slot,
    voice: hit.voice,
  });
}

function drawNote(
  context: CanvasRenderingContext2D,
  hit: NoteHit,
  color: string,
  model: DrawModel,
  style: "solid" | "outline" | "melody",
  selectedKeys: Set<string>,
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
  const selected = selectedKeys.has(hitKey(hit));
  if (selected) {
    if (model.selectedNotes.length > 1) {
      roundedRect(
        context,
        rect.x - 1,
        rect.y - 1,
        rect.width + 2,
        rect.height + 2,
        5,
      );
      context.fillStyle = "rgba(97, 184, 255, .18)";
      context.fill();
    }
    roundedRect(
      context,
      rect.x - 2,
      rect.y - 2,
      rect.width + 4,
      rect.height + 4,
      5,
    );
    context.strokeStyle =
      model.selectedNotes.length > 1 ? "#61b8ff" : "#c7ff5e";
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
  if (hit.editable && selected && model.selectedNotes.length === 1) {
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
  selectedKeys: Set<string>,
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
      drawNote(
        context,
        hit,
        VOICE_COLORS[voice.name],
        model,
        style,
        selectedKeys,
      );
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
  const selectedKeys = new Set(model.selectedNotes.map(selectionKey));
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
    drawNote(context, hit, "#ffffff", model, "melody", selectedKeys);
  });

  if (model.viewMode === "overlay") {
    collectVoiceNotes(
      model.resultA,
      "A",
      model,
      noteHits,
      context,
      model.activeSlot === "A" ? "solid" : "outline",
      selectedKeys,
    );
    collectVoiceNotes(
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
    collectVoiceNotes(
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
    context.fillRect(x, 0, width, ROLL_HEIGHT);
    context.fillStyle = "rgba(199,255,94,.12)";
    context.fillRect(x, 0, width, RULER_HEIGHT);
    context.strokeStyle = "rgba(199,255,94,.46)";
    context.strokeRect(x + 0.5, 0.5, width - 1, ROLL_HEIGHT - 1);
  }

  return { noteHits, violationHits };
}
