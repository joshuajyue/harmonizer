import type {
  HarmonizeResponse,
  Note,
  VoiceName,
} from "../../../../contracts/types";
import { selectionKey } from "../../store/selection";
import { midiToName, VOICE_COLORS } from "../../utils/music";
import { pitchStep, pitchToY } from "./rollGeometry";
import type { DrawModel, NoteHit } from "./rollTypes";

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

function noteRect(
  note: Note,
  lane: "melody" | VoiceName,
  model: DrawModel,
  offset = false,
) {
  const height = model.layout.focusedLane
    ? Math.max(11, pitchStep(lane, model.layout) - 2)
    : lane === "melody"
      ? 11
      : 10;
  return {
    x: note.start * model.pxPerBeat + 2 + (offset ? 2 : 0),
    y:
      pitchToY(lane, note.pitch, model.layout) -
      height / 2 +
      (offset ? 2 : 0),
    width: Math.max(5, note.duration * model.pxPerBeat - 4),
    height,
  };
}

export function drawMelodyNotes(
  context: CanvasRenderingContext2D,
  model: DrawModel,
  hits: NoteHit[],
  selectedKeys: Set<string>,
) {
  model.melody.notes.forEach((note, index) => {
    const hit: NoteHit = {
      rect: noteRect(note, "melody", model),
      note,
      source: "melody",
      index,
      editable: true,
    };
    hits.push(hit);
    drawNote(context, hit, "#ffffff", model, "melody", selectedKeys);
  });
}

export function drawVoiceNotes(
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
    if (
      model.layout.focusedLane
        ? model.layout.focusedLane !== voice.name
        : !model.voiceVisibility[voice.name]
    ) {
      continue;
    }
    voice.notes.forEach((note, index) => {
      const hit: NoteHit = {
        rect: noteRect(note, voice.name, model, style === "outline"),
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
