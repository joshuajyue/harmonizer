import type { Note, VoiceName } from "../../../contracts/types";
import { clamp, VOICE_RANGES } from "../utils/music";
import type {
  ComparisonView,
  SelectedNote,
  SelectionOrigin,
  StudioStore,
} from "./types";

export function selectionKey(selection: SelectedNote) {
  return [
    selection.source,
    selection.slot ?? "",
    selection.voice ?? "",
    selection.index,
  ].join(":");
}

export function dedupeSelection(selections: SelectedNote[]) {
  const unique = new Map<string, SelectedNote>();
  for (const selection of selections) {
    unique.set(selectionKey(selection), selection);
  }
  return [...unique.values()];
}

export function resolveSelectedNote(
  state: StudioStore,
  selection: SelectedNote,
): Note | undefined {
  if (selection.source === "melody") {
    return state.melody.notes[selection.index];
  }
  if (!selection.slot || !selection.voice) return undefined;
  return state.slots[selection.slot].result?.voices
    .find((voice) => voice.name === selection.voice)
    ?.notes.at(selection.index);
}

export function collectSelectionOrigins(state: StudioStore) {
  return state.selectedNotes.flatMap<SelectionOrigin>((selection) => {
    const note = resolveSelectedNote(state, selection);
    return note && isEditableSelection(selection, state.viewMode)
      ? [{ selection, note: { ...note } }]
      : [];
  });
}

export function isEditableSelection(
  selection: SelectedNote,
  viewMode: ComparisonView,
) {
  return (
    selection.source === "melody" ||
    (selection.slot !== undefined &&
      selection.voice !== undefined &&
      selection.slot === viewMode)
  );
}

export function applyVoiceNoteEdits(
  state: StudioStore,
  edits: ReadonlyMap<string, Note | null>,
) {
  let slots = state.slots;
  for (const slot of ["A", "B"] as const) {
    const result = slots[slot].result;
    if (!result) continue;
    let resultChanged = false;
    const voices = result.voices.map((voice) => {
      let voiceChanged = false;
      const notes = voice.notes.flatMap((note, index) => {
        const key = selectionKey({
          source: "voice",
          slot,
          voice: voice.name,
          index,
        });
        if (!edits.has(key)) return [note];
        voiceChanged = true;
        const replacement = edits.get(key);
        return replacement ? [replacement] : [];
      });
      if (!voiceChanged) return voice;
      resultChanged = true;
      return { ...voice, notes };
    });
    if (!resultChanged) continue;
    slots = {
      ...slots,
      [slot]: {
        ...slots[slot],
        result: { ...result, voices },
      },
    };
  }
  return slots;
}

function selectionRange(selection: SelectedNote) {
  return selection.source === "melody"
    ? { min: 0, max: 127 }
    : VOICE_RANGES[selection.voice as VoiceName];
}

export function buildSelectionTransform(
  state: StudioStore,
  origins: SelectionOrigin[],
  requestedBeatDelta: number,
  requestedPitchDelta: number,
): Partial<StudioStore> {
  const editable = origins.filter(({ selection }) =>
    isEditableSelection(selection, state.viewMode),
  );
  if (editable.length === 0) return {};

  const earliestStart = Math.min(...editable.map(({ note }) => note.start));
  const beatDelta = Math.max(-earliestStart, requestedBeatDelta);
  let minimumPitchDelta = -127;
  let maximumPitchDelta = 127;
  for (const { selection, note } of editable) {
    const range = selectionRange(selection);
    minimumPitchDelta = Math.max(
      minimumPitchDelta,
      range.min - note.pitch,
    );
    maximumPitchDelta = Math.min(
      maximumPitchDelta,
      range.max - note.pitch,
    );
  }
  const pitchDelta = clamp(
    Math.round(requestedPitchDelta),
    minimumPitchDelta,
    maximumPitchDelta,
  );
  const transformed = new Map(
    editable.map(({ selection, note }) => [
      selectionKey(selection),
      {
        ...note,
        start: Math.max(0, note.start + beatDelta),
        pitch: note.pitch + pitchDelta,
      },
    ]),
  );
  const melodyChanged = editable.some(
    ({ selection }) => selection.source === "melody",
  );

  return {
    melody: melodyChanged
      ? {
          ...state.melody,
          notes: state.melody.notes.map(
            (note, index) =>
              transformed.get(
                selectionKey({ source: "melody", index }),
              ) ?? note,
          ),
        }
      : state.melody,
    melodyRevision: melodyChanged
      ? state.melodyRevision + 1
      : state.melodyRevision,
    slots: applyVoiceNoteEdits(state, transformed),
  };
}
