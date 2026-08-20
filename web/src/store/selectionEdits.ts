import type { Note } from "../../../contracts/types";
import { quantize } from "../utils/music";
import type { StudioStore } from "./types";
import {
  applyVoiceNoteEdits,
  collectSelectionOrigins,
  selectionKey,
} from "./selection";

export function buildSelectionDelete(
  state: StudioStore,
): Partial<StudioStore> {
  const origins = collectSelectionOrigins(state);
  const selected = new Set(
    origins.map(({ selection }) => selectionKey(selection)),
  );
  const voiceEdits = new Map(
    origins
      .filter(({ selection }) => selection.source === "voice")
      .map(({ selection }) => [selectionKey(selection), null] as const),
  );
  const melodyChanged = origins.some(
    ({ selection }) => selection.source === "melody",
  );
  const melodyNotes = melodyChanged
    ? state.melody.notes.filter(
        (_note, index) =>
          !selected.has(selectionKey({ source: "melody", index })),
      )
    : state.melody.notes;

  return {
    melody: melodyChanged
      ? {
          ...state.melody,
          notes: melodyNotes,
        }
      : state.melody,
    melodyRevision: melodyChanged
      ? state.melodyRevision + 1
      : state.melodyRevision,
    slots: applyVoiceNoteEdits(state, voiceEdits),
    transcriptionRegister:
      melodyChanged && melodyNotes.length === 0
        ? undefined
        : state.transcriptionRegister,
    selectedNotes: [],
  };
}

export function buildSelectionDuration(
  state: StudioStore,
  duration: number,
): Partial<StudioStore> {
  const origins = collectSelectionOrigins(state);
  const selected = new Set(
    origins.map(({ selection }) => selectionKey(selection)),
  );
  const melodyChanged = origins.some(
    ({ selection }) => selection.source === "melody",
  );
  const update = (note: Note, key: string) =>
    selected.has(key) ? { ...note, duration } : note;
  const voiceEdits = new Map(
    origins
      .filter(({ selection }) => selection.source === "voice")
      .map(({ selection, note }) => [
        selectionKey(selection),
        { ...note, duration },
      ]),
  );

  return {
    melody: melodyChanged
      ? {
          ...state.melody,
          notes: state.melody.notes.map((note, index) =>
            update(note, selectionKey({ source: "melody", index })),
          ),
        }
      : state.melody,
    melodyRevision: melodyChanged
      ? state.melodyRevision + 1
      : state.melodyRevision,
    slots: applyVoiceNoteEdits(state, voiceEdits),
  };
}

export function buildSelectionQuantize(
  state: StudioStore,
): Partial<StudioStore> {
  const origins = collectSelectionOrigins(state);
  const edits = new Map(
    origins.map(({ selection, note }) => [
      selectionKey(selection),
      {
        ...note,
        start: Math.max(0, quantize(note.start, state.snap)),
      },
    ]),
  );
  const changed = origins.filter(({ selection, note }) => {
    const next = edits.get(selectionKey(selection));
    return next && Math.abs(next.start - note.start) > 0.0001;
  });
  if (changed.length === 0) return {};
  const melodyChanged = changed.some(
    ({ selection }) => selection.source === "melody",
  );

  return {
    melody: melodyChanged
      ? {
          ...state.melody,
          notes: state.melody.notes.map(
            (note, index) =>
              edits.get(selectionKey({ source: "melody", index })) ?? note,
          ),
        }
      : state.melody,
    melodyRevision: melodyChanged
      ? state.melodyRevision + 1
      : state.melodyRevision,
    slots: applyVoiceNoteEdits(state, edits),
  };
}
