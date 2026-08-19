import type { Note } from "../../../contracts/types";
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
