import type { Note } from "../../../contracts/types";
import type { StudioStore } from "./types";
import {
  collectSelectionOrigins,
  isEditableSelection,
  selectionKey,
} from "./selection";

export function buildSelectionDelete(
  state: StudioStore,
): Partial<StudioStore> {
  const selected = new Set(
    state.selectedNotes
      .filter((selection) => isEditableSelection(selection, state.activeSlot))
      .map(selectionKey),
  );
  const melodyChanged = state.selectedNotes.some(
    (selection) =>
      selection.source === "melody" && selected.has(selectionKey(selection)),
  );
  const activeResult = state.slots[state.activeSlot].result;
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
    slots: activeResult
      ? {
          ...state.slots,
          [state.activeSlot]: {
            ...state.slots[state.activeSlot],
            result: {
              ...activeResult,
              voices: activeResult.voices.map((voice) => ({
                ...voice,
                notes: voice.notes.filter(
                  (_note, index) =>
                    !selected.has(
                      selectionKey({
                        source: "voice",
                        slot: state.activeSlot,
                        voice: voice.name,
                        index,
                      }),
                    ),
                ),
              })),
            },
          },
        }
      : state.slots,
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
  const activeResult = state.slots[state.activeSlot].result;
  const update = (note: Note, key: string) =>
    selected.has(key) ? { ...note, duration } : note;

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
    slots: activeResult
      ? {
          ...state.slots,
          [state.activeSlot]: {
            ...state.slots[state.activeSlot],
            result: {
              ...activeResult,
              voices: activeResult.voices.map((voice) => ({
                ...voice,
                notes: voice.notes.map((note, index) =>
                  update(
                    note,
                    selectionKey({
                      source: "voice",
                      slot: state.activeSlot,
                      voice: voice.name,
                      index,
                    }),
                  ),
                ),
              })),
            },
          },
        }
      : state.slots,
  };
}
