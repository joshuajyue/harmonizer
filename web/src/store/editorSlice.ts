import type { StateCreator } from "zustand";
import { clamp } from "../utils/music";
import {
  buildSelectionTransform,
  collectSelectionOrigins,
  dedupeSelection,
} from "./selection";
import {
  buildSelectionDelete,
  buildSelectionDuration,
} from "./selectionEdits";
import type { EditorSlice, StudioStore } from "./types";

const allEnabled = {
  soprano: true,
  alto: true,
  tenor: true,
  bass: true,
} as const;

const allDisabled = {
  soprano: false,
  alto: false,
  tenor: false,
  bass: false,
} as const;

export const createEditorSlice: StateCreator<
  StudioStore,
  [],
  [],
  EditorSlice
> = (set) => ({
  pxPerBeat: 32,
  snap: 0.25,
  voiceVisibility: { ...allEnabled },
  voiceMute: { ...allDisabled },
  voiceSolo: { ...allDisabled },
  selectedNotes: [],
  inputTab: "piano",
  inputDockOpen:
    typeof window === "undefined" ? true : window.innerHeight >= 800,

  setZoom: (pxPerBeat) => set({ pxPerBeat: clamp(pxPerBeat, 28, 144) }),
  setSnap: (snap) => set({ snap }),
  toggleVoiceVisibility: (voice) =>
    set((state) => ({
      voiceVisibility: {
        ...state.voiceVisibility,
        [voice]: !state.voiceVisibility[voice],
      },
    })),
  toggleVoiceMute: (voice) =>
    set((state) => ({
      voiceMute: {
        ...state.voiceMute,
        [voice]: !state.voiceMute[voice],
      },
    })),
  toggleVoiceSolo: (voice) =>
    set((state) => ({
      voiceSolo: {
        ...state.voiceSolo,
        [voice]: !state.voiceSolo[voice],
      },
    })),
  setSelectedNotes: (selectedNotes) =>
    set({ selectedNotes: dedupeSelection(selectedNotes) }),
  clearSelection: () => set({ selectedNotes: [] }),
  deleteSelectedNotes: () =>
    set((state) => buildSelectionDelete(state)),
  transformSelectedNotes: (origins, deltaBeats, deltaPitch) =>
    set((state) =>
      buildSelectionTransform(state, origins, deltaBeats, deltaPitch),
    ),
  nudgeSelectedNotes: (deltaBeats) =>
    set((state) =>
      buildSelectionTransform(
        state,
        collectSelectionOrigins(state),
        deltaBeats,
        0,
      ),
    ),
  transposeSelectedNotes: (semitones) =>
    set((state) =>
      buildSelectionTransform(
        state,
        collectSelectionOrigins(state),
        0,
        semitones,
      ),
    ),
  setSelectedNotesDuration: (duration) =>
    set((state) =>
      buildSelectionDuration(state, Math.max(state.snap, duration)),
    ),
  setInputTab: (inputTab) => set({ inputTab }),
  setInputDockOpen: (inputDockOpen) => set({ inputDockOpen }),
});
