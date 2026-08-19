import type { StateCreator } from "zustand";
import { clamp } from "../utils/music";
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
  pxPerBeat: 64,
  snap: 0.25,
  voiceVisibility: { ...allEnabled },
  voiceMute: { ...allDisabled },
  voiceSolo: { ...allDisabled },
  inputTab: "piano",
  inputDockOpen: true,

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
  setSelectedNote: (selectedNote) => set({ selectedNote }),
  setInputTab: (inputTab) => set({ inputTab }),
  setInputDockOpen: (inputDockOpen) => set({ inputDockOpen }),
});
