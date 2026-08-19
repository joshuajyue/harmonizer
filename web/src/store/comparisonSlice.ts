import type { StateCreator } from "zustand";
import type { ComparisonSlice, StudioStore } from "./types";

export const createComparisonSlice: StateCreator<
  StudioStore,
  [],
  [],
  ComparisonSlice
> = (set) => ({
  engines: [],
  enginesStatus: "idle",
  slots: {
    A: { engineId: "", status: "idle" },
    B: { engineId: "", status: "idle" },
  },
  viewMode: "A",
  activeSlot: "A",

  setEnginesLoading: () =>
    set({ enginesStatus: "loading", enginesError: undefined }),
  setEngines: (engines) =>
    set((state) => {
      const available = engines.filter((engine) => engine.available);
      const rule = available.find((engine) => !engine.learned) ?? available[0];
      const learned =
        available.find((engine) => engine.learned && engine.id !== rule?.id) ??
        available[1] ??
        rule;
      return {
        engines,
        enginesStatus: "ready",
        enginesError: undefined,
        slots: {
          A: {
            ...state.slots.A,
            engineId: state.slots.A.engineId || rule?.id || "",
          },
          B: {
            ...state.slots.B,
            engineId: state.slots.B.engineId || learned?.id || "",
          },
        },
      };
    }),
  setEnginesError: (message) =>
    set({ enginesStatus: "error", enginesError: message }),
  setSlotEngine: (slot, engineId) =>
    set((state) => ({
      slots: {
        ...state.slots,
        [slot]: {
          ...state.slots[slot],
          engineId,
          status: "idle",
          error: undefined,
        },
      },
    })),
  setSlotLoading: (slot) =>
    set((state) => ({
      slots: {
        ...state.slots,
        [slot]: {
          ...state.slots[slot],
          status: "loading",
          error: undefined,
        },
      },
    })),
  setSlotResult: (slot, result, requestRevision) =>
    set((state) => ({
      slots: {
        ...state.slots,
        [slot]: {
          ...state.slots[slot],
          result,
          requestRevision,
          status: "ready",
          error: undefined,
        },
      },
      activeSlot: slot,
      viewMode: state.viewMode === "overlay" ? "overlay" : slot,
    })),
  setSlotError: (slot, message) =>
    set((state) => ({
      slots: {
        ...state.slots,
        [slot]: {
          ...state.slots[slot],
          status: "error",
          error: message,
        },
      },
    })),
  setViewMode: (viewMode) =>
    set((state) => ({
      viewMode,
      activeSlot: viewMode === "overlay" ? state.activeSlot : viewMode,
      selectedNote: undefined,
    })),
  setActiveSlot: (activeSlot) => set({ activeSlot, selectedNote: undefined }),
  updateVoiceNote: (slot, voiceName, index, patch) =>
    set((state) => {
      const result = state.slots[slot].result;
      if (!result) return state;
      return {
        slots: {
          ...state.slots,
          [slot]: {
            ...state.slots[slot],
            result: {
              ...result,
              voices: result.voices.map((voice) =>
                voice.name === voiceName
                  ? {
                      ...voice,
                      notes: voice.notes.map((note, noteIndex) =>
                        noteIndex === index ? { ...note, ...patch } : note,
                      ),
                    }
                  : voice,
              ),
            },
          },
        },
      };
    }),
  deleteVoiceNote: (slot, voiceName, index) =>
    set((state) => {
      const result = state.slots[slot].result;
      if (!result) return state;
      return {
        slots: {
          ...state.slots,
          [slot]: {
            ...state.slots[slot],
            result: {
              ...result,
              voices: result.voices.map((voice) =>
                voice.name === voiceName
                  ? {
                      ...voice,
                      notes: voice.notes.filter(
                        (_note, noteIndex) => noteIndex !== index,
                      ),
                    }
                  : voice,
              ),
            },
          },
        },
        selectedNote: undefined,
      };
    }),
});
