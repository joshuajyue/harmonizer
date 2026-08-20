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
      const primary =
        available.find((engine) => engine.id === "rules") ??
        available.find(
          (engine) => !engine.learned && engine.id !== "dev-stub",
        ) ??
        available[0];
      const comparison =
        available.find((engine) => engine.id === "fixed_thirds") ??
        available.find(
          (engine) => engine.learned && engine.id !== primary?.id,
        ) ??
        available.find((engine) => engine.id !== primary?.id) ??
        primary;
      return {
        engines,
        enginesStatus: "ready",
        enginesError: undefined,
        slots: {
          A: {
            ...state.slots.A,
            engineId: state.slots.A.engineId || primary?.id || "",
          },
          B: {
            ...state.slots.B,
            engineId: state.slots.B.engineId || comparison?.id || "",
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
          result: undefined,
          requestRevision: undefined,
          status: "idle",
          error: undefined,
        },
      },
      selectedNotes: [],
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
      viewMode: slot,
      selectedNotes: [],
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
    set({
      viewMode,
      activeSlot: viewMode,
      selectedNotes: [],
    }),
  setActiveSlot: (activeSlot) =>
    set({ activeSlot, viewMode: activeSlot, selectedNotes: [] }),
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
        selectedNotes: [],
      };
    }),
});
