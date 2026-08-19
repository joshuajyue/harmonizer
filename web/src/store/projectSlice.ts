import type { Melody } from "../../../contracts/types";
import type { StateCreator } from "zustand";
import { clamp } from "../utils/music";
import type { ProjectSlice, StudioStore } from "./types";

const initialMelody: Melody = {
  notes: [],
  tempo: 88,
  timeSignature: { numerator: 4, denominator: 4 },
};

export const createProjectSlice: StateCreator<
  StudioStore,
  [],
  [],
  ProjectSlice
> = (set) => ({
  projectName: "Untitled melody",
  melody: initialMelody,
  melodyRevision: 0,

  setProjectName: (projectName) => set({ projectName }),
  setTempo: (tempo) =>
    set((state) =>
      Number.isFinite(tempo)
        ? {
            melody: {
              ...state.melody,
              tempo: Math.round(clamp(tempo, 30, 240)),
            },
            melodyRevision: state.melodyRevision + 1,
          }
        : state,
    ),
  setTimeSignature: (timeSignature) =>
    set((state) => ({
      melody: { ...state.melody, timeSignature },
      melodyRevision: state.melodyRevision + 1,
    })),
  setKey: (key) =>
    set((state) => ({
      melody: { ...state.melody, key },
      melodyRevision: state.melodyRevision + 1,
    })),
  replaceMelody: (melody, projectName) =>
    set((state) => ({
      melody: {
        ...melody,
        timeSignature: {
          ...(melody.timeSignature ?? { numerator: 4, denominator: 4 }),
        },
        key: melody.key ? { ...melody.key } : undefined,
        notes: melody.notes.map((note) => ({ ...note })),
      },
      projectName: projectName ?? state.projectName,
      melodyRevision: state.melodyRevision + 1,
      currentBeat: 0,
      isPlaying: false,
      selectedNotes: [],
      loopRangeCustomized: false,
    })),
  addMelodyNote: (note) => {
    let index = 0;
    set((state) => {
      index = state.melody.notes.length;
      return {
        melody: {
          ...state.melody,
          notes: [...state.melody.notes, { ...note }],
        },
        melodyRevision: state.melodyRevision + 1,
      };
    });
    return index;
  },
  updateMelodyNote: (index, patch) =>
    set((state) => ({
      melody: {
        ...state.melody,
        notes: state.melody.notes.map((note, noteIndex) =>
          noteIndex === index ? { ...note, ...patch } : note,
        ),
      },
      melodyRevision: state.melodyRevision + 1,
    })),
  deleteMelodyNote: (index) =>
    set((state) => ({
      melody: {
        ...state.melody,
        notes: state.melody.notes.filter(
          (_note, noteIndex) => noteIndex !== index,
        ),
      },
      melodyRevision: state.melodyRevision + 1,
      selectedNotes: [],
    })),
  clearMelody: () =>
    set((state) => ({
      melody: { ...state.melody, notes: [] },
      melodyRevision: state.melodyRevision + 1,
      currentBeat: 0,
      isPlaying: false,
      selectedNotes: [],
      loopRangeCustomized: false,
    })),
});
