import type { Melody } from "../../../contracts/types";
import type { StateCreator } from "zustand";
import { DEMO_MELODY } from "../fixtures/chorale";
import { clamp } from "../utils/music";
import type { ProjectSlice, StudioStore } from "./types";

const initialMelody: Melody = {
  ...DEMO_MELODY,
  timeSignature: {
    ...(DEMO_MELODY.timeSignature ?? { numerator: 4, denominator: 4 }),
  },
  key: DEMO_MELODY.key ? { ...DEMO_MELODY.key } : undefined,
  notes: DEMO_MELODY.notes.map((note) => ({ ...note })),
};

export const createProjectSlice: StateCreator<
  StudioStore,
  [],
  [],
  ProjectSlice
> = (set) => ({
  projectName: "Chorale study in C",
  melody: initialMelody,
  melodyRevision: 0,

  setProjectName: (projectName) => set({ projectName }),
  setTempo: (tempo) =>
    set((state) => ({
      melody: { ...state.melody, tempo: Math.round(clamp(tempo, 30, 240)) },
      melodyRevision: state.melodyRevision + 1,
    })),
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
      selectedNote: undefined,
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
      selectedNote: undefined,
    })),
  clearMelody: () =>
    set((state) => ({
      melody: { ...state.melody, notes: [] },
      melodyRevision: state.melodyRevision + 1,
      currentBeat: 0,
      isPlaying: false,
      selectedNote: undefined,
    })),
});
