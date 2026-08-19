import type { Melody } from "../../../contracts/types";
import type { StateCreator } from "zustand";
import { clamp } from "../utils/music";
import type { ProjectSlice, StudioStore } from "./types";

const initialMelody: Melody = {
  notes: [],
  tempo: 88,
  timeSignature: { numerator: 4, denominator: 4 },
};

function copyMelody(melody: Melody): Melody {
  return {
    ...melody,
    timeSignature: {
      ...(melody.timeSignature ?? { numerator: 4, denominator: 4 }),
    },
    key: melody.key ? { ...melody.key } : undefined,
    notes: melody.notes.map((note) => ({ ...note })),
  };
}

function shiftedMelody(melody: Melody, octaves: number) {
  const wholeOctaves = Math.trunc(octaves);
  const semitones = wholeOctaves * 12;
  if (
    wholeOctaves === 0 ||
    melody.notes.some(
      (note) => note.pitch + semitones < 0 || note.pitch + semitones > 127,
    )
  ) {
    return undefined;
  }
  return {
    ...melody,
    notes: melody.notes.map((note) => ({
      ...note,
      pitch: note.pitch + semitones,
    })),
  };
}

export const createProjectSlice: StateCreator<
  StudioStore,
  [],
  [],
  ProjectSlice
> = (set) => ({
  projectName: "Untitled melody",
  melody: initialMelody,
  melodyRevision: 0,
  transcriptionRegister: undefined,

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
      melody: copyMelody(melody),
      projectName: projectName ?? state.projectName,
      melodyRevision: state.melodyRevision + 1,
      transcriptionRegister: undefined,
      currentBeat: 0,
      isPlaying: false,
      selectedNotes: [],
      loopRangeCustomized: false,
    })),
  replaceTranscribedMelody: (melody, register, projectName) =>
    set((state) => ({
      melody: copyMelody(melody),
      projectName: projectName ?? state.projectName,
      melodyRevision: state.melodyRevision + 1,
      transcriptionRegister: {
        ...register,
        currentOctaveShift: register.detectedOctaveShift,
      },
      currentBeat: 0,
      isPlaying: false,
      selectedNotes: [],
      loopRangeCustomized: false,
    })),
  shiftMelodyOctave: (octaves) =>
    set((state) => {
      const melody = shiftedMelody(state.melody, octaves);
      if (!melody) return state;
      return {
        melody,
        melodyRevision: state.melodyRevision + 1,
        transcriptionRegister: state.transcriptionRegister
          ? {
              ...state.transcriptionRegister,
              currentOctaveShift:
                state.transcriptionRegister.currentOctaveShift +
                Math.trunc(octaves),
            }
          : undefined,
        selectedNotes: [],
      };
    }),
  restoreSungRegister: () =>
    set((state) => {
      const currentShift =
        state.transcriptionRegister?.currentOctaveShift ?? 0;
      const melody = shiftedMelody(state.melody, -currentShift);
      if (!melody || !state.transcriptionRegister) return state;
      return {
        melody,
        melodyRevision: state.melodyRevision + 1,
        transcriptionRegister: {
          ...state.transcriptionRegister,
          currentOctaveShift: 0,
        },
        selectedNotes: [],
      };
    }),
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
      transcriptionRegister:
        state.melody.notes.length === 1
          ? undefined
          : state.transcriptionRegister,
    })),
  clearMelody: () =>
    set((state) => ({
      melody: { ...state.melody, notes: [] },
      melodyRevision: state.melodyRevision + 1,
      currentBeat: 0,
      isPlaying: false,
      selectedNotes: [],
      loopRangeCustomized: false,
      transcriptionRegister: undefined,
    })),
});
