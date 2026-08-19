import type { StateCreator } from "zustand";
import type { StudioStore, TransportSlice } from "./types";

export const createTransportSlice: StateCreator<
  StudioStore,
  [],
  [],
  TransportSlice
> = (set) => ({
  isPlaying: false,
  currentBeat: 0,
  loopEnabled: false,
  loopStart: 0,
  loopEnd: 16,
  loopRangeCustomized: false,
  metronomeEnabled: true,
  recordingState: "idle",
  countInBars: 1,
  countInRemaining: 0,
  setPlaying: (isPlaying) => set({ isPlaying }),
  setCurrentBeat: (currentBeat) => set({ currentBeat: Math.max(0, currentBeat) }),
  setLoopEnabled: (loopEnabled) => set({ loopEnabled }),
  setLoopRange: (loopStart, loopEnd, loopRangeCustomized = true) =>
    set({
      loopStart: Math.max(0, Math.min(loopStart, loopEnd - 0.25)),
      loopEnd: Math.max(loopStart + 0.25, loopEnd),
      loopRangeCustomized,
    }),
  setMetronomeEnabled: (metronomeEnabled) => set({ metronomeEnabled }),
  setRecordingState: (recordingState) => set({ recordingState }),
  setCountInBars: (countInBars) => set({ countInBars }),
  setCountInRemaining: (countInRemaining) =>
    set({ countInRemaining: Math.max(0, Math.round(countInRemaining)) }),
});
