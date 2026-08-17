// frontend/src/musicConstants.ts
// Shared piano-roll/recording constants and timing helpers used by MidiInput and PianoRoll.

export const NUM_BARS = 8;
export const DIVISIONS_PER_BAR = 16;
export const TOTAL_BOXES = NUM_BARS * DIVISIONS_PER_BAR;
export const MIDI_LOW = 48; // C3
export const MIDI_HIGH = 79; // G5
export const PITCHES = MIDI_HIGH - MIDI_LOW + 1;

export interface MidiNote {
  midi: number;
  startTime: number;
  endTime?: number;
  duration?: number;
}

/** Duration of one 16th note in milliseconds, at the given tempo (BPM). */
export function getSixteenthNoteMs(tempo: number): number {
  return ((60 / tempo) * 1000) / 4;
}

/** Total duration of the fixed-length piano roll (NUM_BARS bars), in milliseconds. */
export function getTotalDurationMs(tempo: number): number {
  return getSixteenthNoteMs(tempo) * TOTAL_BOXES;
}
