import type {
  HarmonizeResponse,
  Melody,
  Note,
  VoiceName,
} from "../../../contracts/types";

export const KEY_NAMES = [
  "C",
  "D♭",
  "D",
  "E♭",
  "E",
  "F",
  "F♯",
  "G",
  "A♭",
  "A",
  "B♭",
  "B",
] as const;

export const VOICE_ORDER: VoiceName[] = [
  "soprano",
  "alto",
  "tenor",
  "bass",
];

export const VOICE_COLORS: Record<VoiceName, string> = {
  soprano: "#ae8bff",
  alto: "#ff7eb6",
  tenor: "#52d6c9",
  bass: "#f5a65b",
};

export const VOICE_RANGES: Record<
  VoiceName | "melody",
  { min: number; max: number }
> = {
  melody: { min: 60, max: 84 },
  soprano: { min: 60, max: 84 },
  alto: { min: 53, max: 77 },
  tenor: { min: 45, max: 69 },
  bass: { min: 33, max: 57 },
};

export function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function quantize(value: number, step: number) {
  return Math.round(value / step) * step;
}

export function midiToName(pitch: number) {
  const names = [
    "C",
    "C♯",
    "D",
    "E♭",
    "E",
    "F",
    "F♯",
    "G",
    "A♭",
    "A",
    "B♭",
    "B",
  ];
  return `${names[((pitch % 12) + 12) % 12]}${Math.floor(pitch / 12) - 1}`;
}

export function pieceLength(
  melody: Melody,
  results: Array<HarmonizeResponse | undefined> = [],
) {
  const ends = melody.notes.map((note) => note.start + note.duration);
  for (const result of results) {
    for (const voice of result?.voices ?? []) {
      ends.push(...voice.notes.map((note) => note.start + note.duration));
    }
    ends.push(
      ...(result?.chords.map((chord) => chord.start + chord.duration) ?? []),
    );
  }
  const raw = Math.max(4, ...ends);
  const bar = melody.timeSignature.numerator * (4 / melody.timeSignature.denominator);
  return Math.ceil(raw / bar) * bar;
}

export function normalizeNotes(notes: Note[]) {
  return notes
    .map((note) => ({
      ...note,
      pitch: Math.round(clamp(note.pitch, 0, 127)),
      start: Math.max(0, note.start),
      duration: Math.max(0.0625, note.duration),
    }))
    .sort((a, b) => a.start - b.start || a.pitch - b.pitch);
}

export function barBeatLabel(
  beat: number,
  numerator: number,
  denominator: number,
) {
  const barLength = numerator * (4 / denominator);
  const bar = Math.floor(beat / barLength) + 1;
  const withinBar = beat % barLength;
  const beatUnit = 4 / denominator;
  const count = Math.floor(withinBar / beatUnit) + 1;
  return `${bar}.${count}`;
}

export function formatTime(seconds: number) {
  const safe = Math.max(0, seconds);
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${Math.floor(safe % 60)
    .toString()
    .padStart(2, "0")}`;
}

export function voiceLabel(voice: VoiceName) {
  return `${voice[0].toUpperCase()}${voice.slice(1)}`;
}
