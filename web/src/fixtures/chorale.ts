import type {
  Chord,
  EngineInfo,
  HarmonizeResponse,
  Melody,
  Note,
  Violation,
  Voice,
} from "../../../contracts/types";

const note = (
  pitch: number,
  start: number,
  duration = 1,
  velocity = 82,
): Note => ({ pitch, start, duration, velocity });

export const DEMO_MELODY: Melody = {
  tempo: 88,
  timeSignature: { numerator: 4, denominator: 4 },
  key: { tonic: 0, mode: "major", confidence: 0.98 },
  notes: [
    note(72, 0),
    note(74, 1),
    note(76, 2),
    note(79, 3),
    note(77, 4, 2),
    note(76, 6),
    note(74, 7),
    note(72, 8),
    note(76, 9),
    note(79, 10, 2),
    note(81, 12),
    note(79, 13),
    note(77, 14),
    note(76, 15),
  ],
};

const chords: Chord[] = [
  {
    start: 0,
    duration: 2,
    roman: "I",
    root: 0,
    quality: "maj",
    inversion: 0,
  },
  {
    start: 2,
    duration: 2,
    roman: "V65",
    root: 7,
    quality: "dom7",
    inversion: 1,
  },
  {
    start: 4,
    duration: 2,
    roman: "vi",
    root: 9,
    quality: "min",
    inversion: 0,
  },
  {
    start: 6,
    duration: 2,
    roman: "V/V",
    root: 2,
    quality: "dom7",
    inversion: 0,
    secondaryOf: 5,
  },
  {
    start: 8,
    duration: 2,
    roman: "IV",
    root: 5,
    quality: "maj",
    inversion: 0,
  },
  {
    start: 10,
    duration: 2,
    roman: "ii6",
    root: 2,
    quality: "min",
    inversion: 1,
  },
  {
    start: 12,
    duration: 2,
    roman: "V7",
    root: 7,
    quality: "dom7",
    inversion: 0,
  },
  {
    start: 14,
    duration: 2,
    roman: "I",
    root: 0,
    quality: "maj",
    inversion: 0,
  },
];

const voices: Voice[] = [
  {
    name: "soprano",
    notes: DEMO_MELODY.notes.map((value) => ({ ...value, velocity: 88 })),
  },
  {
    name: "alto",
    notes: [
      note(67, 0),
      note(69, 1),
      note(67, 2),
      note(71, 3),
      note(69, 4, 2),
      note(71, 6),
      note(69, 7),
      note(67, 8),
      note(69, 9),
      note(71, 10, 2),
      note(72, 12),
      note(71, 13),
      note(69, 14),
      note(67, 15),
    ],
  },
  {
    name: "tenor",
    notes: [
      note(64, 0),
      note(62, 1),
      note(64, 2),
      note(62, 3),
      note(60, 4, 2),
      note(62, 6),
      note(66, 7),
      note(65, 8),
      note(64, 9),
      note(62, 10, 2),
      note(64, 12),
      note(62, 13),
      note(65, 14),
      note(64, 15),
    ],
  },
  {
    name: "bass",
    notes: [
      note(48, 0),
      note(47, 1),
      note(48, 2),
      note(43, 3),
      note(45, 4, 2),
      note(50, 6),
      note(43, 7),
      note(41, 8),
      note(45, 9),
      note(50, 10, 2),
      note(43, 12),
      note(47, 13),
      note(43, 14),
      note(48, 15),
    ],
  },
];

const ruleViolations: Violation[] = [
  {
    kind: "parallel_fifths",
    severity: "warning",
    start: 3,
    voices: ["soprano", "bass"],
    message: "Parallel perfect fifth between soprano and bass into beat 4.",
  },
  {
    kind: "spacing",
    severity: "info",
    start: 6,
    voices: ["alto", "tenor"],
    message: "Upper-voice spacing briefly exceeds an octave.",
  },
  {
    kind: "voice_crossing",
    severity: "error",
    start: 10,
    voices: ["tenor", "bass"],
    message: "Tenor drops below the bass at the ii6 arrival.",
  },
];

const learnedViolations: Violation[] = [
  {
    kind: "parallel_fifths",
    severity: "warning",
    start: 3,
    voices: ["soprano", "bass"],
    message: "Parallel perfect fifth between soprano and bass into beat 4.",
  },
  {
    kind: "unresolved_leading_tone",
    severity: "info",
    start: 13,
    voices: ["alto"],
    message: "The alto leading tone resolves downward rather than to tonic.",
  },
];

export const MOCK_ENGINES: EngineInfo[] = [
  {
    id: "rules-v2",
    name: "Species Rules",
    description: "Deterministic search with explicit voice-leading constraints.",
    available: true,
    learned: false,
  },
  {
    id: "chorale-transformer",
    name: "Chorale Transformer",
    description: "Learned four-part continuation model trained on chorales.",
    available: true,
    learned: true,
  },
  {
    id: "experimental-diffusion",
    name: "Voice Diffusion",
    description: "Exploratory stochastic voicing model.",
    available: false,
    learned: true,
  },
];

const cloneNotes = (notes: Note[]) => notes.map((value) => ({ ...value }));

export function createMockResponse(
  engine: string,
  melody: Melody = DEMO_MELODY,
): HarmonizeResponse {
  const learned = engine !== "rules-v2";
  const responseVoices = voices.map((voice) => ({
    name: voice.name,
    notes:
      voice.name === "soprano"
        ? cloneNotes(melody.notes)
        : cloneNotes(voice.notes),
  }));

  if (learned) {
    const alto = responseVoices.find((voice) => voice.name === "alto");
    const tenor = responseVoices.find((voice) => voice.name === "tenor");
    if (alto) alto.notes[6] = note(69, 6, 1, 84);
    if (tenor) tenor.notes[10] = note(64, 10, 2, 80);
  }

  return {
    key: melody.key ?? { tonic: 0, mode: "major", confidence: 0.91 },
    chords: chords.map((chord) => ({ ...chord })),
    voices: responseVoices,
    violations: (learned ? learnedViolations : ruleViolations).map(
      (violation) => ({ ...violation, voices: [...violation.voices] }),
    ),
    engine,
    latencyMs: learned ? 386 : 74,
  };
}
