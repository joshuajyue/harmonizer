/**
 * Shared API contract for HarmonAIzer v2.
 *
 * SOURCE OF TRUTH. Mirrored in contracts/schema.py — change both together.
 *
 * Core v2 change vs v1: an engine returns *voiced parts* (actual notes, with voice
 * leading), not just a chord label per beat. Chord labels are metadata; the voices
 * are the product.
 */

export type Mode = "major" | "minor";

/** Absolute MIDI pitch, 0-127. Middle C = 60. */
export type Midi = number;

/** Position/duration in quarter-note beats from the start of the piece. */
export type Beats = number;

export interface Note {
  pitch: Midi;
  start: Beats;
  duration: Beats;
  /** 0-127. Defaults to 80 when generated. */
  velocity?: number;
}

export interface KeySignature {
  /** Pitch class of the tonic, 0-11 (C = 0). */
  tonic: number;
  mode: Mode;
  /** Engine confidence in the detection, 0-1. */
  confidence?: number;
}

export interface TimeSignature {
  numerator: number;
  denominator: number;
}

export interface Melody {
  notes: Note[];
  /** Beats per minute. Required — a defaulted tempo renders at the wrong speed silently. */
  tempo: number;
  /** Defaults to 4/4 when omitted. */
  timeSignature?: TimeSignature;
  /** If omitted, the backend detects the key and returns it in the response. */
  key?: KeySignature;
}

/**
 * A harmonic event. `roman` is the display form ("V65", "bII7", "V/V") and is the
 * only field the UI should render as text.
 */
export interface Chord {
  start: Beats;
  duration: Beats;
  roman: string;
  /** Pitch class of the chord root, 0-11. */
  root: number;
  /**
   * Core chord quality, without extensions:
   * "maj" | "min" | "dim" | "aug" | "dom7" | "maj7" | "min7" | "halfdim7" | "dim7"
   * | "minmaj7" | "maj6" | "min6" | "sus2" | "sus4"
   *
   * Extensions and alterations live in `extensions` rather than being enumerated
   * here, because the cross product (13b9#11, 7alt, ...) is unbounded.
   */
  quality: string;
  /** 0 = root position, 1 = first inversion, ... */
  inversion: number;
  /** Non-null when the chord tonicizes another degree (e.g. V/V). */
  secondaryOf?: number | null;
  /**
   * Added and altered tones above the core quality, e.g. ["9", "#11", "b13"].
   * Always present; empty for plain triads and sevenths.
   */
  extensions: string[];
  /**
   * Reharmonization provenance. Non-null only when this chord replaced one in a
   * base progression, so the UI can show what was substituted and why — the same
   * principle as `violations`: explain the decision rather than just emit it.
   */
  substitutionOf?: string | null;
  /**
   * How the substitution was derived: "tritone" | "backdoor" | "modal_interchange"
   * | "relative" | "passing_dim" | "secondary_dominant" | "chromatic_approach"
   * | "extension" | "coltrane"
   */
  substitutionKind?: string | null;
}

export type VoiceName = "soprano" | "alto" | "tenor" | "bass";

export interface Voice {
  name: VoiceName;
  notes: Note[];
}

/** A voice-leading or style rule the result violates. Surfaced in the UI, not hidden. */
export interface Violation {
  /** e.g. "parallel_fifths", "voice_crossing", "unresolved_leading_tone", "spacing" */
  kind: string;
  severity: "info" | "warning" | "error";
  start: Beats;
  voices: VoiceName[];
  message: string;
}

export interface HarmonizeRequest {
  melody: Melody;
  /** Engine id from GET /api/v1/engines. */
  engine: string;
  options?: {
    /** Defaults to 4 (SATB). */
    voiceCount?: number;
    /** 0 = deterministic/argmax. Higher = more adventurous. */
    temperature?: number;
    /** Random seed; same seed + same input must reproduce the same output. */
    seed?: number;
  };
}

export interface HarmonizeResponse {
  key: KeySignature;
  chords: Chord[];
  voices: Voice[];
  violations: Violation[];
  engine: string;
  latencyMs: number;
}

export interface EngineInfo {
  id: string;
  name: string;
  description: string;
  available: boolean;
  /** False for the rule engine; true for learned models. */
  learned: boolean;
}

export interface EnginesResponse {
  engines: EngineInfo[];
}

/** POST /api/v1/render -> audio/wav. Renders voices to audio via the synthesis backend. */
export interface RenderRequest {
  voices: Voice[];
  tempo: number;
  /** Synthesis backend id. Defaults to "sf2" (fast preview); "ddsp" is the neural voice. */
  synth?: string;
  /** Timbre/model id for neural synths. */
  timbre?: string;
}
