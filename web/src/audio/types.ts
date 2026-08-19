import type { TimeSignature, VoiceName } from "../../../contracts/types";

export interface PlaybackNote {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  voice: VoiceName | "melody";
}

export interface PlaybackOptions {
  notes: PlaybackNote[];
  tempo: number;
  startBeat: number;
  endBeat: number;
  loopEnabled: boolean;
  loopStart: number;
  loopEnd: number;
  metronomeEnabled: boolean;
  timeSignature: TimeSignature;
  onPosition: (beat: number) => void;
  onEnded: () => void;
}
