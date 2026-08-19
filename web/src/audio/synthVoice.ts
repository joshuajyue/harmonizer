import type { PlaybackNote } from "./AudioScheduler";

interface VoicePatch {
  oscillator: OscillatorType;
  gain: number;
  lowpass?: {
    minimum: number;
    maximum: number;
    harmonicMultiple: number;
  };
}

const VOICE_PATCHES: Record<PlaybackNote["voice"], VoicePatch> = {
  melody: { oscillator: "sine", gain: 1 },
  soprano: { oscillator: "sine", gain: 1 },
  alto: { oscillator: "triangle", gain: 0.9 },
  tenor: {
    oscillator: "square",
    gain: 0.72,
    lowpass: { minimum: 1_800, maximum: 3_400, harmonicMultiple: 14 },
  },
  bass: {
    oscillator: "sawtooth",
    gain: 0.78,
    lowpass: { minimum: 1_400, maximum: 2_800, harmonicMultiple: 18 },
  },
};

function bassRegisterCompensation(note: PlaybackNote) {
  if (note.voice !== "bass") return 1;
  const lowRegisterRatio = Math.max(0, Math.min(1, (60 - note.pitch) / 20));
  return 1 + lowRegisterRatio * 0.32;
}

export function createSynthVoice(
  context: AudioContext,
  output: AudioNode,
  note: PlaybackNote,
  startTime: number,
  duration: number,
  onEnded: () => void,
) {
  const oscillator = context.createOscillator();
  const gain = context.createGain();
  const patch = VOICE_PATCHES[note.voice];
  const frequency = 440 * 2 ** ((note.pitch - 69) / 12);
  const velocity = Math.max(1, Math.min(127, note.velocity));
  const level =
    0.085 *
    (velocity / 100) *
    patch.gain *
    bassRegisterCompensation(note);
  const releaseAt = Math.max(startTime + 0.025, startTime + duration - 0.06);

  oscillator.type = patch.oscillator;
  oscillator.frequency.setValueAtTime(frequency, startTime);
  gain.gain.setValueAtTime(0.0001, startTime);
  gain.gain.exponentialRampToValueAtTime(level, startTime + 0.012);
  gain.gain.setValueAtTime(level * 0.72, releaseAt);
  gain.gain.exponentialRampToValueAtTime(
    0.0001,
    startTime + duration + 0.035,
  );
  if (patch.lowpass) {
    const filter = context.createBiquadFilter();
    const cutoff = Math.max(
      patch.lowpass.minimum,
      Math.min(
        patch.lowpass.maximum,
        frequency * patch.lowpass.harmonicMultiple,
      ),
    );
    filter.type = "lowpass";
    filter.frequency.setValueAtTime(cutoff, startTime);
    filter.Q.setValueAtTime(0.7, startTime);
    oscillator.connect(filter).connect(gain).connect(output);
  } else {
    oscillator.connect(gain).connect(output);
  }
  oscillator.start(startTime);
  oscillator.stop(startTime + duration + 0.04);
  oscillator.onended = onEnded;
  return oscillator;
}
