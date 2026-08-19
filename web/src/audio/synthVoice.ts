import type { PlaybackNote } from "./AudioScheduler";

const OSCILLATORS: Record<PlaybackNote["voice"], OscillatorType> = {
  melody: "sine",
  soprano: "sine",
  alto: "triangle",
  tenor: "triangle",
  bass: "sine",
};

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
  const frequency = 440 * 2 ** ((note.pitch - 69) / 12);
  const level = 0.085 * (note.velocity / 100);
  const releaseAt = Math.max(startTime + 0.025, startTime + duration - 0.06);

  oscillator.type = OSCILLATORS[note.voice];
  oscillator.frequency.setValueAtTime(frequency, startTime);
  gain.gain.setValueAtTime(0.0001, startTime);
  gain.gain.exponentialRampToValueAtTime(level, startTime + 0.012);
  gain.gain.setValueAtTime(level * 0.72, releaseAt);
  gain.gain.exponentialRampToValueAtTime(
    0.0001,
    startTime + duration + 0.035,
  );
  oscillator.connect(gain).connect(output);
  oscillator.start(startTime);
  oscillator.stop(startTime + duration + 0.04);
  oscillator.onended = onEnded;
  return oscillator;
}
