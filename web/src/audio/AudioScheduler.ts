import type { TimeSignature, VoiceName } from "../../../contracts/types";

export interface PlaybackNote {
  pitch: number;
  start: number;
  duration: number;
  velocity: number;
  voice: VoiceName | "melody";
}

interface PlaybackOptions {
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

const OSCILLATORS: Record<PlaybackNote["voice"], OscillatorType> = {
  melody: "sine",
  soprano: "sine",
  alto: "triangle",
  tenor: "triangle",
  bass: "sine",
};

export class AudioScheduler {
  private context?: AudioContext;
  private output?: GainNode;
  private options?: PlaybackOptions;
  private intervalId?: number;
  private frameId?: number;
  private originTime = 0;
  private originBeat = 0;
  private scheduledThrough = 0;
  private nodes = new Set<OscillatorNode>();

  async start(options: PlaybackOptions) {
    this.stop();
    const context = this.getContext();
    await context.resume();
    this.options = options;
    this.originBeat = options.startBeat;
    this.originTime = context.currentTime + 0.045;
    this.scheduledThrough = options.startBeat - 0.001;
    this.scheduleOverlappingNotes(options.startBeat);
    this.scheduleWindow();
    this.intervalId = window.setInterval(() => this.scheduleWindow(), 25);
    this.updatePosition();
  }

  stop() {
    if (this.intervalId !== undefined) {
      window.clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
    if (this.frameId !== undefined) {
      window.cancelAnimationFrame(this.frameId);
      this.frameId = undefined;
    }
    for (const node of this.nodes) {
      try {
        node.stop();
      } catch {
        // A node may already have ended between frames.
      }
    }
    this.nodes.clear();
    this.options = undefined;
  }

  preview(pitch: number, velocity = 90, duration = 0.35) {
    const context = this.getContext();
    void context.resume();
    this.createVoice(
      {
        pitch,
        start: 0,
        duration: 1,
        velocity,
        voice: "melody",
      },
      context.currentTime + 0.01,
      duration,
    );
  }

  private getContext() {
    if (!this.context) {
      this.context = new AudioContext({ latencyHint: "interactive" });
      this.output = this.context.createGain();
      this.output.gain.value = 0.72;
      this.output.connect(this.context.destination);
    }
    return this.context;
  }

  private scheduleWindow() {
    const options = this.options;
    const context = this.context;
    if (!options || !context) return;
    const beatsPerSecond = options.tempo / 60;
    const nowBeat =
      this.originBeat +
      Math.max(0, context.currentTime - this.originTime) * beatsPerSecond;
    const horizon = nowBeat + 0.16 * beatsPerSecond;
    this.scheduleNotes(this.scheduledThrough, horizon);
    if (options.metronomeEnabled) {
      this.scheduleClicks(this.scheduledThrough, horizon);
    }
    this.scheduledThrough = horizon;
  }

  private scheduleNotes(fromBeat: number, toBeat: number) {
    const options = this.options;
    if (!options) return;
    for (const note of options.notes) {
      if (options.loopEnabled) {
        if (note.start < options.loopStart || note.start >= options.loopEnd) {
          continue;
        }
        const loopLength = options.loopEnd - options.loopStart;
        const firstCycle = Math.max(
          0,
          Math.floor((fromBeat - note.start) / loopLength),
        );
        for (let cycle = firstCycle; cycle <= firstCycle + 2; cycle += 1) {
          const occurrence = note.start + cycle * loopLength;
          if (occurrence >= fromBeat && occurrence < toBeat) {
            const duration = Math.min(
              note.duration,
              options.loopEnd - note.start,
            );
            this.scheduleNote(note, occurrence, duration);
          }
        }
      } else if (note.start >= fromBeat && note.start < toBeat) {
        this.scheduleNote(note, note.start, note.duration);
      }
    }
  }

  private scheduleOverlappingNotes(startBeat: number) {
    const options = this.options;
    if (!options) return;
    for (const note of options.notes) {
      if (note.start < startBeat && note.start + note.duration > startBeat) {
        const remaining = note.start + note.duration - startBeat;
        this.scheduleNote(note, startBeat, remaining);
      }
    }
  }

  private scheduleNote(
    note: PlaybackNote,
    occurrence: number,
    durationBeats: number,
  ) {
    const options = this.options;
    const context = this.context;
    if (!options || !context) return;
    const beatsPerSecond = options.tempo / 60;
    const startTime =
      this.originTime + (occurrence - this.originBeat) / beatsPerSecond;
    const duration = durationBeats / beatsPerSecond;
    if (startTime + duration > context.currentTime) {
      this.createVoice(
        note,
        Math.max(context.currentTime + 0.004, startTime),
        duration,
      );
    }
  }

  private createVoice(note: PlaybackNote, startTime: number, duration: number) {
    const context = this.context;
    const output = this.output;
    if (!context || !output || duration <= 0) return;
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
    this.nodes.add(oscillator);
    oscillator.onended = () => this.nodes.delete(oscillator);
  }

  private scheduleClicks(fromBeat: number, toBeat: number) {
    const options = this.options;
    const context = this.context;
    const output = this.output;
    if (!options || !context || !output) return;
    const beatsPerSecond = options.tempo / 60;
    for (let beat = Math.ceil(fromBeat); beat < toBeat; beat += 1) {
      const displayed = this.displayBeat(beat);
      const barLength =
        options.timeSignature.numerator *
        (4 / options.timeSignature.denominator);
      const accented =
        Math.abs(displayed % barLength) < 0.001 ||
        Math.abs((displayed % barLength) - barLength) < 0.001;
      const time =
        this.originTime + (beat - this.originBeat) / beatsPerSecond;
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.frequency.value = accented ? 1_280 : 890;
      gain.gain.setValueAtTime(accented ? 0.11 : 0.065, time);
      gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.035);
      oscillator.connect(gain).connect(output);
      oscillator.start(time);
      oscillator.stop(time + 0.04);
      this.nodes.add(oscillator);
      oscillator.onended = () => this.nodes.delete(oscillator);
    }
  }

  private displayBeat(rawBeat: number) {
    const options = this.options;
    if (
      !options?.loopEnabled ||
      rawBeat < options.loopEnd ||
      options.loopEnd <= options.loopStart
    ) {
      return rawBeat;
    }
    return (
      options.loopStart +
      ((rawBeat - options.loopStart) %
        (options.loopEnd - options.loopStart))
    );
  }

  private updatePosition = () => {
    const options = this.options;
    const context = this.context;
    if (!options || !context) return;
    const rawBeat =
      this.originBeat +
      Math.max(0, context.currentTime - this.originTime) *
        (options.tempo / 60);
    if (!options.loopEnabled && rawBeat >= options.endBeat) {
      options.onPosition(options.endBeat);
      const onEnded = options.onEnded;
      this.stop();
      onEnded();
      return;
    }
    options.onPosition(this.displayBeat(rawBeat));
    this.frameId = window.requestAnimationFrame(this.updatePosition);
  };
}

export const audioScheduler = new AudioScheduler();
