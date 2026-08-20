import { createSynthVoice } from "./synthVoice";
import type { PlaybackNote, PlaybackOptions } from "./types";
import type { TimeSignature } from "../../../contracts/types";

export type { PlaybackNote } from "./types";

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
  private clickNodes = new Set<OscillatorNode>();
  private generation = 0;
  private liveNotes = new Map<
    number,
    { oscillator: OscillatorNode; gain: GainNode }
  >();

  async start(options: PlaybackOptions) {
    this.stop();
    const generation = this.generation;
    const context = this.getContext();
    await context.resume();
    if (generation !== this.generation) return;
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
    this.generation += 1;
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
    this.clickNodes.clear();
    this.options = undefined;
  }

  preview(
    pitch: number,
    velocity = 90,
    duration = 0.35,
    voice: PlaybackNote["voice"] = "melody",
  ) {
    const context = this.getContext();
    void context.resume();
    this.createVoice(
      {
        pitch,
        start: 0,
        duration: 1,
        velocity,
        voice,
      },
      context.currentTime + 0.01,
      duration,
    );
  }

  previewNoteOn(pitch: number, velocity = 90) {
    this.previewNoteOff(pitch);
    const context = this.getContext();
    const output = this.output;
    if (!output) return;
    void context.resume();
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    const now = context.currentTime;
    const level = 0.085 * (Math.max(1, Math.min(127, velocity)) / 100);
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(
      440 * 2 ** ((pitch - 69) / 12),
      now,
    );
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(level, now + 0.012);
    gain.gain.setTargetAtTime(level * 0.72, now + 0.014, 0.04);
    oscillator.connect(gain).connect(output);
    oscillator.start(now);
    const liveNote = { oscillator, gain };
    this.liveNotes.set(pitch, liveNote);
    oscillator.onended = () => {
      if (this.liveNotes.get(pitch) === liveNote) {
        this.liveNotes.delete(pitch);
      }
    };
  }

  previewNoteOff(pitch: number) {
    const liveNote = this.liveNotes.get(pitch);
    const context = this.context;
    if (!liveNote || !context) return;
    this.liveNotes.delete(pitch);
    const now = context.currentTime;
    liveNote.gain.gain.cancelScheduledValues(now);
    liveNote.gain.gain.setValueAtTime(
      Math.max(0.0001, liveNote.gain.gain.value),
      now,
    );
    liveNote.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.055);
    liveNote.oscillator.stop(now + 0.065);
  }

  previewAllNotesOff() {
    for (const pitch of [...this.liveNotes.keys()]) {
      this.previewNoteOff(pitch);
    }
  }

  getCurrentBeat() {
    const options = this.options;
    const context = this.context;
    if (!options || !context) return undefined;
    const rawBeat =
      this.originBeat +
      Math.max(0, context.currentTime - this.originTime) *
        (options.tempo / 60);
    return this.displayBeat(rawBeat);
  }

  setLoop({
    enabled,
    start,
    end,
  }: {
    enabled: boolean;
    start: number;
    end: number;
  }) {
    const options = this.options;
    if (!options) return;
    const currentBeat = this.getCurrentBeat() ?? this.originBeat;
    const loopStart = Math.max(0, start);
    const loopEnd = Math.max(loopStart + 0.25, end);
    const loopEnabled = enabled && currentBeat < loopEnd;
    const activeStateChanged = options.loopEnabled !== loopEnabled;
    const rangeChanged =
      options.loopStart !== loopStart || options.loopEnd !== loopEnd;
    if (!activeStateChanged && !rangeChanged) return;
    options.loopStart = loopStart;
    options.loopEnd = loopEnd;
    options.loopEnabled = loopEnabled;
    if (activeStateChanged || loopEnabled) {
      this.rebaseTimeline(currentBeat);
    }
  }

  setMetronomeEnabled(enabled: boolean) {
    const options = this.options;
    if (options && options.metronomeEnabled !== enabled) {
      options.metronomeEnabled = enabled;
      if (!enabled) {
        for (const node of this.clickNodes) {
          try {
            node.stop();
          } catch {
            // A click may already have ended between frames.
          }
          this.nodes.delete(node);
        }
        this.clickNodes.clear();
      }
    }
  }

  setTempo(tempo: number) {
    const options = this.options;
    if (
      !options ||
      !Number.isFinite(tempo) ||
      tempo <= 0 ||
      options.tempo === tempo
    ) {
      return;
    }
    const currentBeat = this.getCurrentBeat() ?? this.originBeat;
    options.tempo = tempo;
    this.rebaseTimeline(currentBeat);
  }

  setTimeSignature(timeSignature: TimeSignature) {
    const options = this.options;
    if (!options) return;
    const current = options.timeSignature;
    if (
      current.numerator !== timeSignature.numerator ||
      current.denominator !== timeSignature.denominator
    ) {
      options.timeSignature = timeSignature;
    }
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
        if (
          fromBeat < options.loopStart &&
          note.start >= fromBeat &&
          note.start < Math.min(toBeat, options.loopStart)
        ) {
          this.scheduleNote(note, note.start, note.duration);
        }
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
    const oscillator = createSynthVoice(
      context,
      output,
      note,
      startTime,
      duration,
      () => this.nodes.delete(oscillator),
    );
    this.nodes.add(oscillator);
  }

  private rebaseTimeline(beat: number) {
    const context = this.context;
    if (!this.options || !context) return;
    for (const node of this.nodes) {
      try {
        node.stop();
      } catch {
        // A node may already have ended between frames.
      }
    }
    this.nodes.clear();
    this.clickNodes.clear();
    this.originBeat = beat;
    this.originTime = context.currentTime + 0.01;
    this.scheduledThrough = beat - 0.001;
    this.scheduleOverlappingNotes(beat);
    this.scheduleWindow();
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
      this.clickNodes.add(oscillator);
      oscillator.onended = () => {
        this.nodes.delete(oscillator);
        this.clickNodes.delete(oscillator);
      };
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
