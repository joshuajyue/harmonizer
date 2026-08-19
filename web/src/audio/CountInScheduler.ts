import type { TimeSignature } from "../../../contracts/types";

interface CountInOptions {
  bars: number;
  tempo: number;
  timeSignature: TimeSignature;
  onRemaining: (pulses: number) => void;
}

class CountInScheduler {
  private context?: AudioContext;
  private output?: GainNode;
  private frameId?: number;
  private generation = 0;
  private nodes = new Set<OscillatorNode>();
  private resolve?: (completed: boolean) => void;

  async start(options: CountInOptions) {
    this.cancel();
    const generation = this.generation;
    const context = this.getContext();
    await context.resume();
    if (generation !== this.generation) return false;

    const bars = Math.max(0, Math.round(options.bars));
    const pulses = bars * options.timeSignature.numerator;
    if (pulses === 0) return true;
    const beatUnit = 4 / options.timeSignature.denominator;
    const secondsPerPulse = (60 / options.tempo) * beatUnit;
    const startTime = context.currentTime + 0.06;
    const endTime = startTime + pulses * secondsPerPulse;

    options.onRemaining(pulses);
    for (let pulse = 0; pulse < pulses; pulse += 1) {
      this.scheduleClick(
        startTime + pulse * secondsPerPulse,
        pulse % options.timeSignature.numerator === 0,
      );
    }

    return new Promise<boolean>((resolve) => {
      this.resolve = resolve;
      const update = () => {
        if (generation !== this.generation) return;
        const elapsed = Math.max(0, context.currentTime - startTime);
        const completed = Math.min(
          pulses,
          Math.floor(elapsed / secondsPerPulse),
        );
        options.onRemaining(Math.max(0, pulses - completed));
        if (context.currentTime >= endTime) {
          this.frameId = undefined;
          this.resolve = undefined;
          options.onRemaining(0);
          resolve(true);
          return;
        }
        this.frameId = window.requestAnimationFrame(update);
      };
      this.frameId = window.requestAnimationFrame(update);
    });
  }

  cancel() {
    this.generation += 1;
    if (this.frameId !== undefined) {
      window.cancelAnimationFrame(this.frameId);
      this.frameId = undefined;
    }
    for (const node of this.nodes) {
      try {
        node.stop();
      } catch {
        // A click may already have ended between frames.
      }
    }
    this.nodes.clear();
    this.resolve?.(false);
    this.resolve = undefined;
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

  private scheduleClick(time: number, accented: boolean) {
    const context = this.context;
    const output = this.output;
    if (!context || !output) return;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.frequency.value = accented ? 1_280 : 890;
    gain.gain.setValueAtTime(accented ? 0.12 : 0.075, time);
    gain.gain.exponentialRampToValueAtTime(0.0001, time + 0.045);
    oscillator.connect(gain).connect(output);
    oscillator.start(time);
    oscillator.stop(time + 0.05);
    this.nodes.add(oscillator);
    oscillator.onended = () => this.nodes.delete(oscillator);
  }
}

export const countInScheduler = new CountInScheduler();
