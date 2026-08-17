// frontend/src/audio/metronome.ts
// Synthesizes short metronome click sounds directly via the Web Audio API (no audio files).
import { createAudioContext } from "./audioContext";

export type MetronomeSound = "click" | "beep" | "boop" | "wood" | "tick" | "cowbell";

export function playClick(frequency = 1000, duration = 0.05, soundType: MetronomeSound | string = "click") {
  const ctx = createAudioContext();

  switch (soundType) {
    case "beep": {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = frequency;
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
      break;
    }

    case "boop": {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "triangle";
      osc.frequency.setValueAtTime(frequency * 0.8, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(frequency * 0.6, ctx.currentTime + duration);
      gain.gain.value = 0.25;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
      break;
    }

    case "wood": {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      const filter = ctx.createBiquadFilter();
      osc.type = "square";
      osc.frequency.value = frequency * 2;
      filter.type = "bandpass";
      filter.frequency.value = frequency * 1.5;
      gain.gain.setValueAtTime(0.4, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration * 0.3);
      osc.connect(filter);
      filter.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration * 0.3);
      break;
    }

    case "tick": {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.value = frequency * 4;
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration * 0.2);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration * 0.2);
      break;
    }

    case "cowbell": {
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();
      osc1.type = "square";
      osc2.type = "square";
      osc1.frequency.value = frequency * 2.5;
      osc2.frequency.value = frequency * 3.2;
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + duration * 0.5);
      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);
      osc1.start();
      osc2.start();
      osc1.stop(ctx.currentTime + duration * 0.5);
      osc2.stop(ctx.currentTime + duration * 0.5);
      break;
    }

    default: {
      // "click"
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.value = frequency;
      gain.gain.value = 0.2;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + duration);
      break;
    }
  }

  // Clean up the audio context shortly after the sound finishes.
  setTimeout(() => ctx.close(), (duration + 0.1) * 1000);
}
