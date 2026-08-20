import type { RenderRequest } from "../../../contracts/types";

const SAMPLE_RATE = 22_050;

function writeString(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeWav(samples: Float32Array) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, SAMPLE_RATE, true);
  view.setUint32(28, SAMPLE_RATE * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);

  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]));
    view.setInt16(44 + index * 2, sample * 0x7fff, true);
  }
  return buffer;
}

export function createMockWav(request: RenderRequest) {
  const secondsPerBeat = 60 / request.tempo;
  const lastBeat = Math.max(
    1,
    ...request.voices.flatMap((voice) =>
      voice.notes.map((note) => note.start + note.duration),
    ),
  );
  const samples = new Float32Array(
    Math.ceil((lastBeat * secondsPerBeat + 0.4) * SAMPLE_RATE),
  );

  request.voices.forEach((voice, voiceIndex) => {
    voice.notes.forEach((note) => {
      const start = Math.floor(note.start * secondsPerBeat * SAMPLE_RATE);
      const length = Math.floor(
        note.duration * secondsPerBeat * SAMPLE_RATE,
      );
      const frequency = 440 * 2 ** ((note.pitch - 69) / 12);
      const amplitude = 0.075 * ((note.velocity ?? 80) / 100);

      for (let index = 0; index < length; index += 1) {
        const progress = index / Math.max(1, length);
        const attack = Math.min(1, index / (SAMPLE_RATE * 0.012));
        const release = Math.min(1, (1 - progress) / 0.08);
        const phase = (2 * Math.PI * frequency * index) / SAMPLE_RATE;
        const overtone =
          Math.sin(phase) +
          Math.sin(phase * 2 + voiceIndex * 0.3) * 0.16 +
          Math.sin(phase * 3) * 0.05;
        samples[start + index] +=
          overtone * amplitude * attack * release * 0.72;
      }
    });
  });

  return new Blob([encodeWav(samples)], { type: "audio/wav" });
}
