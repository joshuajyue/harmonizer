import { Midi } from "@tonejs/midi";
import type {
  HarmonizeResponse,
  Melody,
} from "../../../contracts/types";

export function createMockMidi(
  harmonization: HarmonizeResponse,
  tempo: number,
) {
  const midi = new Midi();
  midi.header.setTempo(tempo);
  const ppq = midi.header.ppq;

  for (const voice of harmonization.voices) {
    const track = midi.addTrack();
    track.name = voice.name;
    for (const note of voice.notes) {
      track.addNote({
        midi: note.pitch,
        ticks: Math.round(note.start * ppq),
        durationTicks: Math.max(1, Math.round(note.duration * ppq)),
        velocity: (note.velocity ?? 80) / 127,
      });
    }
  }

  const bytes = midi.toArray();
  return new Blob([bytes.buffer as ArrayBuffer], { type: "audio/midi" });
}

export async function parseMockMidi(
  file: File,
  currentTempo: number,
): Promise<{ melody: Melody; notice?: string }> {
  const midi = new Midi(await file.arrayBuffer());
  const track = [...midi.tracks].sort(
    (first, second) => second.notes.length - first.notes.length,
  )[0];
  if (!track || track.notes.length === 0) {
    throw new Error("No note track found in this MIDI file.");
  }
  const headerTempo = midi.header.tempos[0]?.bpm;
  const timeSignature =
    midi.header.timeSignatures[0]?.timeSignature ?? [4, 4];
  return {
    melody: {
      tempo: Math.round(headerTempo ?? currentTempo),
      timeSignature: {
        numerator: timeSignature[0],
        denominator: timeSignature[1],
      },
      notes: track.notes.map((note) => ({
        pitch: note.midi,
        start: note.ticks / midi.header.ppq,
        duration: note.durationTicks / midi.header.ppq,
        velocity: Math.round(note.velocity * 127),
      })),
    },
    notice: headerTempo
      ? undefined
      : `No tempo event; using ${currentTempo} BPM.`,
  };
}
