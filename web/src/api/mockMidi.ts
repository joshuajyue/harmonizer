import { Midi } from "@tonejs/midi";
import type { HarmonizeResponse } from "../../../contracts/types";

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
