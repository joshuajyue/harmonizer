import { Midi } from "@tonejs/midi";
import { FileMusic, LoaderCircle, Upload } from "lucide-react";
import { useRef, useState } from "react";
import type { Melody } from "../../../../contracts/types";
import { useStudioStore } from "../../store";

export function MidiFileInput() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [message, setMessage] = useState("Drop a standard MIDI file here");
  const currentMelody = useStudioStore((state) => state.melody);
  const replaceMelody = useStudioStore((state) => state.replaceMelody);

  async function loadFile(file?: File) {
    if (!file) return;
    setStatus("loading");
    try {
      const midi = new Midi(await file.arrayBuffer());
      const track = [...midi.tracks].sort(
        (a, b) => b.notes.length - a.notes.length,
      )[0];
      if (!track || track.notes.length === 0) {
        throw new Error("No note track found in this MIDI file.");
      }
      const headerTempo = midi.header.tempos[0]?.bpm;
      const tempo = Math.round(headerTempo ?? currentMelody.tempo);
      const timeSignature =
        midi.header.timeSignatures[0]?.timeSignature ?? [4, 4];
      const melody: Melody = {
        tempo,
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
      };
      replaceMelody(melody, file.name.replace(/\.(mid|midi)$/i, ""));
      setMessage(
        `${file.name} · ${melody.notes.length} melody notes${
          headerTempo
            ? ""
            : ` · no tempo event, using ${currentMelody.tempo} BPM`
        }`,
      );
      setStatus("idle");
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Could not parse MIDI file.",
      );
      setStatus("error");
    }
  }

  return (
    <div
      className={`midi-dropzone ${status === "error" ? "error" : ""}`}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        void loadFile(event.dataTransfer.files[0]);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".mid,.midi,audio/midi,audio/x-midi"
        onChange={(event) => void loadFile(event.currentTarget.files?.[0])}
        hidden
      />
      <div className="input-method-icon">
        {status === "loading" ? (
          <LoaderCircle size={19} className="spin" />
        ) : (
          <FileMusic size={19} />
        )}
      </div>
      <div>
        <strong>{message}</strong>
        <span>Tempo, meter, notes, velocity, and timing are preserved.</span>
      </div>
      <button type="button" onClick={() => inputRef.current?.click()}>
        <Upload size={13} />
        Choose MIDI
      </button>
    </div>
  );
}
