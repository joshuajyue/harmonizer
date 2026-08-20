import { FileMusic, LoaderCircle, Upload } from "lucide-react";
import { useRef, useState } from "react";
import { apiClient } from "../../api/client";
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
      const { melody, notice } = await apiClient.importMidi(
        file,
        currentMelody.tempo,
      );
      replaceMelody(melody, file.name.replace(/\.(mid|midi)$/i, ""));
      setMessage(
        `${file.name} · ${melody.notes.length} melody notes${
          notice ? ` · ${notice}` : ""
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
