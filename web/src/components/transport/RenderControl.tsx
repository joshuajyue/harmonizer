import { Download, FileDown, LoaderCircle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { apiClient } from "../../api/client";
import { useStudioStore } from "../../store";

export function RenderControl() {
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [audioUrl, setAudioUrl] = useState<string>();
  const [synth, setSynth] = useState("sf2");
  const [midiLoading, setMidiLoading] = useState(false);
  const [midiError, setMidiError] = useState(false);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const result = useStudioStore((state) => state.slots[activeSlot].result);
  const tempo = useStudioStore((state) => state.melody.tempo);

  useEffect(
    () => () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    },
    [audioUrl],
  );

  async function render() {
    if (!result) return;
    setStatus("loading");
    try {
      const blob = await apiClient.render({
        voices: result.voices,
        tempo,
        synth,
        timbre: synth === "ddsp" ? "choral-neutral" : undefined,
      });
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(blob));
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }

  async function exportMidi() {
    if (!result) return;
    setMidiLoading(true);
    setMidiError(false);
    try {
      const blob = await apiClient.exportMidi(result, tempo);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `harmonaizer-${activeSlot}.mid`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
    } catch {
      setMidiError(true);
    } finally {
      setMidiLoading(false);
    }
  }

  return (
    <div className="render-control">
      <select
        value={synth}
        onChange={(event) => setSynth(event.currentTarget.value)}
        aria-label="Render synthesizer"
      >
        <option value="sf2">Studio piano</option>
        <option value="ddsp">Neural choir</option>
      </select>
      <button
        type="button"
        className="render-button"
        onClick={() => void render()}
        disabled={!result || status === "loading"}
        title={status === "error" ? "Render failed — retry" : "Render audio"}
      >
        {status === "loading" ? (
          <LoaderCircle size={13} className="spin" />
        ) : (
          <Sparkles size={13} />
        )}
        Render {activeSlot}
      </button>
      <button
        type="button"
        className="midi-export-button"
        onClick={() => void exportMidi()}
        disabled={!result || midiLoading}
        title={
          midiError
            ? "MIDI export failed — retry"
            : `Export result ${activeSlot} as MIDI`
        }
      >
        {midiLoading ? (
          <LoaderCircle size={13} className="spin" />
        ) : (
          <FileDown size={13} />
        )}
        MIDI
      </button>
      {audioUrl && (
        <>
          <audio className="rendered-audio" src={audioUrl} controls />
          <a
            className="download-render"
            href={audioUrl}
            download={`harmonaizer-${activeSlot}.wav`}
            aria-label="Download rendered WAV"
          >
            <Download size={14} />
          </a>
        </>
      )}
    </div>
  );
}
