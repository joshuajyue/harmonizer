import { Download, LoaderCircle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { apiClient } from "../../api/client";
import { useStudioStore } from "../../store";

export function RenderControl() {
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [audioUrl, setAudioUrl] = useState<string>();
  const [synth, setSynth] = useState("sf2");
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
