import { Download, FileDown, LoaderCircle, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import {
  apiClient,
  type RenderResult,
  type SynthInfo,
} from "../../api/client";
import { useStudioStore } from "../../store";

export function RenderControl() {
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [audioUrl, setAudioUrl] = useState<string>();
  const [synth, setSynth] = useState("sf2");
  const [synths, setSynths] = useState<SynthInfo[]>([]);
  const [synthsFailed, setSynthsFailed] = useState(false);
  const [timbre, setTimbre] = useState("");
  const [renderInfo, setRenderInfo] = useState<RenderResult>();
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

  useEffect(() => {
    let cancelled = false;
    void apiClient
      .getSynths()
      .then(({ synths: availableSynths }) => {
        if (cancelled) return;
        setSynths(availableSynths);
        const selected =
          availableSynths.find((item) => item.id === "sf2") ??
          availableSynths[0];
        if (selected) {
          setSynth(selected.id);
          setTimbre(selected.timbres[0] ?? "");
        }
      })
      .catch(() => {
        if (!cancelled) setSynthsFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedSynth = synths.find((item) => item.id === synth);

  async function render() {
    if (!result) return;
    setStatus("loading");
    try {
      const rendered = await apiClient.render({
        voices: result.voices,
        tempo,
        synth,
        timbre: timbre || undefined,
      });
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(URL.createObjectURL(rendered.audio));
      setRenderInfo(rendered);
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
        onChange={(event) => {
          const next = event.currentTarget.value;
          setSynth(next);
          setTimbre(
            synths.find((item) => item.id === next)?.timbres[0] ?? "",
          );
        }}
        aria-label="Render synthesizer"
        title={
          selectedSynth
            ? `${selectedSynth.description}${
                selectedSynth.reason ? ` ${selectedSynth.reason}` : ""
              }`
            : "Loading synthesizer capabilities"
        }
      >
        {synths.length === 0 && (
          <option value="sf2">
            {synthsFailed ? "SoundFont Preview" : "Loading synths…"}
          </option>
        )}
        {synths.map((item) => (
          <option value={item.id} key={item.id}>
            {item.name}
            {!item.available && item.fallback ? " · fallback" : ""}
          </option>
        ))}
      </select>
      {selectedSynth?.timbres.length ? (
        <select
          value={timbre}
          onChange={(event) => setTimbre(event.currentTarget.value)}
          aria-label="Render timbre"
          title="Installed WORLD reference timbre"
        >
          {selectedSynth.timbres.map((item) => (
            <option value={item} key={item}>
              {item}
            </option>
          ))}
        </select>
      ) : null}
      {selectedSynth && !selectedSynth.available && selectedSynth.fallback && (
        <span
          className="synth-capability fallback"
          title={`${selectedSynth.reason ?? ""} Fallback: ${selectedSynth.fallback}`}
        >
          fallback
        </span>
      )}
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
          {renderInfo && (
            <span
              className={`render-used ${renderInfo.fallback ? "fallback" : ""}`}
              title={
                renderInfo.fallback
                  ? `${renderInfo.renderer}: ${renderInfo.fallback}`
                  : `Renderer: ${renderInfo.renderer}`
              }
            >
              {renderInfo.used}
            </span>
          )}
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
