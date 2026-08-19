import {
  ChevronsLeft,
  CircleStop,
  Pause,
  Play,
  Repeat2,
  Timer,
} from "lucide-react";
import { useEffect } from "react";
import { usePlayback } from "../../hooks/usePlayback";
import { useStudioStore } from "../../store";
import { formatTime, KEY_NAMES } from "../../utils/music";
import { RecordControls } from "./RecordControls";

export function TransportBar() {
  const {
    duration,
    isPlaying,
    toggle,
    stop,
    armRecording,
    toggleRecording,
  } = usePlayback();
  const melody = useStudioStore((state) => state.melody);
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const loopEnabled = useStudioStore((state) => state.loopEnabled);
  const loopStart = useStudioStore((state) => state.loopStart);
  const loopEnd = useStudioStore((state) => state.loopEnd);
  const loopRangeCustomized = useStudioStore(
    (state) => state.loopRangeCustomized,
  );
  const recordingState = useStudioStore((state) => state.recordingState);
  const metronome = useStudioStore((state) => state.metronomeEnabled);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);
  const setLoop = useStudioStore((state) => state.setLoopEnabled);
  const setLoopRange = useStudioStore((state) => state.setLoopRange);
  const setMetronome = useStudioStore(
    (state) => state.setMetronomeEnabled,
  );
  const setTempo = useStudioStore((state) => state.setTempo);
  const setTimeSignature = useStudioStore(
    (state) => state.setTimeSignature,
  );
  const setKey = useStudioStore((state) => state.setKey);
  const signature = melody.timeSignature ?? {
    numerator: 4,
    denominator: 4,
  };
  const barLength = signature.numerator * (4 / signature.denominator);
  const positionHorizon =
    recordingState === "recording" ? currentBeat + barLength : currentBeat;
  const displayDuration = Math.max(
    duration,
    Math.ceil(positionHorizon / barLength) * barLength,
  );
  const secondsPerBeat = 60 / melody.tempo;

  useEffect(() => {
    if (
      !loopRangeCustomized &&
      (loopStart !== 0 || Math.abs(loopEnd - duration) > 0.001)
    ) {
      setLoopRange(0, duration, false);
    }
  }, [
    duration,
    loopEnd,
    loopRangeCustomized,
    loopStart,
    setLoopRange,
  ]);

  function seek(beat: number) {
    if (isPlaying) stop();
    setCurrentBeat(beat);
  }

  return (
    <div
      className={`transport-bar ${recordingState === "recording" ? "recording-live" : ""}`}
      aria-label="Transport and score settings"
    >
      <div className="transport-buttons">
        <button
          type="button"
          className="transport-icon"
          onClick={() => seek(0)}
          aria-label="Return to start"
        >
          <ChevronsLeft size={16} />
        </button>
        <button
          type="button"
          className="play-button"
          onClick={toggle}
          aria-label={isPlaying ? "Pause" : "Play"}
        >
          {isPlaying ? <Pause size={17} /> : <Play size={17} fill="currentColor" />}
        </button>
        <button
          type="button"
          className="transport-icon"
          onClick={stop}
          aria-label="Stop"
        >
          <CircleStop size={15} />
        </button>
        <RecordControls
          state={recordingState}
          onArm={armRecording}
          onToggleRecording={toggleRecording}
        />
        <button
          type="button"
          className={`transport-icon ${loopEnabled ? "active" : ""}`}
          onClick={() => setLoop(!loopEnabled)}
          aria-label="Toggle loop"
          aria-pressed={loopEnabled}
        >
          <Repeat2 size={16} />
        </button>
        <button
          type="button"
          className={`metronome-button ${metronome ? "active" : ""}`}
          onClick={() => setMetronome(!metronome)}
          aria-label="Toggle metronome"
          aria-pressed={metronome}
        >
          <Timer size={14} />
          Click
        </button>
      </div>

      <div className="transport-position">
        <span>{formatTime(currentBeat * secondsPerBeat)}</span>
        <input
          type="range"
          min={0}
          max={displayDuration}
          step={0.01}
          value={Math.min(currentBeat, displayDuration)}
          onChange={(event) => seek(event.currentTarget.valueAsNumber)}
          aria-label="Playback position"
        />
        <span>{formatTime(displayDuration * secondsPerBeat)}</span>
      </div>

      <div className="score-settings">
        <label className="setting-field tempo-field">
          <span>TEMPO</span>
          <div>
            <input
              type="number"
              min={30}
              max={240}
              value={melody.tempo}
              onChange={(event) => setTempo(event.currentTarget.valueAsNumber)}
              aria-label="Tempo in beats per minute"
            />
            <small>BPM</small>
          </div>
        </label>
        <label className="setting-field">
          <span>METER</span>
          <select
            value={`${signature.numerator}/${signature.denominator}`}
            onChange={(event) => {
              const [numerator, denominator] = event.currentTarget.value
                .split("/")
                .map(Number);
              if (numerator && denominator) {
                setTimeSignature({ numerator, denominator });
              }
            }}
          >
            <option value="4/4">4 / 4</option>
            <option value="3/4">3 / 4</option>
            <option value="6/8">6 / 8</option>
            <option value="2/4">2 / 4</option>
          </select>
        </label>
        <label className="setting-field key-field">
          <span>KEY</span>
          <select
            value={melody.key?.tonic ?? -1}
            onChange={(event) => {
              const tonic = Number(event.currentTarget.value);
              setKey(
                tonic < 0
                  ? undefined
                  : { tonic, mode: melody.key?.mode ?? "major" },
              );
            }}
          >
            <option value={-1}>Auto</option>
            {KEY_NAMES.map((name, tonic) => (
              <option value={tonic} key={name}>
                {name}
              </option>
            ))}
          </select>
          <select
            value={melody.key?.mode ?? "major"}
            onChange={(event) =>
              setKey({
                tonic: melody.key?.tonic ?? 0,
                mode: event.currentTarget.value as "major" | "minor",
              })
            }
            disabled={!melody.key}
            aria-label="Key mode"
          >
            <option value="major">Major</option>
            <option value="minor">Minor</option>
          </select>
        </label>
      </div>
    </div>
  );
}
