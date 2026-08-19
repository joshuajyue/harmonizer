import { useCallback, useEffect, useRef, useState } from "react";
import { noteCapture } from "../../capture/NoteCapture";
import { MUSICAL_TYPING_OFFSETS } from "../../input/musicalTyping";
import { useStudioStore } from "../../store";
import { midiToName } from "../../utils/music";

const START_PITCH = 60;
const END_PITCH = 84;
const WHITE_CLASSES = new Set([0, 2, 4, 5, 7, 9, 11]);
interface PianoKey {
  pitch: number;
  white: boolean;
  whiteIndex: number;
}

const keys: PianoKey[] = [];
let whiteCount = 0;
for (let pitch = START_PITCH; pitch <= END_PITCH; pitch += 1) {
  const white = WHITE_CLASSES.has(pitch % 12);
  keys.push({ pitch, white, whiteIndex: whiteCount });
  if (white) whiteCount += 1;
}

export function VirtualKeyboard() {
  const pointerNotes = useRef(new Map<number, number>());
  const keyNotes = useRef(new Map<string, number>());
  const [activePitches, setActivePitches] = useState<Set<number>>(
    () => new Set(),
  );
  const recordingState = useStudioStore((state) => state.recordingState);
  const noteInputMode = useStudioStore((state) => state.noteInputMode);

  const setActive = useCallback((pitch: number, active: boolean) => {
    setActivePitches((current) => {
      const next = new Set(current);
      if (active) next.add(pitch);
      else next.delete(pitch);
      return next;
    });
  }, []);

  const noteOn = useCallback((pitch: number, velocity = 94) => {
    noteCapture.noteOn(pitch, velocity);
    setActive(pitch, true);
  }, [setActive]);

  const noteOff = useCallback((pitch: number) => {
    noteCapture.noteOff(pitch);
    setActive(pitch, false);
  }, [setActive]);

  function pointerDown(
    event: React.PointerEvent<HTMLButtonElement>,
    pitch: number,
  ) {
    event.preventDefault();
    pointerNotes.current.set(event.pointerId, pitch);
    event.currentTarget.setPointerCapture(event.pointerId);
    noteOn(pitch);
  }

  function pointerUp(
    event: React.PointerEvent<HTMLButtonElement>,
    pitch: number,
  ) {
    if (pointerNotes.current.get(event.pointerId) !== pitch) return;
    pointerNotes.current.delete(event.pointerId);
    noteOff(pitch);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.repeat ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey ||
        event.shiftKey ||
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLSelectElement ||
        event.target instanceof HTMLTextAreaElement ||
        (event.target instanceof HTMLElement && event.target.isContentEditable)
      ) {
        return;
      }
      const key = event.key.toLowerCase();
      const index = MUSICAL_TYPING_OFFSETS.get(key);
      if (index === undefined || keyNotes.current.has(key)) return;
      event.preventDefault();
      const pitch = START_PITCH + index;
      keyNotes.current.set(key, pitch);
      noteOn(pitch);
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      const pitch = keyNotes.current.get(key);
      if (pitch === undefined) return;
      event.preventDefault();
      keyNotes.current.delete(key);
      noteOff(pitch);
    };
    const releaseAll = () => {
      pointerNotes.current.clear();
      keyNotes.current.clear();
      setActivePitches(new Set());
      noteCapture.releaseAll();
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", releaseAll);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", releaseAll);
      releaseAll();
    };
  }, [noteOff, noteOn]);

  const whiteWidth = 100 / whiteCount;
  const blackWidth = whiteWidth * 0.62;
  const statusCopy =
    noteInputMode === "step"
      ? "Place input · quarter note per press"
      : recordingState === "recording"
      ? "Recording raw note-on / note-off timing"
      : recordingState === "counting"
        ? "Count-in · preview only until capture starts"
      : recordingState === "armed"
        ? "Armed · preview only until recording starts"
        : "Preview only · arm the transport to capture";

  return (
    <div className="virtual-keyboard-panel">
      <div className="keyboard-copy">
        <strong>Live keyboard</strong>
        <span>{statusCopy} · keys A–;</span>
      </div>
      <div
        className={`virtual-keyboard ${recordingState === "recording" ? "recording" : ""} ${noteInputMode === "step" ? "step-entry" : ""}`}
        role="group"
        aria-label="Virtual piano keyboard"
      >
        {keys
          .filter((key) => key.white)
          .map((key) => (
            <PianoKeyButton
              key={key.pitch}
              pianoKey={key}
              active={activePitches.has(key.pitch)}
              style={{
                left: `${key.whiteIndex * whiteWidth}%`,
                width: `${whiteWidth}%`,
              }}
              onPointerDown={pointerDown}
              onPointerEnd={pointerUp}
            />
          ))}
        {keys
          .filter((key) => !key.white)
          .map((key) => (
            <PianoKeyButton
              key={key.pitch}
              pianoKey={key}
              active={activePitches.has(key.pitch)}
              style={{
                left: `${key.whiteIndex * whiteWidth - blackWidth / 2}%`,
                width: `${blackWidth}%`,
              }}
              onPointerDown={pointerDown}
              onPointerEnd={pointerUp}
            />
          ))}
      </div>
    </div>
  );
}

function PianoKeyButton({
  pianoKey,
  active,
  style,
  onPointerDown,
  onPointerEnd,
}: {
  pianoKey: PianoKey;
  active: boolean;
  style: React.CSSProperties;
  onPointerDown: (
    event: React.PointerEvent<HTMLButtonElement>,
    pitch: number,
  ) => void;
  onPointerEnd: (
    event: React.PointerEvent<HTMLButtonElement>,
    pitch: number,
  ) => void;
}) {
  const className = `piano-key ${pianoKey.white ? "white-key" : "black-key"} ${active ? "active" : ""}`;
  return (
    <button
      type="button"
      className={className}
      style={style}
      onPointerDown={(event) => onPointerDown(event, pianoKey.pitch)}
      onPointerUp={(event) => onPointerEnd(event, pianoKey.pitch)}
      onPointerLeave={(event) => onPointerEnd(event, pianoKey.pitch)}
      onPointerCancel={(event) => onPointerEnd(event, pianoKey.pitch)}
      onLostPointerCapture={(event) => onPointerEnd(event, pianoKey.pitch)}
      aria-label={`Play ${midiToName(pianoKey.pitch)}`}
    >
      {pianoKey.white && pianoKey.pitch % 12 === 0 && (
        <span>{midiToName(pianoKey.pitch)}</span>
      )}
    </button>
  );
}
