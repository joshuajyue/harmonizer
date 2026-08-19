import { useEffect } from "react";
import { usePlayback } from "../../hooks/usePlayback";
import { useStudioStore } from "../../store";
import { midiToName } from "../../utils/music";

const START_PITCH = 60;
const END_PITCH = 84;
const WHITE_CLASSES = new Set([0, 2, 4, 5, 7, 9, 11]);
const KEY_BINDINGS = [
  "a",
  "w",
  "s",
  "e",
  "d",
  "f",
  "t",
  "g",
  "y",
  "h",
  "u",
  "j",
  "k",
  "o",
  "l",
  "p",
  ";",
];

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
  const { previewPitch } = usePlayback();
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const isPlaying = useStudioStore((state) => state.isPlaying);
  const snap = useStudioStore((state) => state.snap);
  const addNote = useStudioStore((state) => state.addMelodyNote);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);
  const setSelectedNote = useStudioStore((state) => state.setSelectedNote);

  function capture(pitch: number, velocity = 94) {
    previewPitch(pitch, velocity);
    const duration = Math.max(0.5, snap);
    const index = addNote({
      pitch,
      start: currentBeat,
      duration,
      velocity,
    });
    setSelectedNote({ source: "melody", index });
    if (!isPlaying) setCurrentBeat(currentBeat + duration);
  }

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        event.repeat ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey ||
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLSelectElement ||
        event.target instanceof HTMLTextAreaElement
      ) {
        return;
      }
      const index = KEY_BINDINGS.indexOf(event.key.toLowerCase());
      if (index < 0) return;
      event.preventDefault();
      capture(START_PITCH + index);
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  });

  const whiteWidth = 100 / whiteCount;
  const blackWidth = whiteWidth * 0.62;

  return (
    <div className="virtual-keyboard-panel">
      <div className="keyboard-copy">
        <strong>Step input</strong>
        <span>Click keys or play A–; · cursor advances when stopped</span>
      </div>
      <div
        className="virtual-keyboard"
        role="group"
        aria-label="Virtual piano keyboard"
      >
        {keys
          .filter((key) => key.white)
          .map((key) => (
            <button
              type="button"
              key={key.pitch}
              className="piano-key white-key"
              style={{
                left: `${key.whiteIndex * whiteWidth}%`,
                width: `${whiteWidth}%`,
              }}
              onPointerDown={() => capture(key.pitch)}
              aria-label={`Add ${midiToName(key.pitch)}`}
            >
              {key.pitch % 12 === 0 && <span>{midiToName(key.pitch)}</span>}
            </button>
          ))}
        {keys
          .filter((key) => !key.white)
          .map((key) => (
            <button
              type="button"
              key={key.pitch}
              className="piano-key black-key"
              style={{
                left: `${key.whiteIndex * whiteWidth - blackWidth / 2}%`,
                width: `${blackWidth}%`,
              }}
              onPointerDown={() => capture(key.pitch)}
              aria-label={`Add ${midiToName(key.pitch)}`}
            />
          ))}
      </div>
    </div>
  );
}
