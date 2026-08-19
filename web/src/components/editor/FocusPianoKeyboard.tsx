import { audioScheduler } from "../../audio/AudioScheduler";
import type { FocusedLane } from "../../store";
import { midiToName } from "../../utils/music";
import {
  pitchRange,
  pitchRowRect,
  RULER_HEIGHT,
  type RollLayout,
} from "./rollGeometry";

const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);

export function FocusPianoKeyboard({
  lane,
  layout,
}: {
  lane: FocusedLane;
  layout: RollLayout;
}) {
  const range = pitchRange(lane, layout);
  const pitches = Array.from(
    { length: range.max - range.min + 1 },
    (_value, index) => range.max - index,
  );

  return (
    <div
      className="focus-piano-keyboard"
      style={{ height: layout.laneHeight }}
      role="group"
      aria-label={`${lane} pitch keyboard`}
    >
      {pitches.map((pitch) => {
        const row = pitchRowRect(lane, pitch, layout);
        const black = BLACK_KEYS.has(pitch % 12);
        const noteName = midiToName(pitch);
        return (
          <button
            type="button"
            key={pitch}
            className={`focus-piano-key ${black ? "black" : "white"}`}
            style={{
              top: row.y - RULER_HEIGHT,
              height: row.height,
            }}
            onClick={() => audioScheduler.preview(pitch)}
            aria-label={`Audition ${noteName}`}
            title={noteName}
          >
            {pitch % 12 === 0 && <span>{noteName}</span>}
          </button>
        );
      })}
    </div>
  );
}
