import { AudioLines, RotateCcw } from "lucide-react";
import { useStudioStore } from "../../store";
import { midiToName } from "../../utils/music";

export function MelodyOctaveControl() {
  const notes = useStudioStore((state) => state.melody.notes);
  const register = useStudioStore((state) => state.transcriptionRegister);
  const shiftOctave = useStudioStore((state) => state.shiftMelodyOctave);
  const restoreSung = useStudioStore((state) => state.restoreSungRegister);

  if (!register || notes.length === 0) return null;
  const canShiftDown = notes.every((note) => note.pitch >= 12);
  const canShiftUp = notes.every((note) => note.pitch <= 115);
  const current = register.currentOctaveShift;
  const canRestore = notes.every((note) => {
    const pitch = note.pitch - current * 12;
    return pitch >= 0 && pitch <= 127;
  });
  const detected =
    register.detectedMedianPitch === undefined
      ? undefined
      : midiToName(Math.round(register.detectedMedianPitch));
  const title = [
    detected ? `Detected near ${detected}.` : undefined,
    register.detectedOctaveShift === 0
      ? "The backend kept the sung register."
      : `The backend applied ${signedOctaves(register.detectedOctaveShift)}.`,
    current === 0
      ? "Currently at the true sung register."
      : `Currently ${signedOctaves(current)} from the sung register.`,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className="melody-octave-control" title={title}>
      <AudioLines size={12} />
      <span>{current === 0 ? "Sung" : signedOctaves(current)}</span>
      <button
        type="button"
        onClick={() => shiftOctave(-1)}
        disabled={!canShiftDown}
        aria-label="Shift entire melody down one octave"
      >
        −8va
      </button>
      <button
        type="button"
        onClick={restoreSung}
        disabled={current === 0 || !canRestore}
        aria-label="Restore true sung register"
      >
        <RotateCcw size={10} />
        Sung
      </button>
      <button
        type="button"
        onClick={() => shiftOctave(1)}
        disabled={!canShiftUp}
        aria-label="Shift entire melody up one octave"
      >
        +8va
      </button>
    </div>
  );
}

function signedOctaves(octaves: number) {
  const amount = Math.abs(octaves);
  return `${octaves > 0 ? "+" : "−"}${amount} oct`;
}
