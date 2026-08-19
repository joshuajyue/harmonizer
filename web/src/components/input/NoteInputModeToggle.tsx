import { CircleDot, MousePointer2 } from "lucide-react";
import { useStudioStore, type NoteInputMode } from "../../store";

const modes: Array<{
  id: NoteInputMode;
  label: string;
  title: string;
  icon: typeof CircleDot;
}> = [
  {
    id: "record",
    label: "Record",
    title: "Preview only until transport recording captures held durations",
    icon: CircleDot,
  },
  {
    id: "step",
    label: "Step",
    title: "Place a quarter note and advance one beat on every key press",
    icon: MousePointer2,
  },
];

export function NoteInputModeToggle() {
  const mode = useStudioStore((state) => state.noteInputMode);
  const recordingState = useStudioStore((state) => state.recordingState);
  const setMode = useStudioStore((state) => state.setNoteInputMode);

  return (
    <div
      className="note-input-mode"
      role="group"
      aria-label="Keyboard and MIDI note input mode"
    >
      <span>Input</span>
      {modes.map(({ id, label, title, icon: Icon }) => (
        <button
          type="button"
          key={id}
          className={mode === id ? "active" : ""}
          onClick={() => setMode(id)}
          disabled={recordingState === "recording"}
          aria-pressed={mode === id}
          title={title}
        >
          <Icon size={10} />
          {label}
        </button>
      ))}
    </div>
  );
}
