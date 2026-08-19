import { Trash2, X } from "lucide-react";
import type { Note } from "../../../../contracts/types";
import { useStudioStore } from "../../store";
import { midiToName } from "../../utils/music";

export function SelectedNoteInspector() {
  const selected = useStudioStore((state) => state.selectedNote);
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const snap = useStudioStore((state) => state.snap);
  const setSelected = useStudioStore((state) => state.setSelectedNote);
  const updateMelody = useStudioStore((state) => state.updateMelodyNote);
  const deleteMelody = useStudioStore((state) => state.deleteMelodyNote);
  const updateVoice = useStudioStore((state) => state.updateVoiceNote);
  const deleteVoice = useStudioStore((state) => state.deleteVoiceNote);

  if (!selected) {
    return (
      <div className="note-inspector-empty">
        Select a note to edit exact pitch, position, and duration
      </div>
    );
  }

  let note: Note | undefined;
  let label = "Melody";
  if (selected.source === "melody") {
    note = melody.notes[selected.index];
  } else if (selected.slot && selected.voice) {
    note = slots[selected.slot].result?.voices
      .find((voice) => voice.name === selected.voice)
      ?.notes.at(selected.index);
    label = `${selected.slot} · ${selected.voice}`;
  }
  if (!note) return null;

  const update = (patch: Partial<Note>) => {
    if (selected.source === "melody") {
      updateMelody(selected.index, patch);
    } else if (selected.slot && selected.voice) {
      updateVoice(selected.slot, selected.voice, selected.index, patch);
    }
  };

  const remove = () => {
    if (selected.source === "melody") {
      deleteMelody(selected.index);
    } else if (selected.slot && selected.voice) {
      deleteVoice(selected.slot, selected.voice, selected.index);
    }
  };

  return (
    <div className="note-inspector">
      <div className="note-inspector-name">
        <span>{label}</span>
        <strong>{midiToName(note.pitch)}</strong>
      </div>
      <NumberField
        label="Pitch"
        value={note.pitch}
        min={0}
        max={127}
        step={1}
        onChange={(pitch) => update({ pitch: Math.round(pitch) })}
      />
      <NumberField
        label="Start"
        value={note.start}
        min={0}
        step={snap}
        onChange={(start) => update({ start: Math.max(0, start) })}
      />
      <NumberField
        label="Length"
        value={note.duration}
        min={snap}
        step={snap}
        onChange={(duration) => update({ duration: Math.max(snap, duration) })}
      />
      <button
        type="button"
        className="icon-button danger-hover"
        onClick={remove}
        aria-label="Delete selected note"
        title="Delete note"
      >
        <Trash2 size={15} />
      </button>
      <button
        type="button"
        className="icon-button"
        onClick={() => setSelected(undefined)}
        aria-label="Clear selection"
      >
        <X size={15} />
      </button>
    </div>
  );
}

interface NumberFieldProps {
  label: string;
  value: number;
  min: number;
  max?: number;
  step: number;
  onChange: (value: number) => void;
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: NumberFieldProps) {
  return (
    <label className="compact-number">
      <span>{label}</span>
      <input
        type="number"
        value={Number(value.toFixed(3))}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(event.currentTarget.valueAsNumber)}
      />
    </label>
  );
}
