import { Trash2, X } from "lucide-react";
import type { Note } from "../../../../contracts/types";
import {
  useStudioStore,
  type SelectedNote,
} from "../../store";
import { midiToName } from "../../utils/music";
import { QuantizeSelectionButton } from "./QuantizeSelectionButton";

export function SelectedNoteInspector() {
  const selections = useStudioStore((state) => state.selectedNotes);
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const snap = useStudioStore((state) => state.snap);
  const clearSelection = useStudioStore((state) => state.clearSelection);
  const deleteSelection = useStudioStore(
    (state) => state.deleteSelectedNotes,
  );
  const transposeSelection = useStudioStore(
    (state) => state.transposeSelectedNotes,
  );
  const setSelectionDuration = useStudioStore(
    (state) => state.setSelectedNotesDuration,
  );
  const quantizeStarts = useStudioStore(
    (state) => state.quantizeSelectedNoteStarts,
  );
  const updateMelody = useStudioStore((state) => state.updateMelodyNote);
  const updateVoice = useStudioStore((state) => state.updateVoiceNote);

  const selected = selections.flatMap((selection) => {
    const note = resolveNote(selection, melody.notes, slots);
    return note ? [{ selection, note }] : [];
  });

  if (selected.length === 0) {
    return (
      <div className="note-inspector-empty">
        Drag empty space to select · ruler drag sets cycle range
      </div>
    );
  }

  if (selected.length > 1) {
    const starts = selected.map(({ note }) => note.start);
    const ends = selected.map(({ note }) => note.start + note.duration);
    const durations = selected.map(({ note }) => note.duration);
    const sharedDuration = durations.every(
      (duration) => Math.abs(duration - durations[0]) < 0.0001,
    )
      ? durations[0]
      : undefined;
    const lanes = new Set(
      selected.map(({ selection }) =>
        selection.source === "melody" ? "melody" : selection.voice,
      ),
    );

    return (
      <div className="note-inspector multi-note-inspector">
        <div className="note-inspector-name">
          <span>SELECTION</span>
          <strong>{selected.length} notes</strong>
        </div>
        <span className="selection-summary">
          {lanes.size} tracks · {Math.min(...starts).toFixed(2)}–
          {Math.max(...ends).toFixed(2)} beats
        </span>
        <div className="transpose-cluster" aria-label="Transpose selection">
          <span>Transpose</span>
          {[-12, -1, 1, 12].map((semitones) => (
            <button
              type="button"
              key={semitones}
              onClick={() => transposeSelection(semitones)}
              aria-label={`Transpose ${semitones} semitones`}
            >
              {semitones > 0 ? `+${semitones}` : semitones}
            </button>
          ))}
        </div>
        <QuantizeSelectionButton onClick={quantizeStarts} />
        {sharedDuration !== undefined ? (
          <NumberField
            label="Length"
            value={sharedDuration}
            min={snap}
            step={snap}
            onChange={(duration) => setSelectionDuration(duration)}
          />
        ) : (
          <span className="mixed-value">Mixed lengths</span>
        )}
        <SelectionButtons
          count={selected.length}
          onDelete={deleteSelection}
          onClear={clearSelection}
        />
      </div>
    );
  }

  const { selection, note } = selected[0];
  const label =
    selection.source === "melody"
      ? "Melody"
      : `${selection.slot} · ${selection.voice}`;
  const update = (patch: Partial<Note>) => {
    if (selection.source === "melody") {
      updateMelody(selection.index, patch);
    } else if (selection.slot && selection.voice) {
      updateVoice(selection.slot, selection.voice, selection.index, patch);
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
      <QuantizeSelectionButton onClick={quantizeStarts} />
      <SelectionButtons
        count={1}
        onDelete={deleteSelection}
        onClear={clearSelection}
      />
    </div>
  );
}

function resolveNote(
  selection: SelectedNote,
  melodyNotes: Note[],
  slots: ReturnType<typeof useStudioStore.getState>["slots"],
) {
  if (selection.source === "melody") return melodyNotes[selection.index];
  if (!selection.slot || !selection.voice) return undefined;
  return slots[selection.slot].result?.voices
    .find((voice) => voice.name === selection.voice)
    ?.notes.at(selection.index);
}

function SelectionButtons({
  count,
  onDelete,
  onClear,
}: {
  count: number;
  onDelete: () => void;
  onClear: () => void;
}) {
  return (
    <>
      <button
        type="button"
        className="icon-button danger-hover"
        onClick={onDelete}
        aria-label={`Delete ${count === 1 ? "selected note" : `${count} selected notes`}`}
        title="Delete selection"
      >
        <Trash2 size={15} />
      </button>
      <button
        type="button"
        className="icon-button"
        onClick={onClear}
        aria-label="Clear selection"
      >
        <X size={15} />
      </button>
    </>
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
        onChange={(event) => {
          const next = event.currentTarget.valueAsNumber;
          if (Number.isFinite(next)) onChange(next);
        }}
      />
    </label>
  );
}
