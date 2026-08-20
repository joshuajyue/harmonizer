import {
  ChevronDown,
  ChevronUp,
  FileMusic,
  Keyboard,
  Mic,
  PlugZap,
  RotateCcw,
  Trash2,
} from "lucide-react";
import { useState } from "react";
import { apiClient } from "../../api/client";
import {
  useStudioStore,
  type InputTab,
} from "../../store";
import { MidiDeviceInput } from "./MidiDeviceInput";
import { MidiFileInput } from "./MidiFileInput";
import { MicrophoneInput } from "./MicrophoneInput";
import { NoteInputModeToggle } from "./NoteInputModeToggle";
import { VirtualKeyboard } from "./VirtualKeyboard";

const tabs: Array<{
  id: InputTab;
  label: string;
  icon: typeof Keyboard;
}> = [
  { id: "piano", label: "Piano", icon: Keyboard },
  { id: "midi", label: "MIDI device", icon: PlugZap },
  { id: "file", label: "MIDI file", icon: FileMusic },
  { id: "microphone", label: "Microphone", icon: Mic },
];

export function InputDock() {
  const [restoring, setRestoring] = useState(false);
  const [restoreError, setRestoreError] = useState(false);
  const tab = useStudioStore((state) => state.inputTab);
  const open = useStudioStore((state) => state.inputDockOpen);
  const notes = useStudioStore((state) => state.melody.notes.length);
  const setTab = useStudioStore((state) => state.setInputTab);
  const setOpen = useStudioStore((state) => state.setInputDockOpen);
  const clearMelody = useStudioStore((state) => state.clearMelody);
  const replaceMelody = useStudioStore((state) => state.replaceMelody);

  async function restoreExample() {
    setRestoring(true);
    setRestoreError(false);
    try {
      replaceMelody(
        await apiClient.getExampleMelody(),
        "Canonical eight-bar study",
      );
    } catch {
      setRestoreError(true);
    } finally {
      setRestoring(false);
    }
  }

  return (
    <section className={`input-dock ${open ? "open" : "closed"}`}>
      <div className="input-dock-header">
        <div className="input-tabs" role="tablist" aria-label="Melody input">
          {tabs.map(({ id, label, icon: Icon }) => (
            <button
              type="button"
              key={id}
              role="tab"
              aria-selected={tab === id}
              className={tab === id ? "active" : ""}
              onClick={() => {
                setTab(id);
                setOpen(true);
              }}
            >
              <Icon size={12} />
              {label}
            </button>
          ))}
        </div>
        <NoteInputModeToggle />
        <div className="melody-summary">
          <span>{notes} source notes</span>
          {notes === 0 && apiClient.isMock ? (
            <button
              type="button"
              onClick={() => void restoreExample()}
              disabled={restoring}
              title={
                restoreError
                  ? "Could not load the canonical example"
                  : "Restore canonical example"
              }
            >
              <RotateCcw size={12} />
              {restoring ? "Loading" : "Example"}
            </button>
          ) : (
            <button
              type="button"
              onClick={clearMelody}
              title="Clear melody"
            >
              <Trash2 size={12} />
              Clear
            </button>
          )}
          <button
            type="button"
            className="dock-collapse"
            onClick={() => setOpen(!open)}
            aria-label={open ? "Collapse input dock" : "Expand input dock"}
          >
            {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
          </button>
        </div>
      </div>
      {open && (
        <div className="input-dock-body" role="tabpanel">
          {tab === "piano" && <VirtualKeyboard />}
          {tab === "midi" && <MidiDeviceInput />}
          {tab === "file" && <MidiFileInput />}
          {tab === "microphone" && <MicrophoneInput />}
        </div>
      )}
    </section>
  );
}
