import { Cable, Circle, LoaderCircle } from "lucide-react";
import { useEffect, useState } from "react";
import { noteCapture } from "../../capture/NoteCapture";
import { useStudioStore } from "../../store";

export function MidiDeviceInput() {
  const [access, setAccess] = useState<MIDIAccess>();
  const [inputs, setInputs] = useState<MIDIInput[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [status, setStatus] = useState<"idle" | "requesting" | "error">("idle");
  const [lastNote, setLastNote] = useState<number>();
  const recordingState = useStudioStore((state) => state.recordingState);
  const noteInputMode = useStudioStore((state) => state.noteInputMode);

  async function requestAccess() {
    if (!navigator.requestMIDIAccess) {
      setStatus("error");
      return;
    }
    setStatus("requesting");
    try {
      const midiAccess = await navigator.requestMIDIAccess();
      setAccess(midiAccess);
      const devices = Array.from(midiAccess.inputs.values());
      setInputs(devices);
      setSelectedId(devices[0]?.id ?? "");
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  useEffect(() => {
    if (!access) return;
    const refresh = () => {
      noteCapture.releaseAll();
      const devices = Array.from(access.inputs.values());
      setInputs(devices);
      setSelectedId((id) => {
        if (id && devices.some((device) => device.id === id)) return id;
        return devices[0]?.id ?? "";
      });
    };
    access.onstatechange = refresh;
    return () => {
      access.onstatechange = null;
    };
  }, [access]);

  useEffect(() => {
    const input = inputs.find((device) => device.id === selectedId);
    if (!input) return;
    input.onmidimessage = (event) => {
      const [statusByte = 0, pitch = 0, velocity = 0] = event.data ?? [];
      const command = statusByte & 0xf0;
      if (command === 0x90 && velocity > 0) {
        noteCapture.noteOn(pitch, velocity);
        setLastNote(pitch);
      } else if (command === 0x80 || (command === 0x90 && velocity === 0)) {
        noteCapture.noteOff(pitch);
      }
    };
    return () => {
      input.onmidimessage = null;
      noteCapture.releaseAll();
    };
  }, [inputs, selectedId]);

  return (
    <div className="midi-device-input">
      <div className="input-method-icon">
        <Cable size={19} />
      </div>
      <div className="midi-device-copy">
        <strong>Web MIDI performance input</strong>
        <span>
          {inputs.length > 0
            ? noteInputMode === "step"
              ? "Place input · quarter note per note-on."
              : recordingState === "recording"
              ? "Recording raw timing and MIDI velocity."
              : recordingState === "counting"
                ? "Count-in · preview only until capture starts."
              : `${inputs.length} device${inputs.length === 1 ? "" : "s"} · preview only`
            : "Grant access, then play your connected keyboard."}
        </span>
      </div>
      {access ? (
        <select
          value={selectedId}
          onChange={(event) => setSelectedId(event.currentTarget.value)}
          aria-label="MIDI input device"
        >
          {inputs.map((input) => (
            <option value={input.id} key={input.id}>
              {input.name ?? "MIDI input"}
            </option>
          ))}
        </select>
      ) : (
        <button
          type="button"
          onClick={() => void requestAccess()}
          disabled={status === "requesting"}
        >
          {status === "requesting" ? (
            <LoaderCircle size={13} className="spin" />
          ) : (
            <Cable size={13} />
          )}
          Enable MIDI
        </button>
      )}
      <div className={`midi-activity ${lastNote !== undefined ? "active" : ""}`}>
        <Circle size={8} fill="currentColor" />
        {lastNote !== undefined ? `Note ${lastNote}` : "Waiting"}
      </div>
      {status === "error" && (
        <small className="midi-error">Web MIDI is unavailable or blocked.</small>
      )}
    </div>
  );
}
