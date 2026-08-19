import { Cable, Circle, LoaderCircle } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { usePlayback } from "../../hooks/usePlayback";
import { useStudioStore } from "../../store";
import { quantize } from "../../utils/music";

interface ActiveMidiNote {
  index: number;
  startedAt: number;
}

export function MidiDeviceInput() {
  const activeNotes = useRef(new Map<number, ActiveMidiNote>());
  const [access, setAccess] = useState<MIDIAccess>();
  const [inputs, setInputs] = useState<MIDIInput[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [status, setStatus] = useState<"idle" | "requesting" | "error">("idle");
  const [lastNote, setLastNote] = useState<number>();
  const { previewPitch } = usePlayback();
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const isPlaying = useStudioStore((state) => state.isPlaying);
  const tempo = useStudioStore((state) => state.melody.tempo);
  const snap = useStudioStore((state) => state.snap);
  const addNote = useStudioStore((state) => state.addMelodyNote);
  const updateNote = useStudioStore((state) => state.updateMelodyNote);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);

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
      const devices = Array.from(access.inputs.values());
      setInputs(devices);
      setSelectedId((id) => id || devices[0]?.id || "");
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
        previewPitch(pitch, velocity);
        setLastNote(pitch);
        const index = addNote({
          pitch,
          start: currentBeat,
          duration: Math.max(0.5, snap),
          velocity,
        });
        activeNotes.current.set(pitch, {
          index,
          startedAt: performance.now(),
        });
        if (!isPlaying) setCurrentBeat(currentBeat + Math.max(0.5, snap));
      } else if (command === 0x80 || (command === 0x90 && velocity === 0)) {
        const active = activeNotes.current.get(pitch);
        if (!active) return;
        const playedBeats =
          ((performance.now() - active.startedAt) / 1000) * (tempo / 60);
        updateNote(active.index, {
          duration: Math.max(snap, quantize(playedBeats, snap)),
        });
        activeNotes.current.delete(pitch);
      }
    };
    return () => {
      input.onmidimessage = null;
    };
  }, [
    addNote,
    currentBeat,
    inputs,
    isPlaying,
    previewPitch,
    selectedId,
    setCurrentBeat,
    snap,
    tempo,
    updateNote,
  ]);

  return (
    <div className="midi-device-input">
      <div className="input-method-icon">
        <Cable size={19} />
      </div>
      <div className="midi-device-copy">
        <strong>Web MIDI step / live capture</strong>
        <span>
          {inputs.length > 0
            ? `${inputs.length} device${inputs.length === 1 ? "" : "s"} available`
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
