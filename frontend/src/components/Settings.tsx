import React from "react";

type SettingsProps = {
  isOpen: boolean;
  onClose: () => void;
  allowSharedKeybinds: boolean;
  onSharedKeybindsChange: (enabled: boolean) => void;
  instrument: string;
  onInstrumentChange: (instrument: string) => void;
  tempo: number;
  onTempoChange: (tempo: number) => void;
  metronomeSound: string;
  onMetronomeSoundChange: (sound: string) => void;
};

const Settings: React.FC<SettingsProps> = ({
  isOpen,
  onClose,
  allowSharedKeybinds,
  onSharedKeybindsChange,
  instrument,
  onInstrumentChange,
  tempo,
  onTempoChange,
  metronomeSound,
  onMetronomeSoundChange
}) => {
  if (!isOpen) return null;

  const instruments = [
    "acoustic_grand_piano",
    "electric_piano_1",
    "harpsichord",
    "violin",
    "cello",
    "flute",
    "clarinet",
    "trumpet",
    "trombone",
    "acoustic_guitar_nylon",
    "electric_guitar_clean",
    "electric_bass_finger",
    "synth_lead_1",
    "synth_pad_1"
  ];

  const metronomeSounds = [
    { value: "click", label: "Click" },
    { value: "beep", label: "Beep" },
    { value: "boop", label: "Boop" },
    { value: "wood", label: "Wood Block" },
    { value: "tick", label: "Tick" },
    { value: "cowbell", label: "Cowbell" }
  ];

  return (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "rgba(0, 0, 0, 0.5)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 1000
    }}>
      <div style={{
        background: "white",
        borderRadius: 12,
        padding: 24,
        minWidth: 350,
        maxHeight: "80vh",
        overflowY: "auto",
        boxShadow: "0 10px 25px rgba(0, 0, 0, 0.2)"
      }}>
        <h3 style={{
          margin: "0 0 20px 0",
          fontFamily: "'Montserrat', Arial, sans-serif",
          fontSize: 18,
          fontWeight: 600,
          color: "#1f2937"
        }}>
          Settings
        </h3>
        
        {/* Shared Keybinds */}
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 20
        }}>
          <input
            type="checkbox"
            id="sharedKeybinds"
            checked={allowSharedKeybinds}
            onChange={(e) => onSharedKeybindsChange(e.target.checked)}
            style={{
              width: 16,
              height: 16,
              cursor: "pointer"
            }}
          />
          <label
            htmlFor="sharedKeybinds"
            style={{
              fontFamily: "'Montserrat', Arial, sans-serif",
              fontSize: 14,
              color: "#374151",
              cursor: "pointer"
            }}
          >
            Allow shared keybinds (multiple notes per key)
          </label>
        </div>
        
        {/* Instrument Selection */}
        <div style={{ marginBottom: 20 }}>
          <label style={{
            display: "block",
            fontFamily: "'Montserrat', Arial, sans-serif",
            fontSize: 14,
            fontWeight: 500,
            color: "#374151",
            marginBottom: 8
          }}>
            Instrument
          </label>
          <select
            value={instrument}
            onChange={(e) => onInstrumentChange(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 12px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              fontFamily: "'Montserrat', Arial, sans-serif",
              fontSize: 14,
              cursor: "pointer"
            }}
          >
            {instruments.map((inst) => (
              <option key={inst} value={inst}>
                {inst.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </option>
            ))}
          </select>
        </div>
        
        {/* Tempo Control */}
        <div style={{ marginBottom: 24 }}>
          <label style={{
            display: "block",
            fontFamily: "'Montserrat', Arial, sans-serif",
            fontSize: 14,
            fontWeight: 500,
            color: "#374151",
            marginBottom: 8
          }}>
            Tempo: {tempo} BPM
          </label>
          <input
            type="range"
            min="60"
            max="200"
            value={tempo}
            onChange={(e) => onTempoChange(parseInt(e.target.value))}
            style={{
              width: "100%",
              cursor: "pointer"
            }}
          />
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            fontSize: 12,
            color: "#6b7280",
            fontFamily: "'Montserrat', Arial, sans-serif",
            marginTop: 4
          }}>
            <span>60</span>
            <span>200</span>
          </div>
        </div>
        
        {/* Metronome Sound Selection */}
        <div style={{ marginBottom: 20 }}>
          <label style={{
            display: "block",
            fontFamily: "'Montserrat', Arial, sans-serif",
            fontSize: 14,
            fontWeight: 500,
            color: "#374151",
            marginBottom: 8
          }}>
            Metronome Sound
          </label>
          <select
            value={metronomeSound}
            onChange={(e) => onMetronomeSoundChange(e.target.value)}
            style={{
              width: "100%",
              padding: "8px 12px",
              border: "1px solid #d1d5db",
              borderRadius: 6,
              fontFamily: "'Montserrat', Arial, sans-serif",
              fontSize: 14,
              cursor: "pointer"
            }}
          >
            {metronomeSounds.map((sound) => (
              <option key={sound.value} value={sound.value}>
                {sound.label}
              </option>
            ))}
          </select>
        </div>
        
        <div style={{
          display: "flex",
          justifyContent: "flex-end",
          gap: 8
        }}>
          <button
            onClick={onClose}
            style={{
              padding: "8px 16px",
              background: "#3b82f6",
              color: "white",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontFamily: "'Montserrat', Arial, sans-serif",
              fontSize: 14,
              fontWeight: 500
            }}
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default Settings;
