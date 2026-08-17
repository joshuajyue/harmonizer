// frontend/src/components/PianoRoll.tsx
import React from "react";
import {
  DIVISIONS_PER_BAR,
  MIDI_HIGH,
  PITCHES,
  TOTAL_BOXES,
  getSixteenthNoteMs,
  getTotalDurationMs,
  type MidiNote,
} from "../musicConstants";

interface PianoRollProps {
  notes: MidiNote[];
  cursorTime: number;
  isRecording: boolean;
  availableHeight: number;
  tempo: number;
}

/** Renders the recorded notes and a moving cursor as an SVG piano-roll grid. */
const PianoRoll: React.FC<PianoRollProps> = ({ notes, cursorTime, isRecording, availableHeight, tempo }) => {
  const width = 1280;
  const height = Math.max(200, availableHeight - 20); // Reduced padding for a tighter fit
  const boxWidth = width / TOTAL_BOXES;
  const boxHeight = height / PITCHES;
  const sixteenthNoteMs = getSixteenthNoteMs(tempo);
  const totalDurationMs = getTotalDurationMs(tempo);

  // Top row is the highest note (G5), bottom row is the lowest (C3).
  const midiOrder = Array.from({ length: PITCHES }, (_, i) => MIDI_HIGH - i);

  return (
    <div style={{
      borderRadius: 12,
      padding: 4,
      background: "linear-gradient(135deg, rgba(79, 209, 197, 0.1) 0%, rgba(25, 118, 210, 0.1) 100%)",
      boxShadow: "0 0 20px rgba(79, 209, 197, 0.3), inset 0 0 20px rgba(25, 118, 210, 0.1)"
    }}>
      <svg width={width} height={height} style={{
        background: "linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f0f23 100%)",
        borderRadius: 8,
        filter: "drop-shadow(0 0 10px rgba(79, 209, 197, 0.2))"
      }}>
        {/* Grid */}
        {Array.from({ length: TOTAL_BOXES + 1 }).map((_, i) => (
          <line
            key={`v${i}`}
            x1={i * boxWidth}
            y1={0}
            x2={i * boxWidth}
            y2={height}
            stroke={i % DIVISIONS_PER_BAR === 0 ? "rgba(79, 209, 197, 0.6)" : "rgba(79, 209, 197, 0.2)"}
            strokeWidth={i % DIVISIONS_PER_BAR === 0 ? 2 : 1}
          />
        ))}
        {Array.from({ length: PITCHES + 1 }).map((_, i) => (
          <line
            key={`h${i}`}
            x1={0}
            y1={i * boxHeight}
            x2={width}
            y2={i * boxHeight}
            stroke="rgba(25, 118, 210, 0.3)"
            strokeWidth={1}
          />
        ))}

        {/* Notes, shown as rectangles sized to their duration */}
        {notes.map((note, idx) => {
          const startCol = Math.round(note.startTime / sixteenthNoteMs);
          const duration = note.duration || sixteenthNoteMs;
          const durationCols = Math.max(1, Math.round(duration / sixteenthNoteMs));

          const x = startCol * boxWidth;
          const noteWidth = durationCols * boxWidth - 2; // Small gap between notes

          const midiIdx = midiOrder.indexOf(note.midi);
          if (midiIdx === -1) return null; // Outside the visible C3-G5 range
          const y = midiIdx * boxHeight;

          return (
            <rect
              key={idx}
              x={x}
              y={y + 1}
              width={noteWidth}
              height={boxHeight - 2}
              fill="url(#noteGradient)"
              rx={3}
              opacity={0.9}
              filter="drop-shadow(0 0 8px rgba(79, 209, 197, 0.6))"
            />
          );
        })}

        {/* Moving playback/record cursor */}
        {isRecording && (
          <rect
            x={(cursorTime / totalDurationMs) * width}
            y={0}
            width={3}
            height={height}
            fill="url(#cursorGradient)"
            opacity={0.9}
            filter="drop-shadow(0 0 10px rgba(255, 79, 79, 0.8))"
          />
        )}

        <defs>
          <linearGradient id="noteGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#4fd1c5" />
            <stop offset="50%" stopColor="#38bdf8" />
            <stop offset="100%" stopColor="#1976d2" />
          </linearGradient>
          <linearGradient id="cursorGradient" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#ff6b6b" />
            <stop offset="50%" stopColor="#ff8e8e" />
            <stop offset="100%" stopColor="#ff4f4f" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
};

export default PianoRoll;
