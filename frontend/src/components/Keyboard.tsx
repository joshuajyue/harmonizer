import React from "react";

const NOTES = [
  { name: "C", midi: 60 },
  { name: "C#", midi: 61 },
  { name: "D", midi: 62 },
  { name: "D#", midi: 63 },
  { name: "E", midi: 64 },
  { name: "F", midi: 65 },
  { name: "F#", midi: 66 },
  { name: "G", midi: 67 },
  { name: "G#", midi: 68 },
  { name: "A", midi: 69 },
  { name: "A#", midi: 70 },
  { name: "B", midi: 71 },
];

type KeyboardProps = {
  onNote: (midi: number) => void;
};

const Keyboard: React.FC<KeyboardProps> = ({ onNote }) => (
  <div style={{ display: "flex", gap: 2 }}>
    {NOTES.map((note) => (
      <button
        key={note.midi}
        style={{
          width: 32,
          height: 100,
          background: note.name.includes("#") ? "#333" : "#fff",
          color: note.name.includes("#") ? "#fff" : "#000",
          border: "1px solid #888",
          marginLeft: note.name.includes("#") ? -16 : 0,
          zIndex: note.name.includes("#") ? 1 : 0,
          position: "relative",
        }}
        onClick={() => onNote(note.midi)}
      >
        {note.name}
      </button>
    ))}
  </div>
);

export default Keyboard;