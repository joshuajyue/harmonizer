import React, { useEffect, useState, useRef } from "react";
import PianoKeyboard from "./PianoKeyboard";
import Soundfont from "soundfont-player";
import { Midi } from "@tonejs/midi";
import { useNavigate } from "react-router-dom";

const TEMPOS = [60, 80, 100, 120, 140, 160];
const NUM_BARS = 8;
const DIVISIONS_PER_BAR = 16;
const TOTAL_BOXES = NUM_BARS * DIVISIONS_PER_BAR;
const MIDI_LOW = 48; // C3
const MIDI_HIGH = 79; // G5
const PITCHES = MIDI_HIGH - MIDI_LOW + 1; // 32

const INSTRUMENTS = [
  "acoustic_grand_piano",
  "electric_piano_1",
  "electric_guitar_jazz",
  "violin",
  "cello",
  "flute",
  "trumpet",
  "clarinet",
  "synth_drum"
];

function playClick(frequency = 1000, duration = 0.05) {
  const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.type = "sine";
  osc.frequency.value = frequency;
  gain.gain.value = 0.2;
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + duration);
  osc.onended = () => ctx.close();
}

const MidiInput: React.FC = () => {
  const navigate = useNavigate();
  const [midiSupported, setMidiSupported] = useState<boolean | null>(null);
  const [tempo, setTempo] = useState<number>(120);
  const [isCountingIn, setIsCountingIn] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [count, setCount] = useState(0);
  const [notes, setNotes] = useState<{ midi: number; time: number }[]>([]);
  const startTimeRef = useRef<number | null>(null);
  const [cursorTime, setCursorTime] = useState(0);
  const [isReplaying, setIsReplaying] = useState(false);
  const replayTimeouts = useRef<number[]>([]);
  const [instrument, setInstrument] = useState("acoustic_grand_piano");

  // MIDI setup (as before)
  useEffect(() => {
    if (navigator.requestMIDIAccess) {
      setMidiSupported(true);
      navigator.requestMIDIAccess().then((midiAccess) => {
        for (let input of midiAccess.inputs.values()) {
          input.onmidimessage = () => {};
        }
      });
    } else {
      setMidiSupported(false);
    }
  }, []);

  // Count-in logic
  useEffect(() => {
    if (isCountingIn && count < 4) {
      const interval = setInterval(() => {
        setCount((c) => c + 1);
      }, (60 / tempo) * 1000);
      return () => clearInterval(interval);
    }
    if (isCountingIn && count === 4) {
      setIsCountingIn(false);
      setIsRecording(true);
      startTimeRef.current = performance.now();
      setCursorTime(0);
      setCount(0);
    }
  }, [isCountingIn, count, tempo]);

  // Stop recording after 8 bars (128 16th notes)
  useEffect(() => {
    if (isRecording && cursorTime >= getTotalDurationMs()) {
      setIsRecording(false);
      setCursorTime(getTotalDurationMs());
    }
  }, [isRecording, cursorTime]);

  // Cursor animation (smooth)
  useEffect(() => {
    let raf: number;
    const animate = () => {
      if (isRecording && startTimeRef.current !== null) {
        const elapsed = performance.now() - startTimeRef.current;
        setCursorTime(Math.min(elapsed, getTotalDurationMs()));
        raf = requestAnimationFrame(animate);
      }
    };
    if (isRecording) {
      raf = requestAnimationFrame(animate);
    }
    return () => cancelAnimationFrame(raf);
  }, [isRecording, tempo]);

  // Add this ref to track the last beat
  const lastBeatRef = useRef<number>(-1);

  // Play click on count-in and on each beat during recording
  useEffect(() => {
    if (isCountingIn && count < 4) {
      playClick(1200); // Higher pitch for count-in
    }
    if (isRecording) {
      const beatInterval = getSixteenthNoteMs() * 4; // Quarter note
      const currentBeat = Math.floor(cursorTime / beatInterval);
      if (currentBeat !== lastBeatRef.current) {
        playClick(800); // Lower pitch for recording
        lastBeatRef.current = currentBeat;
      }
    } else {
      lastBeatRef.current = -1; // Reset when not recording
    }
    // eslint-disable-next-line
  }, [count, cursorTime, isCountingIn, isRecording]);

  // Quantize function: snap to nearest 16th note
  const quantizeTime = (timeMs: number) => {
    const gridMs = getSixteenthNoteMs();
    return Math.round(timeMs / gridMs) * gridMs;
  };

  // Get duration of a 16th note in ms
  function getSixteenthNoteMs() {
    return (60 / tempo) * 1000 / 4;
  }
  // Get total duration for 8 bars in ms
  function getTotalDurationMs() {
    return getSixteenthNoteMs() * TOTAL_BOXES;
  }

  // Start recording handler
  const startRecording = () => {
    setNotes([]);
    setCount(0);
    setIsCountingIn(true);
    setCursorTime(0);
  };

  // Stop recording handler
  const stopRecording = () => {
    setIsRecording(false);
    setIsCountingIn(false);
    setCursorTime(0);
    startTimeRef.current = null;
  };

  const stopReplay = () => {
    setIsReplaying(false);
    replayTimeouts.current.forEach(timeout => clearTimeout(timeout));
    replayTimeouts.current = [];
    setCursorTime(0);
  };

  const replayNotes = async () => {
    if (!notes.length || !window.AudioContext) return;
    setIsReplaying(true);
    setCursorTime(0);

    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    // Wait for instrument to load before scheduling notes
    const inst = await Soundfont.instrument(ctx, instrument);
    let start = 0;
    let lastTime = 0;

    notes.forEach((note, i) => {
      const delay = (note.time - start) / 1000;
      // Play note
      const t1 = window.setTimeout(() => {
        console.log("Playing", note.midi, midiToNoteName(note.midi));
        inst.play(midiToNoteName(note.midi));
      }, delay * 1000);
      replayTimeouts.current.push(t1);

      // Animate cursor (optional, for smooth effect)
      const t2 = window.setTimeout(() => {
        setCursorTime(note.time);
      }, delay * 1000);
      replayTimeouts.current.push(t2);

      lastTime = delay;
    });

    // End replay after last note
    const t3 = window.setTimeout(() => {
      setIsReplaying(false);
      setCursorTime(0);
      ctx.close();
    }, (lastTime + 2) * 1000);
    replayTimeouts.current.push(t3);

    // Schedule metronome clicks for each beat during replay
    const totalDuration = getTotalDurationMs();
    const beatInterval = getSixteenthNoteMs() * 4; // quarter note
    const numBeats = Math.ceil(totalDuration / beatInterval);

    for (let i = 0; i < numBeats; i++) {
      const beatTime = i * beatInterval;
      const delay = (beatTime - (notes[0]?.time ?? 0)) / 1000;
      const t = window.setTimeout(() => {
        playClick(900); // Lower pitch for replay metronome
      }, delay * 1000);
      replayTimeouts.current.push(t);
    }
  };

  // Handle virtual piano note (quantized)
  const handleVirtualNote = (midi: number) => {
    if (isCountingIn) {
      // During count-in, treat as first beat (time = 0)
      setNotes((prev) => [...prev, { midi, time: 0 }]);
    } else if (isRecording && startTimeRef.current !== null) {
      const elapsed = performance.now() - startTimeRef.current;
      const quantized = quantizeTime(elapsed);
      setNotes((prev) => [...prev, { midi, time: quantized }]);
    }
  };
  
  const exportToMidi = () => {
  // Debug: See if this runs
  console.log("Exporting MIDI", notes);

  const midi = new Midi();
  const track = midi.addTrack();
  midi.header.setTempo(tempo);

  notes.forEach(note => {
    track.addNote({
      midi: note.midi,
      time: note.time / 1000,
      duration: getSixteenthNoteMs() / 1000,
      velocity: 0.8,
    });
  });

  const blob = new Blob([new Uint8Array(midi.toArray())], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "export.mid";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
  // --- Piano roll grid with more notes ---
  const renderPianoRoll = () => {
    const width = 1280;
    const height = 400;
    const boxWidth = width / TOTAL_BOXES;
    const boxHeight = height / PITCHES;

    // midiOrder: top row is highest note, bottom is lowest
    const midiOrder = Array.from({ length: PITCHES }, (_, i) => MIDI_HIGH - i); // Top is G5, bottom is C3

    
    return (
      <svg width={width} height={height} style={{ background: "#222", borderRadius: 8 }}>
        {/* Grid */}
        {Array.from({ length: TOTAL_BOXES + 1 }).map((_, i) => (
          <line
            key={`v${i}`}
            x1={i * boxWidth}
            y1={0}
            x2={i * boxWidth}
            y2={height}
            stroke="#444"
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
            stroke="#333"
            strokeWidth={1}
          />
        ))}
        {/* Notes */}
        {notes.map((note, idx) => {
          const col = Math.round(note.time / getSixteenthNoteMs());
          const x = col * boxWidth;
          const midiIdx = midiOrder.indexOf(note.midi);
          if (midiIdx === -1) return null; // Only show notes in C3–G5
          const y = midiIdx * boxHeight;
          return (
            <rect
              key={idx}
              x={x}
              y={y}
              width={boxWidth}
              height={boxHeight}
              fill="#4fd1c5"
              rx={3}
            />
          );
        })}
        {/* Moving cursor */}
        {isRecording && (
          <rect
            x={(cursorTime / getTotalDurationMs()) * width}
            y={0}
            width={2}
            height={height}
            fill="#ff4f4f"
            opacity={0.7}
          />
        )}
      </svg>
    );
  };

  // --- Move title to top-right ---
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        minHeight: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        alignItems: "center",
        background: "linear-gradient(135deg,rgb(11, 21, 40) 0%,rgb(54, 118, 208) 100%)",
        overflow: "auto",
      }}
    >
      <header style={{ position: "absolute", top: 24, right: 40, zIndex: 10 }}>
        <h1
          style={{
            fontFamily: "'Montserrat', sans-serif",
            fontWeight: 800,
            fontSize: "2.2rem",
            letterSpacing: "0.1em",
            color: "rgb(181, 179, 187)",
            textShadow: "0 2px 8px #b0c4de",
            margin: 0,
          }}
        >
          HarmonAIzer
        </h1>
      </header>
      {/* Controls */}
      <div style={{ display: "flex", gap: 16, margin: 24, marginTop: 80 }}>
        <label>
          Tempo:
          <input
            type="number"
            min={30}
            max={300}
            value={tempo}
            onChange={e => setTempo(Number(e.target.value))}
            style={{ width: 60, marginLeft: 8 }}
          /> BPM
        </label>
        <div style={{ marginBottom: 16 }}>
          <label>
            Instrument:&nbsp;
            <select
              value={instrument}
              onChange={e => setInstrument(e.target.value)}
              style={{ fontSize: 16, padding: "2px 8px" }}
            >
              {INSTRUMENTS.map(inst => (
                <option key={inst} value={inst}>
                  {inst.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </label>
        </div>
        <button onClick={startRecording} disabled={isCountingIn || isRecording || isReplaying}>
          {isCountingIn ? `Count-in: ${count + 1}` : isRecording ? "Recording..." : "Record"}
        </button>
        <button onClick={() => { stopRecording(); stopReplay(); }} disabled={!isRecording && !isCountingIn && !isReplaying}>
          Stop
        </button>
        <button onClick={replayNotes} disabled={isRecording || notes.length === 0 || isReplaying}>
          Replay
        </button>
        <button onClick={exportToMidi} disabled={notes.length === 0}>
          Export MIDI
        </button>
        <button
          onClick={() => navigate("/ml")}
          style={{
            background: "linear-gradient(90deg, #4fd1c5 0%, #1976d2 100%)",
            color: "#fff",
            border: "none",
            borderRadius: 10,
            padding: "10px 32px",
            fontSize: 18,
            fontWeight: 600,
            boxShadow: "0 2px 8px rgba(80,120,180,0.08)",
            cursor: "pointer",
            marginLeft: 16,
          }}
        >
          Go to ML Harmonizer
        </button>
      </div>
      {/* Piano Roll */}
      <div style={{ width: 1300, height: 620, margin: 24, overflowX: "auto" }}>{renderPianoRoll()}</div>
      {/* Piano at the bottom */}
      <div style={{ width: "100%", display: "flex", justifyContent: "center", alignItems: "flex-end", marginBottom: 40 }}>
        <PianoKeyboard onPlayNote={handleVirtualNote} lowNote={MIDI_LOW} highNote={MIDI_HIGH} />
      </div>
    </div>
  );
};

function midiToNoteName(midi: number) {
  const NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  const note = NOTES[midi % 12];
  const octave = Math.floor(midi / 12) - 1;
  return note + octave;
}




export default MidiInput;