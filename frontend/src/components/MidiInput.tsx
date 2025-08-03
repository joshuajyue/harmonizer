import React, { useEffect, useState, useRef } from "react";
import PianoKeyboard from "./PianoKeyboard";
import HarmonizerPanel from "./HarmonizerPanel";
import Soundfont from "soundfont-player";
import { Midi } from "@tonejs/midi";

interface MidiNote {
  midi: number;
  startTime: number;
  endTime?: number;
  duration?: number;
}

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
  const [tempo, setTempo] = useState<number>(120);
  const [isCountingIn, setIsCountingIn] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [count, setCount] = useState(0);
  const [notes, setNotes] = useState<MidiNote[]>([]);
  const startTimeRef = useRef<number | null>(null);
  const [cursorTime, setCursorTime] = useState(0);
  const [isReplaying, setIsReplaying] = useState(false);
  const replayTimeouts = useRef<number[]>([]);
  const [instrument, setInstrument] = useState("acoustic_grand_piano");
  const [midiBlob, setMidiBlob] = useState<Blob | null>(null);
  const [harmonizerOpen, setHarmonizerOpen] = useState(false);
  
  // Refs for measuring layout
  const controlsRef = useRef<HTMLDivElement>(null);
  const pianoRef = useRef<HTMLDivElement>(null);
  const [availableHeight, setAvailableHeight] = useState(300);

  // Calculate available space between controls and piano
  useEffect(() => {
    const calculateSpace = () => {
      if (controlsRef.current && pianoRef.current) {
        const controlsRect = controlsRef.current.getBoundingClientRect();
        const pianoRect = pianoRef.current.getBoundingClientRect();
        const spaceAvailable = pianoRect.top - controlsRect.bottom;
        // Reserve more space for piano keyboard and bottom margin
        setAvailableHeight(Math.max(200, spaceAvailable - 80)); 
      }
    };

    // Calculate on mount and resize
    calculateSpace();
    window.addEventListener('resize', calculateSpace);
    
    // Also recalculate after a short delay to ensure layout is settled
    const timeout = setTimeout(calculateSpace, 100);
    
    return () => {
      window.removeEventListener('resize', calculateSpace);
      clearTimeout(timeout);
    };
  }, []);

  // MIDI setup
  useEffect(() => {
    if (navigator.requestMIDIAccess) {
      navigator.requestMIDIAccess().then((midiAccess) => {
        for (let input of midiAccess.inputs.values()) {
          input.onmidimessage = () => {};
        }
      });
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
    if (!midiBlob || !window.AudioContext) {
      console.log("No MIDI blob or AudioContext available");
      return;
    }
    
    setIsReplaying(true);
    setCursorTime(0);

    try {
      // Parse the MIDI blob and play it back
      const arrayBuffer = await midiBlob.arrayBuffer();
      const midi = new Midi(arrayBuffer);
      
      const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
      const inst = await Soundfont.instrument(ctx, instrument as any);
      
      // Clear any existing timeouts
      replayTimeouts.current.forEach(timeout => clearTimeout(timeout));
      replayTimeouts.current = [];
      
      let maxEndTime = 0;
      
      // Schedule all notes from the MIDI
      midi.tracks.forEach(track => {
        track.notes.forEach(note => {
          const startDelay = note.time * 1000; // Convert to ms
          const duration = note.duration * 1000; // Convert to ms
          
          maxEndTime = Math.max(maxEndTime, startDelay + duration);
          
          // Schedule note start
          const startTimeout = window.setTimeout(() => {
            console.log("Playing MIDI note", note.midi, note.name);
            inst.play(note.name, ctx.currentTime, { duration: note.duration });
          }, startDelay);
          replayTimeouts.current.push(startTimeout);
          
          // Update cursor
          const cursorTimeout = window.setTimeout(() => {
            setCursorTime(startDelay);
          }, startDelay);
          replayTimeouts.current.push(cursorTimeout);
        });
      });
      
      // Schedule metronome clicks
      const totalDuration = getTotalDurationMs();
      const beatInterval = getSixteenthNoteMs() * 4; // quarter note
      const numBeats = Math.ceil(totalDuration / beatInterval);

      for (let i = 0; i < numBeats; i++) {
        const beatTime = i * beatInterval;
        const clickTimeout = window.setTimeout(() => {
          playClick(900); // Lower pitch for replay metronome
        }, beatTime);
        replayTimeouts.current.push(clickTimeout);
      }
      
      // End replay
      const endTimeout = window.setTimeout(() => {
        setIsReplaying(false);
        setCursorTime(0);
        ctx.close();
      }, Math.max(maxEndTime, totalDuration) + 1000);
      replayTimeouts.current.push(endTimeout);
      
    } catch (error) {
      console.error("Error playing MIDI:", error);
      setIsReplaying(false);
      setCursorTime(0);
    }
  };

  // Handle note on (key press)
  const handleNoteOn = (midi: number, time: number) => {
    console.log("Note on:", midi, "time:", time, "MIDI range:", MIDI_LOW, "to", MIDI_HIGH);
    if (isCountingIn) {
      // During count-in, treat as first beat (time = 0)
      const newNote: MidiNote = { midi, startTime: 0 };
      setNotes((prev) => [...prev, newNote]);
    } else if (isRecording && startTimeRef.current !== null) {
      const quantized = quantizeTime(time);
      const newNote: MidiNote = { midi, startTime: quantized };
      setNotes((prev) => [...prev, newNote]);
    }
  };

  // Handle note off (key release)
  const handleNoteOff = (midi: number, time: number) => {
    console.log("Note off:", midi, "time:", time, "isRecording:", isRecording);
    if (isRecording && startTimeRef.current !== null) {
      const quantized = quantizeTime(time);
      
      // Find the most recent note with this MIDI number that doesn't have an endTime
      setNotes((prev) => {
        const newNotes = [...prev];
        // Find last note with this midi that doesn't have endTime set
        for (let i = newNotes.length - 1; i >= 0; i--) {
          if (newNotes[i].midi === midi && !newNotes[i].endTime) {
            const duration = Math.max(quantized - newNotes[i].startTime, getSixteenthNoteMs());
            console.log("Setting duration for note:", midi, "duration:", duration);
            newNotes[i] = {
              ...newNotes[i],
              endTime: quantized,
              duration: duration
            };
            break;
          }
        }
        return newNotes;
      });
    }
  };
  
  // Auto-generate MIDI when notes change
  useEffect(() => {
    console.log("Notes changed:", notes);
    if (notes.length > 0) {
      generateMidiBlob();
    }
  }, [notes, tempo]);

  // Generate MIDI blob without downloading
  const generateMidiBlob = () => {
    if (notes.length === 0) {
      setMidiBlob(null);
      return;
    }

    console.log("Generating MIDI blob for notes:", notes);
    const midi = new Midi();
    const track = midi.addTrack();
    midi.header.setTempo(tempo);

    notes.forEach(note => {
      const duration = note.duration || getSixteenthNoteMs();
      console.log("Adding note to MIDI:", note.midi, "startTime:", note.startTime, "duration:", duration);
      track.addNote({
        midi: note.midi,
        time: note.startTime / 1000,
        duration: duration / 1000,
        velocity: 0.8,
      });
    });

    const blob = new Blob([new Uint8Array(midi.toArray())], { type: "audio/midi" });
    setMidiBlob(blob);
  };
  
  const exportToMidi = () => {
    if (!midiBlob) {
      generateMidiBlob();
      return;
    }

    console.log("Exporting MIDI", notes);
    
    const url = URL.createObjectURL(midiBlob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "export.mid";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  // --- Piano roll grid with more notes ---
  const renderPianoRoll = (availableHeight: number) => {
    const width = 1280;
    const height = Math.max(200, availableHeight - 20); // Reduced padding for tighter fit
    const boxWidth = width / TOTAL_BOXES;
    const boxHeight = height / PITCHES;

    // midiOrder: top row is highest note, bottom is lowest
    const midiOrder = Array.from({ length: PITCHES }, (_, i) => MIDI_HIGH - i); // Top is G5, bottom is C3

    
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
          {/* Notes - show as rectangles with proper duration */}
          {notes.map((note, idx) => {
            const startCol = Math.round(note.startTime / getSixteenthNoteMs());
            const duration = note.duration || getSixteenthNoteMs();
            const durationCols = Math.max(1, Math.round(duration / getSixteenthNoteMs()));
            
            const x = startCol * boxWidth;
            const noteWidth = durationCols * boxWidth - 2; // Small gap between notes
            
            const midiIdx = midiOrder.indexOf(note.midi);
            if (midiIdx === -1) {
              console.log("Note not found in range:", note.midi, "Available range:", MIDI_LOW, "to", MIDI_HIGH);
              return null; // Only show notes in C3–G5
            }
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
          {/* Moving cursor */}
          {isRecording && (
            <rect
              x={(cursorTime / getTotalDurationMs()) * width}
              y={0}
              width={3}
              height={height}
              fill="url(#cursorGradient)"
              opacity={0.9}
              filter="drop-shadow(0 0 10px rgba(255, 79, 79, 0.8))"
            />
          )}
          
          {/* Gradient definitions */}
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
        background: "linear-gradient(135deg,rgb(11, 21, 40) 0%,rgb(54, 118, 208) 100%)",
        overflow: "auto", // Allow scrolling if content doesn't fit
        justifyContent: "space-between", // Distribute space evenly
        alignItems: "center"
      }}
    >
      {/* Logo in top-left corner */}
      <div style={{ position: "absolute", top: 24, left: 40, zIndex: 10 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "12px 20px",
            background: "linear-gradient(135deg, rgba(79, 209, 197, 0.15) 0%, rgba(25, 118, 210, 0.15) 100%)",
            borderRadius: "16px",
            border: "1px solid rgba(79, 209, 197, 0.3)",
            boxShadow: "0 0 20px rgba(79, 209, 197, 0.4), inset 0 0 20px rgba(25, 118, 210, 0.1)",
            backdropFilter: "blur(10px)"
          }}
        >
          <div style={{
            width: "32px",
            height: "32px",
            background: "linear-gradient(135deg, #4fd1c5 0%, #38bdf8 50%, #1976d2 100%)",
            borderRadius: "8px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "18px",
            boxShadow: "0 0 15px rgba(79, 209, 197, 0.6)"
          }}>
            🎵
          </div>
          <h1
            style={{
              fontFamily: "'Montserrat', sans-serif",
              fontWeight: 800,
              fontSize: "1.8rem",
              letterSpacing: "0.05em",
              background: "linear-gradient(135deg, #4fd1c5 0%, #38bdf8 50%, #1976d2 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
              textShadow: "0 0 20px rgba(79, 209, 197, 0.5)",
              margin: 0,
              filter: "drop-shadow(0 0 10px rgba(79, 209, 197, 0.3))"
            }}
          >
            HarmonAIzer
          </h1>
        </div>
      </div>

      {/* Open Harmonizer button in top-right corner */}
      <div style={{ position: "absolute", top: 24, right: 40, zIndex: 10 }}>
        <button
          onClick={() => setHarmonizerOpen(true)}
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            padding: "12px 20px",
            background: "linear-gradient(135deg, #4fd1c5 0%, #38bdf8 50%, #1976d2 100%)",
            color: "#fff",
            border: "1px solid rgba(79, 209, 197, 0.3)",
            borderRadius: "16px",
            fontSize: "17px",
            fontWeight: 600,
            boxShadow: "0 0 20px rgba(79, 209, 197, 0.5), 0 4px 15px rgba(0,0,0,0.2)",
            cursor: "pointer",
            transition: "all 0.2s",
            transform: "translateY(-1px)",
            backdropFilter: "blur(10px)"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.transform = "translateY(-2px)";
            e.currentTarget.style.boxShadow = "0 0 25px rgba(79, 209, 197, 0.6), 0 6px 20px rgba(0,0,0,0.3)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.transform = "translateY(-1px)";
            e.currentTarget.style.boxShadow = "0 0 20px rgba(79, 209, 197, 0.5), 0 4px 15px rgba(0,0,0,0.2)";
          }}
        >
          🎵 Open Harmonizer
        </button>
      </div>
      {/* Controls - Responsive layout */}
      <div 
        ref={controlsRef}
        style={{ 
          width: "100%",
          maxWidth: "min(1000px, 95vw)", 
          margin: "min(60px, 8vh) auto min(8px, 1vh) auto", // More responsive margins
          display: "flex",
          flexDirection: "column",
          gap: "min(10px, 1.5vh)", // More responsive gap
          alignItems: "center",
          padding: "0 20px",
          boxSizing: "border-box",
          flexShrink: 0 // Prevent controls from shrinking
        }}>
        {/* Settings Row */}
        <div style={{ 
          display: "flex", 
          gap: "min(20px, 2vw)", 
          alignItems: "center", 
          flexWrap: "wrap",
          justifyContent: "center",
          width: "100%"
        }}>
          <label style={{ 
            color: "#e2e8f0", 
            fontWeight: 600, 
            fontSize: "min(16px, 4vw)",
            textShadow: "0 0 10px rgba(79, 209, 197, 0.5)",
            display: "flex",
            alignItems: "center",
            gap: 8,
            whiteSpace: "nowrap"
          }}>
            Tempo:
            <input
              type="number"
              min={30}
              max={300}
              value={tempo}
              onChange={e => setTempo(Number(e.target.value))}
              style={{ 
                width: "min(80px, 20vw)", 
                padding: "8px 12px",
                borderRadius: 8,
                border: "2px solid rgba(79, 209, 197, 0.3)",
                background: "rgba(255, 255, 255, 0.1)",
                color: "#fff",
                fontSize: "min(16px, 4vw)",
                textAlign: "center",
                boxShadow: "0 0 10px rgba(79, 209, 197, 0.2)"
              }}
            />
            <span>BPM</span>
          </label>
          
          <label style={{ 
            color: "#e2e8f0", 
            fontWeight: 600, 
            fontSize: "min(16px, 4vw)",
            textShadow: "0 0 10px rgba(79, 209, 197, 0.5)",
            display: "flex",
            alignItems: "center",
            gap: 8,
            whiteSpace: "nowrap"
          }}>
            Instrument:
            <select
              value={instrument}
              onChange={e => setInstrument(e.target.value)}
              style={{ 
                fontSize: "min(14px, 3.5vw)", 
                padding: "8px 12px",
                borderRadius: 8,
                border: "2px solid rgba(79, 209, 197, 0.3)",
                background: "rgba(255, 255, 255, 0.1)",
                color: "#fff",
                boxShadow: "0 0 10px rgba(79, 209, 197, 0.2)",
                minWidth: "min(180px, 35vw)",
                maxWidth: "220px"
              }}
            >
              {INSTRUMENTS.map(inst => (
                <option key={inst} value={inst} style={{ background: "#1a1a2e", color: "#fff" }}>
                  {inst.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </label>
        </div>

        {/* Transport Controls Row */}
        <div style={{ 
          display: "flex", 
          gap: "min(16px, 2vw)", 
          alignItems: "center",
          flexWrap: "wrap",
          justifyContent: "center",
          width: "100%"
        }}>
          <button 
            onClick={startRecording} 
            disabled={isCountingIn || isRecording || isReplaying}
            style={{
              background: (isCountingIn || isRecording || isReplaying) 
                ? "rgba(100, 100, 100, 0.5)" 
                : "linear-gradient(135deg, #10b981 0%, #059669 100%)",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "12px 20px",
              fontSize: "min(16px, 4vw)",
              fontWeight: 600,
              cursor: (isCountingIn || isRecording || isReplaying) ? "not-allowed" : "pointer",
              transition: "all 0.2s",
              boxShadow: (isCountingIn || isRecording || isReplaying) 
                ? "none" 
                : "0 0 15px rgba(16, 185, 129, 0.4), 0 4px 15px rgba(0,0,0,0.2)",
              transform: (isCountingIn || isRecording || isReplaying) ? "none" : "translateY(-1px)",
              minWidth: "min(140px, 35vw)", // Responsive width
              maxWidth: "200px",
              textAlign: "center"
            }}
          >
            {isCountingIn ? `Count-in: ${count + 1}` : isRecording ? "Recording..." : "Record"}
          </button>
          
          <button 
            onClick={() => { stopRecording(); stopReplay(); }} 
            disabled={!isRecording && !isCountingIn && !isReplaying}
            style={{
              background: (!isRecording && !isCountingIn && !isReplaying) 
                ? "rgba(100, 100, 100, 0.5)" 
                : "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "12px 20px",
              fontSize: "min(16px, 4vw)",
              fontWeight: 600,
              cursor: (!isRecording && !isCountingIn && !isReplaying) ? "not-allowed" : "pointer",
              transition: "all 0.2s",
              boxShadow: (!isRecording && !isCountingIn && !isReplaying) 
                ? "none" 
                : "0 0 15px rgba(239, 68, 68, 0.4), 0 4px 15px rgba(0,0,0,0.2)",
              transform: (!isRecording && !isCountingIn && !isReplaying) ? "none" : "translateY(-1px)",
              minWidth: "min(80px, 20vw)",
              maxWidth: "120px"
            }}
          >
            Stop
          </button>
          
          <button 
            onClick={replayNotes} 
            disabled={isRecording || !midiBlob || isReplaying}
            style={{
              background: (isRecording || !midiBlob || isReplaying) 
                ? "rgba(100, 100, 100, 0.5)" 
                : "linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "12px 20px",
              fontSize: "min(16px, 4vw)",
              fontWeight: 600,
              cursor: (isRecording || !midiBlob || isReplaying) ? "not-allowed" : "pointer",
              transition: "all 0.2s",
              boxShadow: (isRecording || !midiBlob || isReplaying) 
                ? "none" 
                : "0 0 15px rgba(139, 92, 246, 0.4), 0 4px 15px rgba(0,0,0,0.2)",
              transform: (isRecording || !midiBlob || isReplaying) ? "none" : "translateY(-1px)",
              minWidth: "min(80px, 20vw)",
              maxWidth: "120px"
            }}
          >
            Replay
          </button>
          
          <button 
            onClick={exportToMidi} 
            disabled={notes.length === 0}
            style={{
              background: (notes.length === 0) 
                ? "rgba(100, 100, 100, 0.5)" 
                : "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
              color: "#fff",
              border: "none",
              borderRadius: 8,
              padding: "12px 20px",
              fontSize: "min(16px, 4vw)",
              fontWeight: 600,
              cursor: (notes.length === 0) ? "not-allowed" : "pointer",
              transition: "all 0.2s",
              boxShadow: (notes.length === 0) 
                ? "none" 
                : "0 0 15px rgba(245, 158, 11, 0.4), 0 4px 15px rgba(0,0,0,0.2)",
              transform: (notes.length === 0) ? "none" : "translateY(-1px)",
              minWidth: "min(120px, 30vw)",
              maxWidth: "160px"
            }}
          >
            Export MIDI
          </button>
        </div>
      </div>
      {/* Piano Roll */}
      <div style={{ 
        width: "100%",
        maxWidth: "min(1320px, 95vw)", 
        height: `${availableHeight}px`, // Use calculated height
        margin: "min(10px, 1vh) auto min(10px, 1vh) auto", // Balanced margins
        overflowX: "auto",
        borderRadius: 16,
        padding: 8,
        background: "linear-gradient(135deg, rgba(79, 209, 197, 0.05) 0%, rgba(25, 118, 210, 0.05) 100%)",
        boxShadow: "0 0 30px rgba(79, 209, 197, 0.2), inset 0 0 30px rgba(25, 118, 210, 0.1)",
        flexShrink: 0 // Prevent shrinking
      }}>
        {renderPianoRoll(availableHeight)}
      </div>
      {/* Piano at the bottom */}
      <div 
        ref={pianoRef}
        style={{ 
          width: "100%", 
          display: "flex", 
          justifyContent: "center", 
          alignItems: "flex-end", 
          marginBottom: "min(40px, 5vh)", // More bottom margin for dead space
          marginTop: "min(15px, 2vh)", // Adequate top margin
          padding: "0 20px",
          boxSizing: "border-box",
          flexShrink: 0 // Prevent shrinking
        }}>
        <PianoKeyboard 
          onNoteOn={handleNoteOn} 
          onNoteOff={handleNoteOff}
          lowNote={MIDI_LOW} 
          highNote={MIDI_HIGH}
          isRecording={isRecording}
          startTime={startTimeRef.current}
          instrument={instrument}
        />
      </div>

      {/* Harmonizer Panel */}
      <HarmonizerPanel 
        isOpen={harmonizerOpen}
        onClose={() => setHarmonizerOpen(false)}
        midiBlob={midiBlob}
      />
    </div>
  );
};

export default MidiInput;