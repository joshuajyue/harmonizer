import React, { useState, useEffect } from "react";
import Soundfont from "soundfont-player";

type PianoKeyboardProps = {
  onNoteOn: (midi: number, time: number) => void;
  onNoteOff: (midi: number, time: number) => void;
  lowNote: number;
  highNote: number;
  isRecording: boolean;
  startTime: number | null;
  instrument: string;
};

const KEYBOARD_CHROMATIC = [
  "q","2","w","3","e","r","5","t","6","y","7","u","i","9","o","0","p",
  "z","s","x","d","c","f","v","b","h","n","j","m",",","l","."
];

const KEYBOARD_LAYOUT = KEYBOARD_CHROMATIC.map((key, i) => ({
  key,
  midi: 48 + i // C3 is MIDI 48
}));

function midiToNoteName(midi: number) {
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  return names[midi % 12] + Math.floor(midi / 12 - 1);
}

const isWhite = (midi: number) => !midiToNoteName(midi).includes("#");

const PianoKeyboard: React.FC<PianoKeyboardProps> = ({ 
  onNoteOn, 
  onNoteOff, 
  lowNote, 
  highNote, 
  isRecording, 
  startTime,
  instrument
}) => {
  const [pressed, setPressed] = useState<Set<number>>(new Set());
  const [audio, setAudio] = useState<any>(null);
  const [activeAudioNotes, setActiveAudioNotes] = useState<Map<number, any>>(new Map());

  useEffect(() => {
    let isMounted = true;
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    Soundfont.instrument(ctx, instrument as any).then((inst) => {
      if (isMounted) setAudio(inst);
    });
    return () => { isMounted = false; ctx.close(); };
  }, [instrument]);

  const getCurrentTime = () => {
    if (isRecording && startTime !== null) {
      return performance.now() - startTime;
    }
    return 0;
  };

  const handleDown = (midi: number) => {
    if (pressed.has(midi)) return; // Prevent multiple triggers for same note
    setPressed(prev => new Set(prev).add(midi));
    if (audio) {
      const note = audio.play(midiToNoteName(midi));
      // Store the note instance for later stopping
      if (note && note.stop) {
        setActiveAudioNotes(prev => new Map(prev).set(midi, note));
      }
    }
    onNoteOn(midi, getCurrentTime());
  };

  const handleUp = (midi: number) => {
    setPressed(prev => {
      const next = new Set(prev);
      next.delete(midi);
      return next;
    });
    
    // Stop the audio note
    const activeNote = activeAudioNotes.get(midi);
    if (activeNote && activeNote.stop) {
      activeNote.stop();
    }
    setActiveAudioNotes(prev => {
      const next = new Map(prev);
      next.delete(midi);
      return next;
    });
    
    onNoteOff(midi, getCurrentTime());
  };

  useEffect(() => {
    const downHandler = (e: KeyboardEvent) => {
      if (e.repeat) return; // Prevent repeated firing when holding a key
      const mapping = KEYBOARD_LAYOUT.find(k => k.key === e.key);
      if (mapping && mapping.midi >= lowNote && mapping.midi <= highNote && !pressed.has(mapping.midi)) {
        handleDown(mapping.midi);
      }
    };
    const upHandler = (e: KeyboardEvent) => {
      const mapping = KEYBOARD_LAYOUT.find(k => k.key === e.key);
      if (mapping && mapping.midi >= lowNote && mapping.midi <= highNote) {
        handleUp(mapping.midi);
      }
    };
    window.addEventListener("keydown", downHandler);
    window.addEventListener("keyup", upHandler);
    return () => {
      window.removeEventListener("keydown", downHandler);
      window.removeEventListener("keyup", upHandler);
    };
    // eslint-disable-next-line
  }, [audio, lowNote, highNote, onNoteOn, onNoteOff, pressed]);

  const WHITE_KEY_WIDTH = 36;
  const BLACK_KEY_WIDTH = 24;
  const WHITE_KEY_HEIGHT = 180;
  const BLACK_KEY_HEIGHT = 110;

  // Render keys in chromatic order
  const keys = KEYBOARD_LAYOUT.filter(
    k => k.midi >= lowNote && k.midi <= highNote
  );
  const whiteKeys = keys.filter(k => isWhite(k.midi));
  const blackKeys = keys.filter(k => !isWhite(k.midi));

  return (
    <div style={{ position: "relative", display: "flex", userSelect: "none", height: WHITE_KEY_HEIGHT }}>
      {/* White keys */}
      {whiteKeys.map((key, i) => (
        <div
          key={key.midi}
          style={{
            width: WHITE_KEY_WIDTH,
            height: WHITE_KEY_HEIGHT,
            background: pressed.has(key.midi) 
              ? "linear-gradient(135deg, #4fd1c5 0%, #38bdf8 50%, #1976d2 100%)" 
              : "linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #e2e8f0 100%)",
            border: pressed.has(key.midi) ? "2px solid #4fd1c5" : "1px solid #cbd5e0",
            marginLeft: i === 0 ? 0 : -1,
            zIndex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            fontFamily: "'Montserrat', 'Arial Rounded MT Bold', Arial, sans-serif",
            fontSize: 22,
            borderRadius: 6,
            boxShadow: pressed.has(key.midi) 
              ? "0 0 15px rgba(79, 209, 197, 0.6), 0 4px 15px rgba(0,0,0,0.2)" 
              : "0 2px 8px rgba(0,0,0,0.1)",
            position: "relative",
            transition: "all 0.2s ease",
            transform: pressed.has(key.midi) ? "translateY(2px)" : "translateY(0)"
          }}
          onMouseDown={() => handleDown(key.midi)}
          onMouseUp={() => handleUp(key.midi)}
          onMouseLeave={() => handleUp(key.midi)}
        >
          <span style={{
            color: pressed.has(key.midi) ? "#ffffff" : "#64748b",
            fontSize: 18,
            marginBottom: 10,
            position: "absolute",
            bottom: 8,
            left: 0,
            right: 0,
            textAlign: "center",
            width: "100%",
            fontWeight: pressed.has(key.midi) ? 600 : 400,
            textShadow: pressed.has(key.midi) ? "0 0 8px rgba(255,255,255,0.5)" : "none"
          }}>
            {key.key}
          </span>
        </div>
      ))}
      {/* Black keys */}
      {blackKeys.map((key) => {
        const prevWhiteIdx = whiteKeys.findIndex(wk => wk.midi > key.midi) - 1;
        if (prevWhiteIdx < 0) return null;
        // Position black keys relative to white keys, using original EXTRA_SPACE logic
        const EXTRA_SPACE = 2; // Space added between white keys for visual separation
        const left = (prevWhiteIdx + 1) * (WHITE_KEY_WIDTH - 1) - (BLACK_KEY_WIDTH / 2) + prevWhiteIdx * EXTRA_SPACE;

        return (
          <div
            key={key.midi}
            style={{
              width: BLACK_KEY_WIDTH,
              height: BLACK_KEY_HEIGHT,
              background: pressed.has(key.midi) 
                ? "linear-gradient(135deg, #ff6b6b 0%, #ff8e8e 50%, #ff4f4f 100%)" 
                : "linear-gradient(135deg, #1e293b 0%, #334155 50%, #0f172a 100%)",
              color: "#fff",
              border: pressed.has(key.midi) ? "2px solid #ff6b6b" : "1px solid #475569",
              position: "absolute",
              left,
              zIndex: 2,
              top: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "flex-end",
              fontFamily: "'Montserrat', 'Arial Rounded MT Bold', Arial, sans-serif",
              fontSize: 16,
              borderRadius: 4,
              boxShadow: pressed.has(key.midi) 
                ? "0 0 15px rgba(255, 107, 107, 0.6), 0 4px 15px rgba(0,0,0,0.4)" 
                : "0 4px 12px rgba(0,0,0,0.3)",
              pointerEvents: "auto",
              transition: "all 0.2s ease",
              transform: pressed.has(key.midi) ? "translateY(2px)" : "translateY(0)"
            }}
            onMouseDown={() => handleDown(key.midi)}
            onMouseUp={() => handleUp(key.midi)}
            onMouseLeave={() => handleUp(key.midi)}
          >
            <span style={{
              color: "#fff",
              fontSize: 15,
              marginBottom: 10,
              position: "absolute",
              bottom: 6,
              left: 0,
              right: 0,
              textAlign: "center",
              width: "100%",
              fontWeight: pressed.has(key.midi) ? 600 : 400,
              textShadow: pressed.has(key.midi) ? "0 0 8px rgba(255,255,255,0.8)" : "0 1px 2px rgba(0,0,0,0.5)"
            }}>
              {key.key}
            </span>
          </div>
        );
      })}
    </div>
  );
};

export default PianoKeyboard;
