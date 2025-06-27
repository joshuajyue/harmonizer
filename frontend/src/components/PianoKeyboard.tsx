import React, { useState, useEffect } from "react";
import Soundfont from "soundfont-player";

type PianoKeyboardProps = {
  onPlayNote: (midi: number) => void;
  lowNote: number;
  highNote: number;
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

const PianoKeyboard: React.FC<PianoKeyboardProps> = ({ onPlayNote, lowNote, highNote }) => {
  const [pressed, setPressed] = useState<number | null>(null);
  const [audio, setAudio] = useState<any>(null);

  useEffect(() => {
    let isMounted = true;
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    Soundfont.instrument(ctx, "acoustic_grand_piano").then((inst) => {
      if (isMounted) setAudio(inst);
    });
    return () => { isMounted = false; ctx.close(); };
  }, []);

  useEffect(() => {
    const downHandler = (e: KeyboardEvent) => {
      const mapping = KEYBOARD_LAYOUT.find(k => k.key === e.key);
      if (mapping && mapping.midi >= lowNote && mapping.midi <= highNote) {
        setPressed(mapping.midi);
        if (audio) audio.play(midiToNoteName(mapping.midi));
        onPlayNote(mapping.midi);
      }
    };
    const upHandler = () => setPressed(null);
    window.addEventListener("keydown", downHandler);
    window.addEventListener("keyup", upHandler);
    return () => {
      window.removeEventListener("keydown", downHandler);
      window.removeEventListener("keyup", upHandler);
    };
    // eslint-disable-next-line
  }, [audio, lowNote, highNote, onPlayNote]);

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
            background: pressed === key.midi ? "#b3e5fc" : "#fff",
            border: "1px solid #bbb",
            marginLeft: i === 0 ? 0 : -1,
            zIndex: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            fontFamily: "'Montserrat', 'Arial Rounded MT Bold', Arial, sans-serif",
            fontSize: 22,
            borderRadius: 6,
            boxShadow: pressed === key.midi ? "0 0 8px #4fd1c5" : "none",
            position: "relative",
            transition: "background 0.1s, box-shadow 0.1s"
          }}
          onMouseDown={() => {
            setPressed(key.midi);
            if (audio) audio.play(midiToNoteName(key.midi));
            onPlayNote(key.midi);
          }}
          onMouseUp={() => setPressed(null)}
        >
          <span style={{
            color: "#888",
            fontSize: 18,
            marginBottom: 10,
            position: "absolute",
            bottom: 8,
            left: 0,
            right: 0,
            textAlign: "center",
            width: "100%"
          }}>
            {key.key}
          </span>
        </div>
      ))}
      {/* Black keys */}
      {blackKeys.map((key) => {
        // Find the index of this key in the full keys array
        const keyIdx = keys.findIndex(k => k.midi === key.midi);

        // Find previous white key
        let prevWhiteIdx = keyIdx - 1;
        while (prevWhiteIdx >= 0 && !isWhite(keys[prevWhiteIdx].midi)) prevWhiteIdx--;

        // Find next white key
        let nextWhiteIdx = keyIdx + 1;
        while (nextWhiteIdx < keys.length && !isWhite(keys[nextWhiteIdx].midi)) nextWhiteIdx++;

        // If either white key is missing, skip rendering this black key
        if (prevWhiteIdx < 0 || nextWhiteIdx >= keys.length) return null;

        // Calculate the left position: midpoint between the two adjacent white keys
        const left = (prevWhiteIdx * WHITE_KEY_WIDTH + nextWhiteIdx * WHITE_KEY_WIDTH) / 2 + (WHITE_KEY_WIDTH - BLACK_KEY_WIDTH) / 2;

        return (
          <div
            key={key.midi}
            style={{
              width: BLACK_KEY_WIDTH,
              height: BLACK_KEY_HEIGHT,
              background: pressed === key.midi ? "#0288d1" : "#222",
              color: "#fff",
              border: "1px solid #333",
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
              boxShadow: pressed === key.midi ? "0 0 8px #4fd1c5" : "0 2px 8px #111",
              pointerEvents: "auto",
              transition: "background 0.1s, box-shadow 0.1s"
            }}
            onMouseDown={() => {
              setPressed(key.midi);
              if (audio) audio.play(midiToNoteName(key.midi));
              onPlayNote(key.midi);
            }}
            onMouseUp={() => setPressed(null)}
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
              width: "100%"
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