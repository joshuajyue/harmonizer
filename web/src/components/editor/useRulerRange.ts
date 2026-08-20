import { useRef } from "react";
import { audioScheduler } from "../../audio/AudioScheduler";
import { useStudioStore } from "../../store";
import { clamp, quantize } from "../../utils/music";

interface RulerSession {
  anchorBeat: number;
  anchorX: number;
}

export function useRulerRange(duration: number) {
  const sessionRef = useRef<RulerSession | undefined>(undefined);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const snap = useStudioStore((state) => state.snap);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);
  const setPlaying = useStudioStore((state) => state.setPlaying);
  const setLoopEnabled = useStudioStore((state) => state.setLoopEnabled);
  const setLoopRange = useStudioStore((state) => state.setLoopRange);

  function beatAt(x: number) {
    return clamp(quantize(x / pxPerBeat, snap), 0, duration);
  }

  function start(x: number) {
    const anchorBeat = beatAt(x);
    audioScheduler.stop();
    setPlaying(false);
    sessionRef.current = { anchorBeat, anchorX: x };
    setCurrentBeat(anchorBeat);
  }

  function update(x: number) {
    const session = sessionRef.current;
    if (!session) return;
    const beat = beatAt(x);
    setCurrentBeat(beat);
    if (Math.abs(x - session.anchorX) < 4) return;
    const minimumLength = Math.max(0.25, snap);
    let startBeat = Math.min(session.anchorBeat, beat);
    let endBeat = Math.max(session.anchorBeat, beat);
    if (endBeat - startBeat < minimumLength) {
      if (startBeat + minimumLength <= duration) {
        endBeat = startBeat + minimumLength;
      } else {
        startBeat = Math.max(0, endBeat - minimumLength);
      }
    }
    setLoopEnabled(true);
    setLoopRange(startBeat, endBeat, true);
  }

  function finish() {
    sessionRef.current = undefined;
  }

  return { start, update, finish };
}
