import { useCallback, useEffect, useMemo } from "react";
import { audioScheduler, type PlaybackNote } from "../audio/AudioScheduler";
import { useStudioStore } from "../store";
import { pieceLength, VOICE_ORDER } from "../utils/music";

export function usePlayback() {
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const isPlaying = useStudioStore((state) => state.isPlaying);
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const loopEnabled = useStudioStore((state) => state.loopEnabled);
  const loopStart = useStudioStore((state) => state.loopStart);
  const loopEnd = useStudioStore((state) => state.loopEnd);
  const metronomeEnabled = useStudioStore(
    (state) => state.metronomeEnabled,
  );
  const voiceMute = useStudioStore((state) => state.voiceMute);
  const voiceSolo = useStudioStore((state) => state.voiceSolo);
  const setPlaying = useStudioStore((state) => state.setPlaying);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);

  const duration = useMemo(
    () => pieceLength(melody, [slots.A.result, slots.B.result]),
    [melody, slots.A.result, slots.B.result],
  );

  const notes = useMemo(() => {
    const result = slots[activeSlot].result;
    if (!result) {
      return melody.notes.map<PlaybackNote>((note) => ({
        ...note,
        velocity: note.velocity ?? 88,
        voice: "melody",
      }));
    }
    const hasSolo = VOICE_ORDER.some((voice) => voiceSolo[voice]);
    return result.voices.flatMap((voice) => {
      if (voiceMute[voice.name] || (hasSolo && !voiceSolo[voice.name])) {
        return [];
      }
      return voice.notes.map<PlaybackNote>((note) => ({
        ...note,
        velocity: note.velocity ?? 80,
        voice: voice.name,
      }));
    });
  }, [activeSlot, melody.notes, slots, voiceMute, voiceSolo]);

  const stop = useCallback(() => {
    audioScheduler.stop();
    setPlaying(false);
  }, [setPlaying]);

  const play = useCallback(async () => {
    if (notes.length === 0) return;
    let startBeat = currentBeat >= duration ? 0 : currentBeat;
    if (loopEnabled && (startBeat < loopStart || startBeat >= loopEnd)) {
      startBeat = loopStart;
      setCurrentBeat(startBeat);
    }
    setPlaying(true);
    try {
      await audioScheduler.start({
        notes,
        tempo: melody.tempo,
        startBeat,
        endBeat: duration,
        loopEnabled,
        loopStart,
        loopEnd: Math.min(loopEnd, duration),
        metronomeEnabled,
        timeSignature: melody.timeSignature ?? {
          numerator: 4,
          denominator: 4,
        },
        onPosition: setCurrentBeat,
        onEnded: () => setPlaying(false),
      });
    } catch {
      setPlaying(false);
    }
  }, [
    currentBeat,
    duration,
    loopEnabled,
    loopEnd,
    loopStart,
    melody.tempo,
    melody.timeSignature,
    metronomeEnabled,
    notes,
    setCurrentBeat,
    setPlaying,
  ]);

  const toggle = useCallback(() => {
    if (isPlaying) stop();
    else void play();
  }, [isPlaying, play, stop]);

  useEffect(() => () => audioScheduler.stop(), []);

  return {
    duration,
    isPlaying,
    play,
    stop,
    toggle,
    previewPitch: (pitch: number, velocity?: number) =>
      audioScheduler.preview(pitch, velocity),
  };
}
