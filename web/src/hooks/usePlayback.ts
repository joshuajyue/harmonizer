import { useCallback, useEffect, useMemo } from "react";
import { audioScheduler, type PlaybackNote } from "../audio/AudioScheduler";
import { countInScheduler } from "../audio/CountInScheduler";
import { noteCapture } from "../capture/NoteCapture";
import { useStudioStore } from "../store";
import { pieceLength, VOICE_ORDER } from "../utils/music";
import { useRecording } from "./useRecording";

const DEFAULT_TIME_SIGNATURE = { numerator: 4, denominator: 4 } as const;

function playbackLoopRange(start: number, end: number, duration: number) {
  const boundedEnd = Math.max(0.25, Math.min(end, duration));
  return {
    start: Math.max(0, Math.min(start, boundedEnd - 0.25)),
    end: boundedEnd,
  };
}

export function usePlayback() {
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const viewMode = useStudioStore((state) => state.viewMode);
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
  const setRecordingState = useStudioStore(
    (state) => state.setRecordingState,
  );
  const setCountInRemaining = useStudioStore(
    (state) => state.setCountInRemaining,
  );
  const timeSignature =
    melody.timeSignature ?? DEFAULT_TIME_SIGNATURE;

  const duration = useMemo(
    () =>
      pieceLength(
        melody,
        [slots[viewMode].result],
      ),
    [melody, slots, viewMode],
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
    const liveState = useStudioStore.getState();
    const stopBeat = audioScheduler.getCurrentBeat() ?? liveState.currentBeat;
    if (liveState.recordingState === "counting") {
      countInScheduler.cancel();
      setCountInRemaining(0);
      setRecordingState("armed");
    } else if (liveState.recordingState === "recording") {
      noteCapture.finishTake(stopBeat);
      setCurrentBeat(stopBeat);
      setRecordingState("armed");
    }
    audioScheduler.stop();
    setPlaying(false);
  }, [
    setCountInRemaining,
    setCurrentBeat,
    setPlaying,
    setRecordingState,
  ]);

  const play = useCallback(async () => {
    if (notes.length === 0) return;
    const playbackLoop = playbackLoopRange(loopStart, loopEnd, duration);
    let startBeat = currentBeat >= duration ? 0 : currentBeat;
    if (
      loopEnabled &&
      (startBeat < playbackLoop.start || startBeat >= playbackLoop.end)
    ) {
      startBeat = playbackLoop.start;
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
        loopStart: playbackLoop.start,
        loopEnd: playbackLoop.end,
        metronomeEnabled,
        timeSignature,
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
    metronomeEnabled,
    notes,
    setCurrentBeat,
    setPlaying,
    timeSignature,
  ]);

  const toggle = useCallback(() => {
    const liveState = useStudioStore.getState();
    if (liveState.isPlaying || liveState.recordingState === "counting") stop();
    else void play();
  }, [play, stop]);

  const { armRecording, toggleRecording } = useRecording({
    duration,
    notes,
    stop,
  });

  useEffect(() => {
    if (!isPlaying) return;
    const playbackLoop = playbackLoopRange(loopStart, loopEnd, duration);
    audioScheduler.setLoop({
      enabled: loopEnabled,
      start: playbackLoop.start,
      end: playbackLoop.end,
    });
  }, [duration, isPlaying, loopEnabled, loopEnd, loopStart]);

  useEffect(() => {
    if (isPlaying) {
      audioScheduler.setMetronomeEnabled(metronomeEnabled);
    }
  }, [isPlaying, metronomeEnabled]);

  useEffect(() => {
    if (isPlaying) audioScheduler.setTempo(melody.tempo);
  }, [isPlaying, melody.tempo]);

  useEffect(() => {
    if (isPlaying) audioScheduler.setTimeSignature(timeSignature);
  }, [isPlaying, timeSignature]);

  useEffect(
    () => () => {
      countInScheduler.cancel();
      noteCapture.releaseAll();
      audioScheduler.stop();
    },
    [],
  );

  return {
    duration,
    isPlaying,
    play,
    stop,
    toggle,
    armRecording,
    toggleRecording,
    previewPitch: (pitch: number, velocity?: number) =>
      audioScheduler.preview(pitch, velocity),
  };
}
