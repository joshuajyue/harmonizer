import { useCallback, useEffect, useMemo } from "react";
import { audioScheduler, type PlaybackNote } from "../audio/AudioScheduler";
import { noteCapture } from "../capture/NoteCapture";
import { useStudioStore } from "../store";
import { pieceLength, VOICE_ORDER } from "../utils/music";

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
  const recordingState = useStudioStore((state) => state.recordingState);
  const voiceMute = useStudioStore((state) => state.voiceMute);
  const voiceSolo = useStudioStore((state) => state.voiceSolo);
  const setPlaying = useStudioStore((state) => state.setPlaying);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);
  const setLoopEnabled = useStudioStore((state) => state.setLoopEnabled);
  const setRecordingState = useStudioStore(
    (state) => state.setRecordingState,
  );

  const duration = useMemo(
    () =>
      pieceLength(
        melody,
        viewMode === "overlay"
          ? [slots.A.result, slots.B.result]
          : [slots[viewMode].result],
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
    const stopBeat = audioScheduler.getCurrentBeat() ?? currentBeat;
    if (recordingState === "recording") {
      noteCapture.finishTake(stopBeat);
      setCurrentBeat(stopBeat);
      setRecordingState("armed");
    }
    audioScheduler.stop();
    setPlaying(false);
  }, [
    currentBeat,
    recordingState,
    setCurrentBeat,
    setPlaying,
    setRecordingState,
  ]);

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

  const armRecording = useCallback(() => {
    if (recordingState === "recording") return;
    noteCapture.releaseAll();
    setRecordingState(recordingState === "armed" ? "idle" : "armed");
  }, [recordingState, setRecordingState]);

  const startRecording = useCallback(async () => {
    if (useStudioStore.getState().recordingState === "recording") return;
    const startBeat = Math.max(0, useStudioStore.getState().currentBeat);
    audioScheduler.stop();
    noteCapture.beginTake();
    setLoopEnabled(false);
    setRecordingState("recording");
    setPlaying(true);
    try {
      await audioScheduler.start({
        notes,
        tempo: melody.tempo,
        startBeat,
        endBeat: Math.max(duration, startBeat + 1_024),
        loopEnabled: false,
        loopStart: 0,
        loopEnd: 0,
        metronomeEnabled,
        timeSignature: melody.timeSignature ?? {
          numerator: 4,
          denominator: 4,
        },
        onPosition: setCurrentBeat,
        onEnded: () => {
          noteCapture.finishTake(
            audioScheduler.getCurrentBeat() ?? startBeat + 1_024,
          );
          setPlaying(false);
          setRecordingState("armed");
        },
      });
    } catch {
      noteCapture.finishTake(startBeat);
      setPlaying(false);
      setRecordingState("armed");
    }
  }, [
    duration,
    melody.tempo,
    melody.timeSignature,
    metronomeEnabled,
    notes,
    setCurrentBeat,
    setLoopEnabled,
    setPlaying,
    setRecordingState,
  ]);

  const toggleRecording = useCallback(() => {
    if (recordingState === "recording") {
      stop();
    } else {
      void startRecording();
    }
  }, [recordingState, startRecording, stop]);

  useEffect(
    () => () => {
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
