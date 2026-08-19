import { useCallback, useEffect, useMemo } from "react";
import { audioScheduler, type PlaybackNote } from "../audio/AudioScheduler";
import { noteCapture } from "../capture/NoteCapture";
import { useStudioStore } from "../store";
import { pieceLength, VOICE_ORDER } from "../utils/music";

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
  const recordingState = useStudioStore((state) => state.recordingState);
  const voiceMute = useStudioStore((state) => state.voiceMute);
  const voiceSolo = useStudioStore((state) => state.voiceSolo);
  const setPlaying = useStudioStore((state) => state.setPlaying);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);
  const setLoopEnabled = useStudioStore((state) => state.setLoopEnabled);
  const setRecordingState = useStudioStore(
    (state) => state.setRecordingState,
  );
  const setNoteInputMode = useStudioStore(
    (state) => state.setNoteInputMode,
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
    setNoteInputMode("record");
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
        timeSignature,
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
    metronomeEnabled,
    notes,
    setCurrentBeat,
    setLoopEnabled,
    setNoteInputMode,
    setPlaying,
    setRecordingState,
    timeSignature,
  ]);

  const toggleRecording = useCallback(() => {
    if (recordingState === "recording") {
      stop();
    } else {
      void startRecording();
    }
  }, [recordingState, startRecording, stop]);

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
