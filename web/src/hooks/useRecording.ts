import { useCallback } from "react";
import type { PlaybackNote } from "../audio/AudioScheduler";
import { audioScheduler } from "../audio/AudioScheduler";
import { countInScheduler } from "../audio/CountInScheduler";
import { noteCapture } from "../capture/NoteCapture";
import { useStudioStore } from "../store";

const DEFAULT_TIME_SIGNATURE = { numerator: 4, denominator: 4 } as const;

interface RecordingOptions {
  duration: number;
  notes: PlaybackNote[];
  stop: () => void;
}

export function useRecording({
  duration,
  notes,
  stop,
}: RecordingOptions) {
  const melody = useStudioStore((state) => state.melody);
  const metronomeEnabled = useStudioStore(
    (state) => state.metronomeEnabled,
  );
  const recordingState = useStudioStore((state) => state.recordingState);
  const countInBars = useStudioStore((state) => state.countInBars);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);
  const setLoopEnabled = useStudioStore((state) => state.setLoopEnabled);
  const setNoteInputMode = useStudioStore(
    (state) => state.setNoteInputMode,
  );
  const setPlaying = useStudioStore((state) => state.setPlaying);
  const setRecordingState = useStudioStore(
    (state) => state.setRecordingState,
  );
  const setCountInRemaining = useStudioStore(
    (state) => state.setCountInRemaining,
  );
  const timeSignature =
    melody.timeSignature ?? DEFAULT_TIME_SIGNATURE;

  const armRecording = useCallback(() => {
    if (
      recordingState === "counting" ||
      recordingState === "recording"
    ) {
      return;
    }
    noteCapture.releaseAll();
    setRecordingState(recordingState === "armed" ? "idle" : "armed");
  }, [recordingState, setRecordingState]);

  const startRecording = useCallback(async () => {
    const initialState = useStudioStore.getState();
    if (
      initialState.recordingState === "counting" ||
      initialState.recordingState === "recording"
    ) {
      return;
    }
    const startBeat = Math.max(0, initialState.currentBeat);
    audioScheduler.stop();
    countInScheduler.cancel();
    setNoteInputMode("record");
    setLoopEnabled(false);
    if (countInBars > 0) {
      setRecordingState("counting");
      try {
        const completed = await countInScheduler.start({
          bars: countInBars,
          tempo: melody.tempo,
          timeSignature,
          onRemaining: setCountInRemaining,
        });
        if (
          !completed ||
          useStudioStore.getState().recordingState !== "counting"
        ) {
          return;
        }
      } catch {
        setCountInRemaining(0);
        setRecordingState("armed");
        return;
      }
    }
    setCountInRemaining(0);
    noteCapture.beginTake();
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
    countInBars,
    duration,
    melody.tempo,
    metronomeEnabled,
    notes,
    setCurrentBeat,
    setCountInRemaining,
    setLoopEnabled,
    setNoteInputMode,
    setPlaying,
    setRecordingState,
    timeSignature,
  ]);

  const toggleRecording = useCallback(() => {
    const liveState = useStudioStore.getState().recordingState;
    if (liveState === "counting" || liveState === "recording") {
      stop();
    } else {
      void startRecording();
    }
  }, [startRecording, stop]);

  return { armRecording, toggleRecording };
}
