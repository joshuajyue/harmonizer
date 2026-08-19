import { audioScheduler } from "../audio/AudioScheduler";
import { useStudioStore } from "../store";

const MIN_CAPTURE_BEATS = 1 / 64;

interface OpenNote {
  pitch: number;
  velocity: number;
  startBeat: number;
  startedAt: number;
  tempo: number;
}

class NoteCapture {
  private openNotes = new Map<number, OpenNote>();

  constructor() {
    if (typeof window !== "undefined") {
      window.addEventListener("blur", this.releaseAll);
      document.addEventListener("visibilitychange", () => {
        if (document.hidden) this.releaseAll();
      });
    }
    useStudioStore.subscribe((state, previous) => {
      if (
        previous.recordingState === "recording" &&
        state.recordingState !== "recording" &&
        this.openNotes.size > 0
      ) {
        this.finishTake(state.currentBeat);
      }
    });
  }

  beginTake() {
    this.releaseAll();
    useStudioStore.getState().clearSelection();
  }

  noteOn(pitch: number, velocity = 94) {
    const normalizedPitch = Math.max(0, Math.min(127, Math.round(pitch)));
    const normalizedVelocity = Math.max(1, Math.min(127, Math.round(velocity)));
    const state = useStudioStore.getState();
    const startBeat = Math.max(
      0,
      audioScheduler.getCurrentBeat() ?? state.currentBeat,
    );
    if (this.openNotes.has(normalizedPitch)) {
      this.closeNote(normalizedPitch, startBeat);
    }
    audioScheduler.previewNoteOn(normalizedPitch, normalizedVelocity);

    if (state.recordingState !== "recording") return;
    this.openNotes.set(normalizedPitch, {
      pitch: normalizedPitch,
      velocity: normalizedVelocity,
      startBeat,
      startedAt: performance.now(),
      tempo: state.melody.tempo,
    });
  }

  noteOff(pitch: number) {
    const normalizedPitch = Math.max(0, Math.min(127, Math.round(pitch)));
    audioScheduler.previewNoteOff(normalizedPitch);
    this.closeNote(normalizedPitch);
  }

  finishTake(stopBeat?: number) {
    for (const pitch of [...this.openNotes.keys()]) {
      this.closeNote(pitch, stopBeat);
    }
    audioScheduler.previewAllNotesOff();
  }

  releaseAll = () => {
    this.finishTake();
  };

  get openNoteCount() {
    return this.openNotes.size;
  }

  private closeNote(pitch: number, stopBeat?: number) {
    const open = this.openNotes.get(pitch);
    if (!open) return;
    this.openNotes.delete(pitch);
    const elapsedBeats =
      ((performance.now() - open.startedAt) / 1000) * (open.tempo / 60);
    const stoppedDuration =
      stopBeat === undefined ? elapsedBeats : stopBeat - open.startBeat;
    const duration = Math.max(
      MIN_CAPTURE_BEATS,
      Number.isFinite(stoppedDuration) ? stoppedDuration : elapsedBeats,
    );
    const state = useStudioStore.getState();
    const index = state.addMelodyNote({
      pitch: open.pitch,
      start: open.startBeat,
      duration,
      velocity: open.velocity,
    });
    const selection = { source: "melody" as const, index };
    useStudioStore
      .getState()
      .setSelectedNotes([
        ...useStudioStore.getState().selectedNotes,
        selection,
      ]);
  }
}

export const noteCapture = new NoteCapture();
