import { useEffect } from "react";
import { useHarmonize } from "./useHarmonize";
import { usePlayback } from "./usePlayback";
import { useStudioStore } from "../store";

function isInteractiveTarget(target: EventTarget | null) {
  return (
    target instanceof Element &&
    Boolean(
      target.closest(
        "input, select, textarea, button, a, audio, [contenteditable='true']",
      ),
    )
  );
}

export function useKeyboardShortcuts() {
  const { toggle, stop } = usePlayback();
  const { compareBoth } = useHarmonize();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isInteractiveTarget(event.target) || event.metaKey || event.ctrlKey) {
        return;
      }
      const state = useStudioStore.getState();
      if (event.code === "Space") {
        event.preventDefault();
        toggle();
      } else if (event.key.toLowerCase() === "l") {
        state.setLoopEnabled(!state.loopEnabled);
      } else if (event.key.toLowerCase() === "m") {
        state.setMetronomeEnabled(!state.metronomeEnabled);
      } else if (event.key.toLowerCase() === "r") {
        void compareBoth();
      } else if (event.key === "1" && state.slots.A.result) {
        state.setViewMode("A");
      } else if (event.key === "2" && state.slots.B.result) {
        state.setViewMode("B");
      } else if (
        event.key === "0" &&
        state.slots.A.result &&
        state.slots.B.result
      ) {
        state.setViewMode("overlay");
      } else if (event.key === "Home") {
        stop();
        state.setCurrentBeat(0);
      } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        stop();
        state.setCurrentBeat(
          Math.max(
            0,
            state.currentBeat +
              (event.key === "ArrowLeft" ? -state.snap : state.snap),
          ),
        );
      } else if (event.key === "Escape") {
        state.setSelectedNote(undefined);
      } else if (
        (event.key === "Delete" || event.key === "Backspace") &&
        state.selectedNote
      ) {
        event.preventDefault();
        const selected = state.selectedNote;
        if (selected.source === "melody") {
          state.deleteMelodyNote(selected.index);
        } else if (selected.slot && selected.voice) {
          state.deleteVoiceNote(
            selected.slot,
            selected.voice,
            selected.index,
          );
        }
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [compareBoth, stop, toggle]);
}
