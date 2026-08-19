import { useEffect } from "react";
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
  const { toggle, stop, toggleRecording } = usePlayback();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.repeat ||
        isInteractiveTarget(event.target) ||
        event.metaKey ||
        event.ctrlKey ||
        event.altKey
      ) {
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
        event.preventDefault();
        toggleRecording();
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
      } else if (
        state.selectedNotes.length > 0 &&
        (event.key === "ArrowUp" || event.key === "ArrowDown")
      ) {
        event.preventDefault();
        state.transposeSelectedNotes(
          (event.key === "ArrowUp" ? 1 : -1) * (event.shiftKey ? 12 : 1),
        );
      } else if (
        state.selectedNotes.length > 0 &&
        (event.key === "ArrowLeft" || event.key === "ArrowRight")
      ) {
        event.preventDefault();
        state.nudgeSelectedNotes(
          event.key === "ArrowLeft" ? -state.snap : state.snap,
        );
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
        if (state.focusedLane) state.setFocusedLane(undefined);
        else state.clearSelection();
      } else if (
        (event.key === "Delete" || event.key === "Backspace") &&
        state.selectedNotes.length > 0
      ) {
        event.preventDefault();
        state.deleteSelectedNotes();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [stop, toggle, toggleRecording]);
}
