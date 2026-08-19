import { useEffect } from "react";
import { useHarmonize } from "./useHarmonize";
import { usePlayback } from "./usePlayback";
import { MUSICAL_TYPING_KEY_SET } from "../input/musicalTyping";
import { useStudioStore } from "../store";

const CHARACTER_SHORTCUTS = [
  { key: "/", action: "loop" },
  { key: "m", action: "metronome" },
  { key: "r", action: "record" },
  { key: "1", action: "result-a" },
  { key: "2", action: "result-b" },
] as const;

const collisions = CHARACTER_SHORTCUTS.filter(({ key }) =>
  MUSICAL_TYPING_KEY_SET.has(key),
);
if (collisions.length > 0) {
  throw new Error(
    `Global shortcuts collide with musical typing: ${collisions
      .map(({ key }) => key)
      .join(", ")}`,
  );
}

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
  const { harmonizeSlot } = useHarmonize();

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.repeat ||
        isInteractiveTarget(event.target) ||
        event.altKey
      ) {
        return;
      }
      const state = useStudioStore.getState();
      if (
        event.key === "Enter" &&
        (event.metaKey || event.ctrlKey)
      ) {
        event.preventDefault();
        void harmonizeSlot(state.activeSlot);
        return;
      }
      if (event.metaKey || event.ctrlKey) return;
      const characterShortcut = CHARACTER_SHORTCUTS.find(
        ({ key }) => key === event.key.toLowerCase(),
      );
      if (event.code === "Space") {
        event.preventDefault();
        toggle();
      } else if (characterShortcut?.action === "loop") {
        state.setLoopEnabled(!state.loopEnabled);
      } else if (characterShortcut?.action === "metronome") {
        state.setMetronomeEnabled(!state.metronomeEnabled);
      } else if (characterShortcut?.action === "record") {
        event.preventDefault();
        toggleRecording();
      } else if (
        characterShortcut?.action === "result-a" &&
        state.slots.A.result
      ) {
        state.setViewMode("A");
      } else if (
        characterShortcut?.action === "result-b" &&
        state.slots.B.result
      ) {
        state.setViewMode("B");
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
  }, [harmonizeSlot, stop, toggle, toggleRecording]);
}
