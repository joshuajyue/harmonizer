import { useRef } from "react";
import type { Note } from "../../../../contracts/types";
import {
  useStudioStore,
  type SelectedNote,
  type SelectionOrigin,
} from "../../store";
import {
  collectSelectionOrigins,
  selectionKey,
} from "../../store/selection";
import { quantize } from "../../utils/music";
import { yToPitch } from "./rollGeometry";
import type { NoteHit } from "./rollTypes";

interface NoteDragState {
  hit: NoteHit;
  original: Note;
  origins: SelectionOrigin[];
  startX: number;
  pointerOffsetY: number;
  mode: "move" | "resize";
}

function selectionFromHit(hit: NoteHit): SelectedNote {
  return {
    source: hit.source,
    index: hit.index,
    slot: hit.slot,
    voice: hit.voice,
  };
}

export function useNoteDrag() {
  const dragRef = useRef<NoteDragState | undefined>(undefined);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const snap = useStudioStore((state) => state.snap);
  const setSelectedNotes = useStudioStore(
    (state) => state.setSelectedNotes,
  );
  const setActiveSlot = useStudioStore((state) => state.setActiveSlot);
  const updateMelodyNote = useStudioStore(
    (state) => state.updateMelodyNote,
  );
  const updateVoiceNote = useStudioStore((state) => state.updateVoiceNote);
  const transformSelectedNotes = useStudioStore(
    (state) => state.transformSelectedNotes,
  );

  function updateNote(hit: NoteHit, patch: Partial<Note>) {
    if (hit.source === "melody") {
      updateMelodyNote(hit.index, patch);
    } else if (hit.slot && hit.voice) {
      updateVoiceNote(hit.slot, hit.voice, hit.index, patch);
    }
  }

  function start(
    event: React.PointerEvent<HTMLCanvasElement>,
    point: { x: number; y: number },
    hit: NoteHit,
  ) {
    const clicked = selectionFromHit(hit);
    let current = useStudioStore.getState().selectedNotes;
    const clickedKey = selectionKey(clicked);
    const alreadySelected = current.some(
      (selection) => selectionKey(selection) === clickedKey,
    );

    if (hit.slot && hit.slot !== activeSlot) {
      setActiveSlot(hit.slot);
      current = [];
    }
    if (event.metaKey || event.ctrlKey) {
      current = alreadySelected
        ? current.filter(
            (selection) => selectionKey(selection) !== clickedKey,
          )
        : [...current, clicked];
      setSelectedNotes(current);
      if (alreadySelected) return false;
    } else if (event.shiftKey) {
      if (!alreadySelected) current = [...current, clicked];
      setSelectedNotes(current);
    } else if (!alreadySelected || current.length === 0) {
      current = [clicked];
      setSelectedNotes(current);
    }

    const origins = collectSelectionOrigins(useStudioStore.getState());
    if (origins.length === 0) return false;
    dragRef.current = {
      hit,
      original: { ...hit.note },
      origins,
      startX: point.x,
      pointerOffsetY: point.y - hit.rect.y,
      mode:
        origins.length === 1 &&
        point.x >= hit.rect.x + hit.rect.width - 8
          ? "resize"
          : "move",
    };
    return true;
  }

  function update(point: { x: number; y: number }) {
    const drag = dragRef.current;
    if (!drag) return;
    if (drag.mode === "resize") {
      const delta = quantize((point.x - drag.startX) / pxPerBeat, snap);
      updateNote(drag.hit, {
        duration: Math.max(snap, drag.original.duration + delta),
      });
      return;
    }

    const lane = drag.hit.source === "melody" ? "melody" : drag.hit.voice;
    if (!lane) return;
    const centerY =
      point.y - drag.pointerOffsetY + drag.hit.rect.height / 2;
    transformSelectedNotes(
      drag.origins,
      quantize((point.x - drag.startX) / pxPerBeat, snap),
      yToPitch(lane, centerY) - drag.original.pitch,
    );
  }

  function finish() {
    dragRef.current = undefined;
  }

  return { start, update, finish };
}
