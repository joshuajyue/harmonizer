import { useRef, useState } from "react";
import { useStudioStore, type SelectedNote } from "../../store";
import { selectionKey } from "../../store/selection";
import { quantize } from "../../utils/music";
import {
  contains,
  laneAtY,
  RULER_HEIGHT,
  type RollLayout,
  yToPitch,
} from "./rollGeometry";
import type {
  DrawResult,
  NoteHit,
  ViolationHit,
} from "./rollTypes";
import { useMarqueeSelection } from "./useMarqueeSelection";
import { useNoteDrag } from "./useNoteDrag";
import { useRulerRange } from "./useRulerRange";

type InteractionMode = "note" | "marquee" | "ruler";

function selectionFromHit(hit: NoteHit): SelectedNote {
  return {
    source: hit.source,
    index: hit.index,
    slot: hit.slot,
    voice: hit.voice,
  };
}

export function useRollInteraction(
  drawResultRef: React.RefObject<DrawResult | undefined>,
  duration: number,
  layout: RollLayout,
) {
  const modeRef = useRef<InteractionMode | undefined>(undefined);
  const [hoveredViolation, setHoveredViolation] =
    useState<ViolationHit>();
  const marquee = useMarqueeSelection(drawResultRef, layout);
  const noteDrag = useNoteDrag(layout);
  const ruler = useRulerRange(duration);
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const snap = useStudioStore((state) => state.snap);
  const setSelectedNotes = useStudioStore(
    (state) => state.setSelectedNotes,
  );
  const addMelodyNote = useStudioStore((state) => state.addMelodyNote);
  const deleteSelectedNotes = useStudioStore(
    (state) => state.deleteSelectedNotes,
  );

  function pointerPoint(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  }

  function findNote(x: number, y: number) {
    const matches = (drawResultRef.current?.noteHits ?? []).filter((hit) =>
      contains(hit.rect, x, y),
    );
    return (
      matches.find(
        (hit) => hit.slot === activeSlot && hit.source === "voice",
      ) ??
      matches.find((hit) => hit.source === "melody") ??
      matches.at(-1)
    );
  }

  function onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    if (event.button !== 0) return;
    const point = pointerPoint(event);
    setHoveredViolation(undefined);
    if (point.y < RULER_HEIGHT) {
      modeRef.current = "ruler";
      ruler.start(point.x);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }

    const hit = findNote(point.x, point.y);
    if (hit) {
      if (noteDrag.start(event, point, hit)) {
        modeRef.current = "note";
        event.currentTarget.setPointerCapture(event.pointerId);
      }
      return;
    }
    if (point.y <= layout.noteAreaBottom) {
      modeRef.current = "marquee";
      marquee.start(
        point,
        event.shiftKey
          ? "add"
          : event.metaKey || event.ctrlKey
            ? "toggle"
            : "replace",
      );
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    if (!event.shiftKey && !event.metaKey && !event.ctrlKey) {
      setSelectedNotes([]);
    }
  }

  function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = pointerPoint(event);
    if (modeRef.current === "marquee") {
      marquee.update(point);
      return;
    }
    if (modeRef.current === "ruler") {
      ruler.update(point.x);
      return;
    }
    if (modeRef.current === "note") {
      noteDrag.update(point);
      return;
    }

    const note = findNote(point.x, point.y);
    const violation = (drawResultRef.current?.violationHits ?? []).find(
      (hit) => contains(hit.rect, point.x, point.y),
    );
    setHoveredViolation(violation);
    event.currentTarget.style.cursor =
      point.y < RULER_HEIGHT
        ? "col-resize"
        : violation
          ? "help"
          : note?.editable
            ? point.x >= note.rect.x + note.rect.width - 8
              ? "ew-resize"
              : "grab"
            : "crosshair";
  }

  function onPointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    if (modeRef.current === "marquee") marquee.finish();
    if (modeRef.current === "ruler") ruler.finish();
    noteDrag.finish();
    modeRef.current = undefined;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function onDoubleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (findNote(x, y) || laneAtY(y, layout) !== "melody") return;
    const index = addMelodyNote({
      pitch: yToPitch("melody", y, layout),
      start: Math.max(0, quantize(x / pxPerBeat, snap)),
      duration: Math.max(0.5, snap),
      velocity: 88,
    });
    setSelectedNotes([{ source: "melody", index }]);
  }

  function onContextMenu(event: React.MouseEvent<HTMLCanvasElement>) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const hit = findNote(event.clientX - rect.left, event.clientY - rect.top);
    if (!hit?.editable) return;
    const selection = selectionFromHit(hit);
    const selected = useStudioStore.getState().selectedNotes;
    if (
      !selected.some(
        (candidate) => selectionKey(candidate) === selectionKey(selection),
      )
    ) {
      setSelectedNotes([selection]);
    }
    deleteSelectedNotes();
  }

  return {
    hoveredViolation,
    marquee: marquee.marquee,
    onPointerDown,
    onPointerMove,
    onPointerLeave: () => setHoveredViolation(undefined),
    onPointerUp,
    onDoubleClick,
    onContextMenu,
  };
}
