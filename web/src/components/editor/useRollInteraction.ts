import { useRef, useState } from "react";
import type { Note } from "../../../../contracts/types";
import { useStudioStore } from "../../store";
import { clamp, quantize } from "../../utils/music";
import {
  contains,
  laneAtY,
  yToPitch,
} from "./rollGeometry";
import type {
  DrawResult,
  NoteHit,
  ViolationHit,
} from "./rollTypes";

interface DragState {
  hit: NoteHit;
  original: Note;
  startX: number;
  pointerOffsetY: number;
  mode: "move" | "resize";
}

export function useRollInteraction(
  drawResultRef: React.RefObject<DrawResult | undefined>,
  duration: number,
) {
  const dragRef = useRef<DragState | undefined>(undefined);
  const [hoveredViolation, setHoveredViolation] =
    useState<ViolationHit>();
  const activeSlot = useStudioStore((state) => state.activeSlot);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const snap = useStudioStore((state) => state.snap);
  const setCurrentBeat = useStudioStore((state) => state.setCurrentBeat);
  const setSelectedNote = useStudioStore((state) => state.setSelectedNote);
  const setActiveSlot = useStudioStore((state) => state.setActiveSlot);
  const updateMelodyNote = useStudioStore(
    (state) => state.updateMelodyNote,
  );
  const addMelodyNote = useStudioStore((state) => state.addMelodyNote);
  const deleteMelodyNote = useStudioStore(
    (state) => state.deleteMelodyNote,
  );
  const updateVoiceNote = useStudioStore((state) => state.updateVoiceNote);
  const deleteVoiceNote = useStudioStore((state) => state.deleteVoiceNote);

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

  function updateNote(hit: NoteHit, patch: Partial<Note>) {
    if (hit.source === "melody") {
      updateMelodyNote(hit.index, patch);
    } else if (hit.slot && hit.voice) {
      updateVoiceNote(hit.slot, hit.voice, hit.index, patch);
    }
  }

  function deleteNote(hit: NoteHit) {
    if (hit.source === "melody") {
      deleteMelodyNote(hit.index);
    } else if (hit.slot && hit.voice) {
      deleteVoiceNote(hit.slot, hit.voice, hit.index);
    }
  }

  function onPointerDown(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = pointerPoint(event);
    const hit = findNote(point.x, point.y);
    setHoveredViolation(undefined);
    if (!hit) {
      setSelectedNote(undefined);
      setCurrentBeat(clamp(quantize(point.x / pxPerBeat, snap), 0, duration));
      return;
    }
    if (hit.slot && hit.slot !== activeSlot) setActiveSlot(hit.slot);
    setSelectedNote({
      source: hit.source,
      index: hit.index,
      slot: hit.slot,
      voice: hit.voice,
    });
    if (!hit.editable && hit.slot !== activeSlot) return;
    dragRef.current = {
      hit,
      original: { ...hit.note },
      startX: point.x,
      pointerOffsetY: point.y - hit.rect.y,
      mode:
        point.x >= hit.rect.x + hit.rect.width - 8 ? "resize" : "move",
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function onPointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const point = pointerPoint(event);
    const drag = dragRef.current;
    if (drag) {
      if (drag.mode === "resize") {
        const delta = quantize((point.x - drag.startX) / pxPerBeat, snap);
        updateNote(drag.hit, {
          duration: Math.max(snap, drag.original.duration + delta),
        });
      } else {
        const lane =
          drag.hit.source === "melody" ? "melody" : drag.hit.voice;
        if (!lane) return;
        const centerY =
          point.y - drag.pointerOffsetY + drag.hit.rect.height / 2;
        const delta = quantize((point.x - drag.startX) / pxPerBeat, snap);
        updateNote(drag.hit, {
          start: Math.max(0, drag.original.start + delta),
          pitch: yToPitch(lane, centerY),
        });
      }
      return;
    }
    const note = findNote(point.x, point.y);
    const violation = (drawResultRef.current?.violationHits ?? []).find(
      (hit) => contains(hit.rect, point.x, point.y),
    );
    setHoveredViolation(violation);
    event.currentTarget.style.cursor = violation
      ? "help"
      : note?.editable
        ? point.x >= note.rect.x + note.rect.width - 8
          ? "ew-resize"
          : "grab"
        : "crosshair";
  }

  function onPointerUp(event: React.PointerEvent<HTMLCanvasElement>) {
    dragRef.current = undefined;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function onDoubleClick(event: React.MouseEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    if (findNote(x, y) || laneAtY(y) !== "melody") return;
    const index = addMelodyNote({
      pitch: yToPitch("melody", y),
      start: Math.max(0, quantize(x / pxPerBeat, snap)),
      duration: Math.max(0.5, snap),
      velocity: 88,
    });
    setSelectedNote({ source: "melody", index });
  }

  function onContextMenu(event: React.MouseEvent<HTMLCanvasElement>) {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const hit = findNote(event.clientX - rect.left, event.clientY - rect.top);
    if (hit?.editable) deleteNote(hit);
  }

  return {
    hoveredViolation,
    onPointerDown,
    onPointerMove,
    onPointerLeave: () => setHoveredViolation(undefined),
    onPointerUp,
    onDoubleClick,
    onContextMenu,
  };
}
