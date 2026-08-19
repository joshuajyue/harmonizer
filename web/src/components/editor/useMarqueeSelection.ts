import { useRef, useState } from "react";
import { useStudioStore, type SelectedNote } from "../../store";
import {
  dedupeSelection,
  selectionKey,
} from "../../store/selection";
import {
  intersects,
  NOTE_AREA_BOTTOM,
  rectBetween,
  RULER_HEIGHT,
  type Rect,
} from "./rollGeometry";
import type { DrawResult, NoteHit } from "./rollTypes";

type MarqueeMode = "replace" | "add" | "toggle";

interface MarqueeSession {
  anchor: { x: number; y: number };
  base: SelectedNote[];
  mode: MarqueeMode;
}

function selectionFromHit(hit: NoteHit): SelectedNote {
  return {
    source: hit.source,
    index: hit.index,
    slot: hit.slot,
    voice: hit.voice,
  };
}

export function useMarqueeSelection(
  drawResultRef: React.RefObject<DrawResult | undefined>,
) {
  const sessionRef = useRef<MarqueeSession | undefined>(undefined);
  const [marquee, setMarquee] = useState<Rect>();
  const setSelectedNotes = useStudioStore(
    (state) => state.setSelectedNotes,
  );

  function start(
    point: { x: number; y: number },
    mode: MarqueeMode,
  ) {
    const base =
      mode === "replace"
        ? []
        : [...useStudioStore.getState().selectedNotes];
    const anchor = {
      x: point.x,
      y: Math.max(RULER_HEIGHT, Math.min(NOTE_AREA_BOTTOM, point.y)),
    };
    sessionRef.current = { anchor, base, mode };
    setSelectedNotes(base);
    setMarquee({ x: anchor.x, y: anchor.y, width: 0, height: 0 });
  }

  function update(point: { x: number; y: number }) {
    const session = sessionRef.current;
    if (!session) return;
    const rectangle = rectBetween(session.anchor, {
      x: point.x,
      y: Math.max(RULER_HEIGHT, Math.min(NOTE_AREA_BOTTOM, point.y)),
    });
    const hits = (drawResultRef.current?.noteHits ?? [])
      .filter((hit) => hit.editable && intersects(hit.rect, rectangle))
      .map(selectionFromHit);
    let next: SelectedNote[];
    if (session.mode === "add") {
      next = dedupeSelection([...session.base, ...hits]);
    } else if (session.mode === "toggle") {
      const toggled = new Map(
        session.base.map((selection) => [selectionKey(selection), selection]),
      );
      for (const selection of hits) {
        const key = selectionKey(selection);
        if (toggled.has(key)) toggled.delete(key);
        else toggled.set(key, selection);
      }
      next = [...toggled.values()];
    } else {
      next = hits;
    }
    setSelectedNotes(next);
    setMarquee(rectangle);
  }

  function finish() {
    sessionRef.current = undefined;
    setMarquee(undefined);
  }

  return { marquee, start, update, finish };
}
