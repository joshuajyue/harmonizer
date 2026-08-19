import { useEffect, useMemo, useRef, useState } from "react";
import { useStudioStore } from "../../store";
import { pieceLength } from "../../utils/music";
import { EditorToolbar } from "./EditorToolbar";
import { LaneSidebar } from "./LaneSidebar";
import { PianoRollCanvas } from "./PianoRollCanvas";
import {
  createRollLayout,
  ROLL_HEIGHT,
  SIDEBAR_WIDTH,
} from "./rollGeometry";

export function PianoRoll() {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const [availableHeight, setAvailableHeight] = useState(ROLL_HEIGHT);
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const isPlaying = useStudioStore((state) => state.isPlaying);
  const focusedLane = useStudioStore((state) => state.focusedLane);
  const duration = useMemo(
    () => pieceLength(melody, [slots.A.result, slots.B.result]),
    [melody, slots.A.result, slots.B.result],
  );
  const canvasWidth = Math.max(780, duration * pxPerBeat);
  const layout = useMemo(
    () => createRollLayout(focusedLane, availableHeight),
    [availableHeight, focusedLane],
  );

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const updateHeight = () => setAvailableHeight(scroller.clientHeight);
    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(scroller);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (scrollerRef.current) scrollerRef.current.scrollTop = 0;
  }, [focusedLane]);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller || !isPlaying) return;
    const playhead = SIDEBAR_WIDTH + currentBeat * pxPerBeat;
    const leftEdge = scroller.scrollLeft + SIDEBAR_WIDTH + 40;
    const rightEdge = scroller.scrollLeft + scroller.clientWidth - 80;
    if (playhead > rightEdge || playhead < leftEdge) {
      scroller.scrollTo({
        left: Math.max(0, playhead - SIDEBAR_WIDTH - 100),
        behavior: "smooth",
      });
    }
  }, [currentBeat, isPlaying, pxPerBeat]);

  return (
    <section
      className={`piano-roll-panel ${focusedLane ? "lane-focus-active" : ""}`}
      aria-label={
        focusedLane
          ? `${focusedLane} focused piano roll`
          : "Multi-voice piano roll"
      }
    >
      <EditorToolbar />
      <div className="roll-scroller" ref={scrollerRef}>
        <div
          className="roll-stage"
          style={{
            width: canvasWidth + SIDEBAR_WIDTH,
            height: layout.rollHeight,
          }}
        >
          <LaneSidebar layout={layout} />
          <div
            className="roll-canvas-position"
            style={{ left: SIDEBAR_WIDTH }}
          >
            <PianoRollCanvas
              width={canvasWidth}
              duration={duration}
              layout={layout}
            />
          </div>
        </div>
      </div>
    </section>
  );
}
