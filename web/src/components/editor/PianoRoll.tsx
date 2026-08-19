import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useStudioStore } from "../../store";
import { pieceLength, VOICE_RANGES } from "../../utils/music";
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
  const [availableWidth, setAvailableWidth] = useState(0);
  const [melodyFocusRange, setMelodyFocusRange] = useState<
    { min: number; max: number } | undefined
  >();
  const melody = useStudioStore((state) => state.melody);
  const slots = useStudioStore((state) => state.slots);
  const pxPerBeat = useStudioStore((state) => state.pxPerBeat);
  const currentBeat = useStudioStore((state) => state.currentBeat);
  const isPlaying = useStudioStore((state) => state.isPlaying);
  const focusedLane = useStudioStore((state) => state.focusedLane);
  const recordingState = useStudioStore((state) => state.recordingState);
  const pieceDuration = useMemo(
    () => pieceLength(melody, [slots.A.result, slots.B.result]),
    [melody, slots.A.result, slots.B.result],
  );
  const signature = melody.timeSignature ?? {
    numerator: 4,
    denominator: 4,
  };
  const barLength = signature.numerator * (4 / signature.denominator);
  const positionHorizon =
    recordingState === "recording" ? currentBeat + barLength : currentBeat;
  const duration = Math.max(
    pieceDuration,
    Math.ceil(positionHorizon / barLength) * barLength,
  );
  const canvasWidth = Math.max(
    780,
    availableWidth - SIDEBAR_WIDTH,
    duration * pxPerBeat,
  );
  const editorDuration = canvasWidth / pxPerBeat;
  const layout = useMemo(
    () =>
      createRollLayout(
        focusedLane,
        availableHeight,
        focusedLane === "melody" ? melodyFocusRange : undefined,
      ),
    [availableHeight, focusedLane, melodyFocusRange],
  );

  useLayoutEffect(() => {
    if (focusedLane !== "melody" || melody.notes.length === 0) {
      setMelodyFocusRange(undefined);
      return;
    }
    const pitches = melody.notes.map((note) => note.pitch);
    const minimum = Math.min(...pitches);
    const maximum = Math.max(...pitches);
    setMelodyFocusRange((current) => {
      if (
        current &&
        minimum >= current.min &&
        maximum <= current.max
      ) {
        return current;
      }
      return melodyPitchWindow(minimum, maximum);
    });
  }, [focusedLane, melody.notes]);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller) return;
    const updateSize = () => {
      setAvailableHeight(scroller.clientHeight);
      setAvailableWidth(scroller.clientWidth);
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
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
              duration={editorDuration}
              layout={layout}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

function melodyPitchWindow(minimum: number, maximum: number) {
  if (
    minimum >= VOICE_RANGES.melody.min &&
    maximum <= VOICE_RANGES.melody.max
  ) {
    return undefined;
  }
  let min = Math.max(0, Math.floor(minimum / 12) * 12);
  let max = min + 24;
  while (max < maximum) max += 12;
  if (max > 127) {
    min = Math.max(0, min - (max - 127));
    max = 127;
  }
  return { min, max };
}
