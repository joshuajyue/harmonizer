import type { VoiceName } from "../../../../contracts/types";
import type { FocusedLane } from "../../store";
import { clamp, VOICE_RANGES } from "../../utils/music";

export type LaneName = "melody" | VoiceName;

export const RULER_HEIGHT = 30;
export const LANE_HEIGHT = 82;
export const CHORD_HEIGHT = 52;
export const SIDEBAR_WIDTH = 148;
export const LANES: LaneName[] = [
  "melody",
  "soprano",
  "alto",
  "tenor",
  "bass",
];
export const ROLL_HEIGHT =
  RULER_HEIGHT + LANES.length * LANE_HEIGHT + CHORD_HEIGHT;
export const NOTE_AREA_BOTTOM =
  RULER_HEIGHT + LANES.length * LANE_HEIGHT;
export const MIN_FOCUS_ROLL_HEIGHT = 320;

export interface RollLayout {
  focusedLane?: FocusedLane;
  lanes: LaneName[];
  laneHeight: number;
  noteAreaBottom: number;
  rollHeight: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function createRollLayout(
  focusedLane?: FocusedLane,
  availableHeight = ROLL_HEIGHT,
): RollLayout {
  if (!focusedLane) {
    return {
      lanes: LANES,
      laneHeight: LANE_HEIGHT,
      noteAreaBottom: NOTE_AREA_BOTTOM,
      rollHeight: ROLL_HEIGHT,
    };
  }
  const rollHeight = Math.max(MIN_FOCUS_ROLL_HEIGHT, availableHeight);
  const laneHeight = rollHeight - RULER_HEIGHT - CHORD_HEIGHT;
  return {
    focusedLane,
    lanes: [focusedLane],
    laneHeight,
    noteAreaBottom: RULER_HEIGHT + laneHeight,
    rollHeight,
  };
}

export function laneTop(lane: LaneName, layout: RollLayout) {
  const index = layout.lanes.indexOf(lane);
  return RULER_HEIGHT + Math.max(0, index) * layout.laneHeight;
}

export function laneCenter(lane: LaneName, layout: RollLayout) {
  return laneTop(lane, layout) + layout.laneHeight / 2;
}

export function laneAtY(y: number, layout: RollLayout): LaneName | undefined {
  if (y < RULER_HEIGHT || y > layout.noteAreaBottom) return undefined;
  const index = Math.floor((y - RULER_HEIGHT) / layout.laneHeight);
  return layout.lanes[index];
}

export function pitchStep(lane: LaneName, layout: RollLayout) {
  const range = VOICE_RANGES[lane];
  return (layout.laneHeight - 16) / (range.max - range.min);
}

export function pitchToY(
  lane: LaneName,
  pitch: number,
  layout: RollLayout,
) {
  const range = VOICE_RANGES[lane];
  const innerHeight = layout.laneHeight - 16;
  const ratio = (range.max - pitch) / (range.max - range.min);
  return laneTop(lane, layout) + 8 + clamp(ratio, 0, 1) * innerHeight;
}

export function pitchRowRect(
  lane: LaneName,
  pitch: number,
  layout: RollLayout,
): Rect {
  const range = VOICE_RANGES[lane];
  const step = pitchStep(lane, layout);
  const center = pitchToY(lane, pitch, layout);
  const top =
    pitch === range.max ? laneTop(lane, layout) : center - step / 2;
  const bottom =
    pitch === range.min
      ? laneTop(lane, layout) + layout.laneHeight
      : center + step / 2;
  return { x: 0, y: top, width: SIDEBAR_WIDTH, height: bottom - top };
}

export function yToPitch(
  lane: LaneName,
  y: number,
  layout: RollLayout,
) {
  const range = VOICE_RANGES[lane];
  const innerHeight = layout.laneHeight - 16;
  const ratio = clamp(
    (y - laneTop(lane, layout) - 8) / innerHeight,
    0,
    1,
  );
  return Math.round(range.max - ratio * (range.max - range.min));
}

export function contains(rect: Rect, x: number, y: number) {
  return (
    x >= rect.x &&
    x <= rect.x + rect.width &&
    y >= rect.y &&
    y <= rect.y + rect.height
  );
}

export function intersects(first: Rect, second: Rect) {
  return (
    first.x <= second.x + second.width &&
    first.x + first.width >= second.x &&
    first.y <= second.y + second.height &&
    first.y + first.height >= second.y
  );
}

export function rectBetween(
  start: { x: number; y: number },
  end: { x: number; y: number },
): Rect {
  return {
    x: Math.min(start.x, end.x),
    y: Math.min(start.y, end.y),
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  };
}
