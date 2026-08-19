import type { VoiceName } from "../../../../contracts/types";
import { clamp, VOICE_RANGES } from "../../utils/music";

export type LaneName = "melody" | VoiceName;

export const RULER_HEIGHT = 30;
export const LANE_HEIGHT = 92;
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

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function laneTop(lane: LaneName) {
  return RULER_HEIGHT + LANES.indexOf(lane) * LANE_HEIGHT;
}

export function laneCenter(lane: LaneName) {
  return laneTop(lane) + LANE_HEIGHT / 2;
}

export function laneAtY(y: number): LaneName | undefined {
  const index = Math.floor((y - RULER_HEIGHT) / LANE_HEIGHT);
  return LANES[index];
}

export function pitchToY(lane: LaneName, pitch: number) {
  const range = VOICE_RANGES[lane];
  const innerHeight = LANE_HEIGHT - 16;
  const ratio = (range.max - pitch) / (range.max - range.min);
  return laneTop(lane) + 8 + clamp(ratio, 0, 1) * innerHeight;
}

export function yToPitch(lane: LaneName, y: number) {
  const range = VOICE_RANGES[lane];
  const innerHeight = LANE_HEIGHT - 16;
  const ratio = clamp((y - laneTop(lane) - 8) / innerHeight, 0, 1);
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
