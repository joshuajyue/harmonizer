import type { RefObject } from "react";
import { useStudioStore } from "../../store";
import type { RollLayout } from "./rollGeometry";
import type { DrawResult } from "./rollTypes";

export function getFreshDrawResult(
  drawResultRef: RefObject<DrawResult | undefined>,
  duration: number,
  layout: RollLayout,
) {
  const result = drawResultRef.current;
  if (!result) return undefined;
  const state = useStudioStore.getState();
  const model = result.model;
  if (
    model.melody !== state.melody ||
    model.resultA !== state.slots.A.result ||
    model.resultB !== state.slots.B.result ||
    model.viewMode !== state.viewMode ||
    model.activeSlot !== state.activeSlot ||
    model.pxPerBeat !== state.pxPerBeat ||
    model.duration !== duration ||
    model.layout !== layout ||
    model.voiceVisibility !== state.voiceVisibility
  ) {
    return undefined;
  }
  return result;
}
