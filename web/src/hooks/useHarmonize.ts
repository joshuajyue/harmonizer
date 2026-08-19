import { useCallback } from "react";
import { apiClient } from "../api/client";
import { useStudioStore, type ComparisonSlotId } from "../store";

export function useHarmonize() {
  const harmonizeSlot = useCallback(async (slot: ComparisonSlotId) => {
    const state = useStudioStore.getState();
    const engine = state.slots[slot].engineId;
    if (!engine || state.melody.notes.length === 0) return;
    const revision = state.melodyRevision;
    state.setSlotLoading(slot);
    try {
      const result = await apiClient.harmonize({
        melody: state.melody,
        engine,
        options: { voiceCount: 4, temperature: engine.includes("rule") ? 0 : 0.2 },
      });
      useStudioStore.getState().setSlotResult(slot, result, revision);
    } catch (error) {
      useStudioStore
        .getState()
        .setSlotError(
          slot,
          error instanceof Error ? error.message : "Harmonization failed.",
        );
    }
  }, []);

  const compareBoth = useCallback(async () => {
    const activeSlot = useStudioStore.getState().activeSlot;
    await Promise.all([harmonizeSlot("A"), harmonizeSlot("B")]);
    const state = useStudioStore.getState();
    if (state.slots.A.result && state.slots.B.result) {
      state.setActiveSlot(activeSlot);
      state.setViewMode("overlay");
    }
  }, [harmonizeSlot]);

  return { harmonizeSlot, compareBoth };
}
