import { useEffect } from "react";
import { apiClient } from "../api/client";
import { useStudioStore } from "../store";

let bootstrapPromise: Promise<void> | undefined;

async function bootstrap() {
  const state = useStudioStore.getState();
  state.setEnginesLoading();
  try {
    const { engines } = await apiClient.getEngines();
    state.setEngines(engines);
    if (!apiClient.isMock) return;

    const exampleMelody = await apiClient.getExampleMelody();
    const available = engines.filter((engine) => engine.available);
    const engineA = available.find((engine) => !engine.learned) ?? available[0];
    const engineB =
      available.find((engine) => engine.learned) ?? available[1] ?? engineA;
    if (!engineA || !engineB) return;
    const current = useStudioStore.getState();
    current.replaceMelody(exampleMelody, "Canonical eight-bar study");
    const seeded = useStudioStore.getState();
    seeded.setSlotLoading("A");
    seeded.setSlotLoading("B");
    const [resultA, resultB] = await Promise.all([
      apiClient.harmonize({
        melody: seeded.melody,
        engine: engineA.id,
        options: { voiceCount: 4, temperature: 0 },
      }),
      apiClient.harmonize({
        melody: seeded.melody,
        engine: engineB.id,
        options: { voiceCount: 4, temperature: 0.2 },
      }),
    ]);
    const latest = useStudioStore.getState();
    latest.setSlotResult("A", resultA, seeded.melodyRevision);
    latest.setSlotResult("B", resultB, seeded.melodyRevision);
    latest.setViewMode("A");
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Could not load engines.";
    const current = useStudioStore.getState();
    if (current.engines.length === 0 || apiClient.isMock) {
      current.setEnginesError(message);
    }
    if (current.slots.A.status === "loading") current.setSlotError("A", message);
    if (current.slots.B.status === "loading") current.setSlotError("B", message);
  }
}

export function useStudioBootstrap() {
  useEffect(() => {
    bootstrapPromise ??= bootstrap();
  }, []);
}
