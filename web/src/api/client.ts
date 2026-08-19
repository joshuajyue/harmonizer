import type {
  EnginesResponse,
  HarmonizeRequest,
  HarmonizeResponse,
  Melody,
  RenderRequest,
  TimeSignature,
} from "../../../contracts/types";
import {
  loadCanonicalRequest,
  loadCanonicalResponse,
  MOCK_ENGINES,
} from "../fixtures/canonical";
import { normalizeNotes } from "../utils/music";
import { createMockWav } from "./mockAudio";
import { createMockMidi } from "./mockMidi";

const mockEnabled =
  import.meta.env.VITE_USE_MOCK_API === "true" ||
  (import.meta.env.DEV && import.meta.env.VITE_USE_MOCK_API !== "false");
const DEFAULT_TIME_SIGNATURE: TimeSignature = {
  numerator: 4,
  denominator: 4,
};

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function wait(milliseconds = 320) {
  await new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });
  if (!response.ok) {
    const message = await response.text();
    throw new ApiError(message || response.statusText, response.status);
  }
  return (await response.json()) as T;
}

function completeMelody(
  melody: Melody,
): Melody & { timeSignature: TimeSignature } {
  if (!Number.isFinite(melody.tempo) || melody.tempo <= 0) {
    throw new ApiError("Melody tempo must be an explicit positive number.");
  }
  const timeSignature = melody.timeSignature ?? DEFAULT_TIME_SIGNATURE;
  if (
    !Number.isFinite(timeSignature.numerator) ||
    timeSignature.numerator <= 0 ||
    !Number.isFinite(timeSignature.denominator) ||
    timeSignature.denominator <= 0
  ) {
    throw new ApiError(
      "Time signature numerator and denominator must both be positive.",
    );
  }
  return {
    ...melody,
    tempo: melody.tempo,
    timeSignature: { ...timeSignature },
    notes: normalizeNotes(melody.notes),
  };
}

export const apiClient = {
  isMock: mockEnabled,

  async getEngines(): Promise<EnginesResponse> {
    if (mockEnabled) {
      await wait(180);
      return { engines: MOCK_ENGINES.map((engine) => ({ ...engine })) };
    }
    return requestJson<EnginesResponse>("/api/v1/engines");
  },

  async harmonize(request: HarmonizeRequest): Promise<HarmonizeResponse> {
    const normalizedRequest = {
      ...request,
      melody: completeMelody(request.melody),
    };
    if (mockEnabled) {
      await wait(request.engine === "rules" ? 460 : 780);
      const response = await loadCanonicalResponse();
      return { ...response, engine: request.engine };
    }
    return requestJson<HarmonizeResponse>("/api/v1/harmonize", {
      method: "POST",
      body: JSON.stringify(normalizedRequest),
    });
  },

  async render(request: RenderRequest): Promise<Blob> {
    if (!Number.isFinite(request.tempo) || request.tempo <= 0) {
      throw new ApiError("Render tempo must be an explicit positive number.");
    }
    const normalizedRequest: RenderRequest = {
      ...request,
      tempo: request.tempo,
      synth: request.synth ?? "sf2",
    };
    if (mockEnabled) {
      await wait(620);
      return createMockWav(normalizedRequest);
    }
    const response = await fetch("/api/v1/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(normalizedRequest),
    });
    if (!response.ok) {
      throw new ApiError(await response.text(), response.status);
    }
    return response.blob();
  },

  async transcribe(
    audio: Blob,
    timing: { tempo: number; timeSignature: TimeSignature },
  ): Promise<Melody> {
    const validatedTiming = completeMelody({
      notes: [],
      tempo: timing.tempo,
      timeSignature: timing.timeSignature,
    });
    if (mockEnabled) {
      await wait(950);
      return completeMelody({
        ...(await loadCanonicalRequest()).melody,
        tempo: validatedTiming.tempo,
        timeSignature: validatedTiming.timeSignature,
      });
    }

    const form = new FormData();
    form.append("audio", audio, "recording.webm");
    const params = new URLSearchParams({
      tempo: String(validatedTiming.tempo),
      numerator: String(validatedTiming.timeSignature.numerator),
      denominator: String(validatedTiming.timeSignature.denominator),
    });
    const response = await fetch(`/api/v1/transcribe?${params}`, {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      throw new ApiError(await response.text(), response.status);
    }
    const payload = (await response.json()) as Melody | { melody: Melody };
    return completeMelody("melody" in payload ? payload.melody : payload);
  },

  async exportMidi(
    harmonization: HarmonizeResponse,
    tempo: number,
  ): Promise<Blob> {
    if (!Number.isFinite(tempo) || tempo <= 0) {
      throw new ApiError("MIDI export tempo must be a positive number.");
    }
    if (mockEnabled) return createMockMidi(harmonization, tempo);
    const params = new URLSearchParams({ tempo: String(tempo) });
    const response = await fetch(`/api/v1/midi/export?${params}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(harmonization),
    });
    if (!response.ok) {
      throw new ApiError(await response.text(), response.status);
    }
    return response.blob();
  },

  async getExampleMelody(): Promise<Melody> {
    if (!mockEnabled) {
      throw new ApiError("Canonical examples are only available in mock mode.");
    }
    return completeMelody((await loadCanonicalRequest()).melody);
  },
};
