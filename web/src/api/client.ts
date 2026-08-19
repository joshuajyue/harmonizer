import type {
  EnginesResponse,
  HarmonizeRequest,
  HarmonizeResponse,
  Melody,
  RenderRequest,
} from "../../../contracts/types";
import {
  createMockResponse,
  DEMO_MELODY,
  MOCK_ENGINES,
} from "../fixtures/chorale";
import { normalizeNotes } from "../utils/music";
import { createMockWav } from "./mockAudio";

const mockEnabled = import.meta.env.VITE_USE_MOCK_API !== "false";

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
      melody: {
        ...request.melody,
        notes: normalizeNotes(request.melody.notes),
      },
    };
    if (mockEnabled) {
      await wait(request.engine === "rules-v2" ? 460 : 780);
      return createMockResponse(request.engine, normalizedRequest.melody);
    }
    return requestJson<HarmonizeResponse>("/api/v1/harmonize", {
      method: "POST",
      body: JSON.stringify(normalizedRequest),
    });
  },

  async render(request: RenderRequest): Promise<Blob> {
    if (mockEnabled) {
      await wait(620);
      return createMockWav(request);
    }
    const response = await fetch("/api/v1/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new ApiError(await response.text(), response.status);
    }
    return response.blob();
  },

  async transcribe(audio: Blob): Promise<Melody> {
    if (mockEnabled) {
      await wait(950);
      return {
        ...DEMO_MELODY,
        notes: DEMO_MELODY.notes.map((note) => ({ ...note })),
      };
    }

    const form = new FormData();
    form.append("audio", audio, "recording.webm");
    const response = await fetch("/api/v1/transcribe", {
      method: "POST",
      body: form,
    });
    if (!response.ok) {
      throw new ApiError(await response.text(), response.status);
    }
    const payload = (await response.json()) as Melody | { melody: Melody };
    return "melody" in payload ? payload.melody : payload;
  },
};
