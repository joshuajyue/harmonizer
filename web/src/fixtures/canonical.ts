import type {
  EngineInfo,
  HarmonizeRequest,
  HarmonizeResponse,
} from "../../../contracts/types";

const REQUEST_URL = "/__contracts/examples/melody.request.json";
const RESPONSE_URL = "/__contracts/examples/harmonize.response.json";

let requestPromise: Promise<HarmonizeRequest> | undefined;
let responsePromise: Promise<HarmonizeResponse> | undefined;

async function loadJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Could not load canonical fixture: ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export async function loadCanonicalRequest() {
  requestPromise ??= loadJson<HarmonizeRequest>(REQUEST_URL);
  return structuredClone(await requestPromise);
}

export async function loadCanonicalResponse() {
  responsePromise ??= loadJson<HarmonizeResponse>(RESPONSE_URL);
  return structuredClone(await responsePromise);
}

export const MOCK_ENGINES: EngineInfo[] = [
  {
    id: "rules",
    name: "Species Rules",
    description: "Deterministic search with explicit voice-leading constraints.",
    available: true,
    learned: false,
  },
  {
    id: "chorale-transformer",
    name: "Chorale Transformer",
    description: "Learned four-part continuation model trained on chorales.",
    available: true,
    learned: true,
  },
  {
    id: "experimental-diffusion",
    name: "Voice Diffusion",
    description: "Exploratory stochastic voicing model.",
    available: false,
    learned: true,
  },
];
