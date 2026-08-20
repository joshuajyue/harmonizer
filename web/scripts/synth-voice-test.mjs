import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const vite = await createServer({
  root,
  logLevel: "error",
  server: { middlewareMode: true },
  appType: "custom",
});

function audioHarness() {
  const state = { oscillators: [], gains: [], filters: [] };
  const output = { kind: "output" };
  const connectable = (node) => ({
    ...node,
    connections: [],
    connect(next) {
      this.connections.push(next.kind);
      return next;
    },
  });
  const parameter = () => ({
    events: [],
    setValueAtTime(value, time) {
      this.events.push({ method: "set", value, time });
    },
    exponentialRampToValueAtTime(value, time) {
      this.events.push({ method: "exponential", value, time });
    },
  });
  const context = {
    createOscillator() {
      const oscillator = connectable({
        kind: "oscillator",
        type: "sine",
        frequency: parameter(),
        startTime: undefined,
        stopTime: undefined,
        start(time) {
          this.startTime = time;
        },
        stop(time) {
          this.stopTime = time;
        },
        onended: undefined,
      });
      state.oscillators.push(oscillator);
      return oscillator;
    },
    createGain() {
      const gain = connectable({ kind: "gain", gain: parameter() });
      state.gains.push(gain);
      return gain;
    },
    createBiquadFilter() {
      const filter = connectable({
        kind: "filter",
        type: "allpass",
        frequency: parameter(),
        Q: parameter(),
      });
      state.filters.push(filter);
      return filter;
    },
  };
  return { context, output, state };
}

function render(createSynthVoice, voice, pitch) {
  const harness = audioHarness();
  createSynthVoice(
    harness.context,
    harness.output,
    {
      voice,
      pitch,
      start: 0,
      duration: 1,
      velocity: 80,
    },
    1,
    1,
    () => {},
  );
  return harness.state;
}

try {
  const { createSynthVoice } = await vite.ssrLoadModule(
    "/src/audio/synthVoice.ts",
  );
  const soprano = render(createSynthVoice, "soprano", 72);
  const alto = render(createSynthVoice, "alto", 64);
  const tenor = render(createSynthVoice, "tenor", 52);
  const bassLow = render(createSynthVoice, "bass", 40);
  const bassHigh = render(createSynthVoice, "bass", 60);

  assert.equal(soprano.oscillators[0].type, "sine");
  assert.equal(alto.oscillators[0].type, "triangle");
  assert.equal(tenor.oscillators[0].type, "square");
  assert.equal(bassLow.oscillators[0].type, "sawtooth");
  assert.equal(soprano.filters.length, 0);
  assert.equal(alto.filters.length, 0);
  assert.equal(tenor.filters[0].type, "lowpass");
  assert.equal(bassLow.filters[0].type, "lowpass");

  const bassCutoff = bassLow.filters[0].frequency.events[0].value;
  assert.ok(bassCutoff >= 1_400 && bassCutoff <= 2_800);
  assert.deepEqual(bassLow.oscillators[0].connections, ["filter"]);
  assert.deepEqual(bassLow.filters[0].connections, ["gain"]);
  assert.deepEqual(bassLow.gains[0].connections, ["output"]);

  const attackLevel = (state) =>
    state.gains[0].gain.events.find(
      ({ method }) => method === "exponential",
    ).value;
  assert.ok(
    attackLevel(bassLow) > attackLevel(bassHigh),
    "low bass notes need register-dependent gain compensation",
  );
  assert.ok(
    attackLevel(bassLow) / attackLevel(bassHigh) > 1.3,
    "MIDI 40 should receive the full low-register compensation",
  );

  console.log("Preview synth voice checks passed");
} finally {
  await vite.close();
}
