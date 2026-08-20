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
const originalWindow = globalThis.window;
const frames = new Map();
let nextFrameId = 0;
globalThis.window = {
  requestAnimationFrame(callback) {
    const id = ++nextFrameId;
    frames.set(id, callback);
    return id;
  },
  cancelAnimationFrame(id) {
    frames.delete(id);
  },
};

const clicks = [];
const context = {
  currentTime: 0,
  resume: async () => {},
  createGain() {
    return {
      gain: {
        value: 0,
        setValueAtTime: () => {},
        exponentialRampToValueAtTime: () => {},
      },
      connect(node) {
        return node;
      },
    };
  },
  createOscillator() {
    return {
      frequency: { value: 0 },
      connect(node) {
        return node;
      },
      start(time) {
        clicks.push({ time, frequency: this.frequency.value });
      },
      stop: () => {},
      onended: undefined,
    };
  },
};
const output = { connect: () => output };

async function waitForFrame() {
  for (let attempts = 0; attempts < 20 && frames.size === 0; attempts += 1) {
    await Promise.resolve();
  }
  assert.notEqual(frames.size, 0, "count-in did not request an update frame");
}

function runFrame() {
  const entry = frames.entries().next().value;
  assert.ok(entry, "no count-in frame is ready");
  const [id, callback] = entry;
  frames.delete(id);
  callback();
}

try {
  const { countInScheduler } = await vite.ssrLoadModule(
    "/src/audio/CountInScheduler.ts",
  );
  countInScheduler.context = context;
  countInScheduler.output = output;

  const remaining = [];
  const completed = countInScheduler.start({
    bars: 1,
    tempo: 120,
    timeSignature: { numerator: 6, denominator: 8 },
    onRemaining: (pulses) => remaining.push(pulses),
  });
  await waitForFrame();
  assert.deepEqual(remaining, [6]);
  assert.equal(clicks.length, 6);
  assert.deepEqual(
    clicks.map(({ frequency }) => frequency),
    [1280, 890, 890, 890, 890, 890],
  );
  assert.deepEqual(
    clicks.map(({ time }) => Number(time.toFixed(2))),
    [0.06, 0.31, 0.56, 0.81, 1.06, 1.31],
  );

  context.currentTime = 0.32;
  runFrame();
  assert.equal(remaining.at(-1), 5);
  context.currentTime = 1.56;
  runFrame();
  assert.equal(await completed, true);
  assert.equal(remaining.at(-1), 0);

  clicks.length = 0;
  frames.clear();
  context.currentTime = 2;
  const cancelledRemaining = [];
  const cancelled = countInScheduler.start({
    bars: 2,
    tempo: 120,
    timeSignature: { numerator: 3, denominator: 4 },
    onRemaining: (pulses) => cancelledRemaining.push(pulses),
  });
  await waitForFrame();
  assert.deepEqual(cancelledRemaining, [6]);
  assert.deepEqual(
    clicks.map(({ frequency }) => frequency),
    [1280, 890, 890, 1280, 890, 890],
    "each bar must retain its downbeat accent",
  );
  countInScheduler.cancel();
  assert.equal(await cancelled, false);
  assert.equal(frames.size, 0);

  console.log("Count-in scheduler checks passed");
} finally {
  if (originalWindow === undefined) delete globalThis.window;
  else globalThis.window = originalWindow;
  await vite.close();
}
