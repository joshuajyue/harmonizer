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

function playbackOptions(overrides = {}) {
  return {
    notes: [],
    tempo: 96,
    startBeat: 0,
    endBeat: 32,
    loopEnabled: true,
    loopStart: 4,
    loopEnd: 8,
    metronomeEnabled: true,
    timeSignature: { numerator: 4, denominator: 4 },
    onPosition: () => {},
    onEnded: () => {},
    ...overrides,
  };
}

try {
  const { AudioScheduler } = await vite.ssrLoadModule(
    "/src/audio/AudioScheduler.ts",
  );
  const scheduler = new AudioScheduler();
  const rebases = [];
  let currentBeat = 6;
  scheduler.options = playbackOptions();
  scheduler.getCurrentBeat = () => currentBeat;
  scheduler.rebaseTimeline = (beat) => rebases.push(beat);

  scheduler.setLoop({ enabled: false, start: 4, end: 8 });
  assert.equal(scheduler.options.loopEnabled, false);
  assert.deepEqual(rebases, [6]);

  scheduler.setLoop({ enabled: true, start: 4, end: 8 });
  assert.equal(scheduler.options.loopEnabled, true);
  assert.deepEqual(rebases, [6, 6]);

  currentBeat = 9;
  scheduler.setLoop({ enabled: true, start: 4, end: 8 });
  assert.equal(
    scheduler.options.loopEnabled,
    false,
    "a loop end behind the playhead must not spin or rewind immediately",
  );
  assert.deepEqual(rebases, [6, 6, 9]);

  currentBeat = 5;
  scheduler.setLoop({ enabled: true, start: 7, end: 7 });
  assert.equal(scheduler.options.loopStart, 7);
  assert.equal(scheduler.options.loopEnd, 7.25);
  assert.equal(scheduler.options.loopEnabled, true);
  assert.deepEqual(rebases, [6, 6, 9, 5]);

  currentBeat = 5.5;
  scheduler.setTempo(120);
  assert.equal(scheduler.options.tempo, 120);
  assert.deepEqual(rebases, [6, 6, 9, 5, 5.5]);
  scheduler.setTempo(120);
  assert.deepEqual(
    rebases,
    [6, 6, 9, 5, 5.5],
    "setting the same tempo must not disturb playback",
  );

  scheduler.setTimeSignature({ numerator: 6, denominator: 8 });
  assert.deepEqual(scheduler.options.timeSignature, {
    numerator: 6,
    denominator: 8,
  });
  scheduler.setMetronomeEnabled(false);
  assert.equal(scheduler.options.metronomeEnabled, false);

  console.log("Live audio scheduler checks passed");
} finally {
  await vite.close();
}
