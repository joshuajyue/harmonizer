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

function drawContext() {
  const labels = [];
  const noop = () => {};
  return {
    labels,
    context: {
      canvas: { width: 1_024, height: 600 },
      beginPath: noop,
      clearRect: noop,
      fill: noop,
      fillRect: noop,
      fillText: (text) => labels.push(String(text)),
      lineTo: noop,
      moveTo: noop,
      restore: noop,
      rotate: noop,
      roundRect: noop,
      save: noop,
      setLineDash: noop,
      setTransform: noop,
      stroke: noop,
      strokeRect: noop,
      translate: noop,
    },
  };
}

function result(engine, end, roman, violationStart) {
  return {
    engine,
    latencyMs: 1,
    key: { tonic: 0, mode: "major", confidence: 1 },
    voices: [
      {
        name: "soprano",
        notes: [
          {
            pitch: engine === "A" ? 72 : 74,
            start: 0,
            duration: end,
            velocity: 80,
          },
        ],
      },
    ],
    chords: [
      {
        start: 0,
        duration: end,
        roman,
        root: 0,
        quality: "maj",
        inversion: 0,
        secondaryOf: null,
        extensions: [],
        substitutionOf: null,
        substitutionKind: null,
      },
    ],
    violations: [
      {
        kind: "voice_crossing",
        severity: "warning",
        start: violationStart,
        voices: ["alto", "tenor"],
        message: `${engine} violation`,
      },
    ],
  };
}

try {
  const { drawRoll } = await vite.ssrLoadModule(
    "/src/components/editor/drawRoll.ts",
  );
  const { createRollLayout } = await vite.ssrLoadModule(
    "/src/components/editor/rollGeometry.ts",
  );
  const { pieceLength } = await vite.ssrLoadModule("/src/utils/music.ts");
  const resultA = result("A", 4, "A-ONLY", 2);
  const resultB = result("B", 12, "B-ONLY", 9);
  const melody = {
    notes: [],
    tempo: 96,
    timeSignature: { numerator: 4, denominator: 4 },
  };
  const baseModel = {
    melody,
    resultA,
    resultB,
    selectedNotes: [],
    pxPerBeat: 32,
    duration: 12,
    loopEnabled: false,
    loopStart: 0,
    loopEnd: 12,
    voiceVisibility: {
      soprano: true,
      alto: true,
      tenor: true,
      bass: true,
    },
    layout: createRollLayout(),
    empty: false,
  };

  const aContext = drawContext();
  const drawnA = drawRoll(aContext.context, {
    ...baseModel,
    viewMode: "A",
    activeSlot: "A",
  });
  assert.equal(drawnA.noteHits.length, 1);
  assert.equal(drawnA.noteHits[0].slot, "A");
  assert.equal(drawnA.noteHits[0].note.pitch, 72);
  assert.deepEqual(
    drawnA.violationHits.map(({ slot, violation }) => [
      slot,
      violation.start,
    ]),
    [["A", 2]],
  );
  assert.equal(aContext.labels.includes("A-ONLY"), true);
  assert.equal(aContext.labels.includes("B-ONLY"), false);

  const bContext = drawContext();
  const drawnB = drawRoll(bContext.context, {
    ...baseModel,
    viewMode: "B",
    activeSlot: "B",
  });
  assert.equal(drawnB.noteHits.length, 1);
  assert.equal(drawnB.noteHits[0].slot, "B");
  assert.equal(drawnB.noteHits[0].note.pitch, 74);
  assert.deepEqual(
    drawnB.violationHits.map(({ slot, violation }) => [
      slot,
      violation.start,
    ]),
    [["B", 9]],
  );
  assert.equal(bContext.labels.includes("B-ONLY"), true);
  assert.equal(bContext.labels.includes("A-ONLY"), false);

  assert.equal(pieceLength(melody, [resultA]), 4);
  assert.equal(pieceLength(melody, [resultB]), 12);

  console.log("Active comparison slot checks passed");
} finally {
  await vite.close();
}
