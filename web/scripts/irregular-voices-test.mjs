import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const fixture = JSON.parse(
  await readFile(
    new URL(
      "../../contracts/examples/harmonize.response.json",
      import.meta.url,
    ),
  ),
);
const request = JSON.parse(
  await readFile(
    new URL(
      "../../contracts/examples/melody.request.json",
      import.meta.url,
    ),
  ),
);
const vite = await createServer({
  root,
  logLevel: "error",
  server: { middlewareMode: true },
  appType: "custom",
});

const noop = () => {};
const context = {
  beginPath: noop,
  fill: noop,
  fillRect: noop,
  fillText: noop,
  lineTo: noop,
  moveTo: noop,
  restore: noop,
  rotate: noop,
  roundRect: noop,
  save: noop,
  setLineDash: noop,
  stroke: noop,
  translate: noop,
};

try {
  const { drawVoiceNotes } = await vite.ssrLoadModule(
    "/src/components/editor/drawNotes.ts",
  );
  const { drawChords, drawViolations } = await vite.ssrLoadModule(
    "/src/components/editor/drawAnnotations.ts",
  );
  const {
    contains,
    createRollLayout,
    intersects,
  } = await vite.ssrLoadModule(
    "/src/components/editor/rollGeometry.ts",
  );
  const { pieceLength } = await vite.ssrLoadModule("/src/utils/music.ts");
  const { buildSelectionTransform } = await vite.ssrLoadModule(
    "/src/store/selection.ts",
  );
  const { buildSelectionDelete } = await vite.ssrLoadModule(
    "/src/store/selectionEdits.ts",
  );

  const counts = Object.fromEntries(
    fixture.voices.map((voice) => [voice.name, voice.notes.length]),
  );
  assert.deepEqual(counts, {
    soprano: 16,
    alto: 9,
    tenor: 13,
    bass: 14,
  });
  for (const voice of fixture.voices) {
    assert.equal(voice.notes[0].start, 0);
    for (let index = 1; index < voice.notes.length; index += 1) {
      const previous = voice.notes[index - 1];
      assert.equal(
        voice.notes[index].start,
        previous.start + previous.duration,
        `${voice.name} must stay contiguous`,
      );
    }
    const last = voice.notes.at(-1);
    assert.equal(last.start + last.duration, 32);
  }

  const layout = createRollLayout();
  const model = {
    layout,
    pxPerBeat: 32,
    selectedNotes: [],
    viewMode: "A",
    voiceVisibility: {
      soprano: true,
      alto: true,
      tenor: true,
      bass: true,
    },
  };
  const hits = [];
  drawVoiceNotes(fixture, "A", model, hits, context, new Set());
  assert.equal(hits.length, 52);
  assert.deepEqual(
    Object.fromEntries(
      ["soprano", "alto", "tenor", "bass"].map((name) => [
        name,
        hits.filter((hit) => hit.voice === name).length,
      ]),
    ),
    counts,
  );

  const sustained = hits.find(
    (hit) => hit.voice === "alto" && hit.note.duration === 8,
  );
  assert.ok(sustained, "the eight-beat alto note must be drawn");
  assert.equal(sustained.rect.width, 8 * model.pxPerBeat - 4);
  const centerY = sustained.rect.y + sustained.rect.height / 2;
  for (const x of [
    sustained.rect.x + 1,
    sustained.rect.x + sustained.rect.width / 2,
    sustained.rect.x + sustained.rect.width - 1,
  ]) {
    assert.equal(
      contains(sustained.rect, x, centerY),
      true,
      "a sustained note must hit-test across its full drawn width",
    );
  }
  assert.equal(
    intersects(sustained.rect, {
      x: sustained.rect.x + sustained.rect.width - 2,
      y: centerY - 2,
      width: 4,
      height: 4,
    }),
    true,
    "a marquee crossing the tail must select the sustained note",
  );

  const selection = {
    source: "voice",
    slot: "A",
    voice: "alto",
    index: sustained.index,
  };
  const state = {
    melody: request.melody,
    melodyRevision: 0,
    snap: 0.25,
    viewMode: "A",
    selectedNotes: [selection],
    slots: {
      A: { engineId: "rules", status: "ready", result: fixture },
      B: { engineId: "learned", status: "idle" },
    },
  };
  const moved = buildSelectionTransform(
    state,
    [{ selection, note: { ...sustained.note } }],
    0.25,
    1,
  );
  const movedNote = moved.slots.A.result.voices
    .find((voice) => voice.name === "alto")
    .notes[sustained.index];
  assert.equal(movedNote.start, sustained.note.start + 0.25);
  assert.equal(movedNote.pitch, sustained.note.pitch + 1);
  assert.equal(movedNote.duration, 8);

  const deleted = buildSelectionDelete(state);
  assert.deepEqual(
    Object.fromEntries(
      deleted.slots.A.result.voices.map((voice) => [
        voice.name,
        voice.notes.length,
      ]),
    ),
    { ...counts, alto: 8 },
    "deleting one held alto note must not affect other voices",
  );

  assert.equal(pieceLength(request.melody, [fixture]), 32);
  const chordRects = [];
  drawChords(
    {
      ...context,
      roundRect: (...args) => chordRects.push(args),
    },
    fixture.chords,
    model,
  );
  assert.equal(chordRects.length, fixture.chords.length);
  fixture.chords.forEach((chord, index) => {
    assert.equal(chordRects[index][0], chord.start * model.pxPerBeat + 2);
    assert.equal(
      chordRects[index][2],
      Math.max(8, chord.duration * model.pxPerBeat - 4),
    );
  });

  const violationHits = [];
  drawViolations(context, fixture, "A", model, violationHits);
  assert.deepEqual(
    violationHits.map(({ violation }) => ({
      start: violation.start,
      voices: violation.voices,
    })),
    [
      { start: 9, voices: ["tenor", "bass"] },
      { start: 22, voices: ["soprano", "bass"] },
      { start: 25, voices: ["alto", "tenor"] },
    ],
  );
  violationHits.forEach((hit) => {
    assert.equal(
      hit.rect.x + 9,
      hit.violation.start * model.pxPerBeat,
    );
  });

  console.log("Irregular voice fixture checks passed");
} finally {
  await vite.close();
}
