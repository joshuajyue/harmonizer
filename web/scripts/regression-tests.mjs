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

try {
  const {
    CHARACTER_SHORTCUTS,
  } = await vite.ssrLoadModule("/src/input/globalShortcuts.ts");
  const {
    MUSICAL_TYPING_KEYS,
    MUSICAL_TYPING_OFFSETS,
  } = await vite.ssrLoadModule("/src/input/musicalTyping.ts");
  const {
    buildSelectionQuantize,
  } = await vite.ssrLoadModule("/src/store/selectionEdits.ts");

  assert.equal(
    new Set(MUSICAL_TYPING_KEYS).size,
    MUSICAL_TYPING_KEYS.length,
    "musical typing keys must be unique",
  );
  MUSICAL_TYPING_KEYS.forEach((key, index) => {
    assert.equal(MUSICAL_TYPING_OFFSETS.get(key), index);
  });
  const shortcutKeys = new Set(
    CHARACTER_SHORTCUTS.map(({ key }) => key),
  );
  assert.equal(shortcutKeys.has("l"), false);
  assert.equal(shortcutKeys.has("h"), false);
  assert.deepEqual(
    Object.fromEntries(
      CHARACTER_SHORTCUTS.map(({ key, action }) => [key, action]),
    ),
    {
      "/": "loop",
      m: "metronome",
      r: "record",
      "1": "result-a",
      "2": "result-b",
    },
  );

  const melodyNote = {
    pitch: 60,
    start: 0.37,
    duration: 0.73,
    velocity: 90,
  };
  const activeVoiceNote = {
    pitch: 72,
    start: 0.63,
    duration: 1.37,
    velocity: 80,
  };
  const inactiveVoiceNote = {
    pitch: 71,
    start: 0.38,
    duration: 0.91,
    velocity: 78,
  };
  const state = {
    melody: {
      notes: [melodyNote],
      tempo: 96,
      timeSignature: { numerator: 4, denominator: 4 },
    },
    melodyRevision: 7,
    snap: 0.25,
    viewMode: "A",
    selectedNotes: [
      { source: "melody", index: 0 },
      {
        source: "voice",
        slot: "A",
        voice: "soprano",
        index: 0,
      },
      {
        source: "voice",
        slot: "B",
        voice: "soprano",
        index: 0,
      },
    ],
    slots: {
      A: {
        engineId: "rules",
        status: "ready",
        result: {
          voices: [{ name: "soprano", notes: [activeVoiceNote] }],
          chords: [],
          violations: [],
        },
      },
      B: {
        engineId: "learned",
        status: "ready",
        result: {
          voices: [{ name: "soprano", notes: [inactiveVoiceNote] }],
          chords: [],
          violations: [],
        },
      },
    },
  };

  const update = buildSelectionQuantize(state);
  assert.equal(update.melody.notes[0].start, 0.25);
  assert.equal(update.melody.notes[0].duration, melodyNote.duration);
  assert.equal(update.melodyRevision, 8);
  assert.equal(
    update.slots.A.result.voices[0].notes[0].start,
    0.75,
  );
  assert.equal(
    update.slots.A.result.voices[0].notes[0].duration,
    activeVoiceNote.duration,
  );
  assert.equal(
    update.slots.B.result.voices[0].notes[0],
    inactiveVoiceNote,
    "inactive result must remain untouched",
  );
  assert.equal(melodyNote.start, 0.37, "source notes must not be mutated");
  assert.equal(activeVoiceNote.start, 0.63, "source voices must not be mutated");

  const alreadySnapped = {
    ...state,
    melody: {
      ...state.melody,
      notes: [{ ...melodyNote, start: 0.5 }],
    },
    selectedNotes: [{ source: "melody", index: 0 }],
  };
  assert.deepEqual(buildSelectionQuantize(alreadySnapped), {});

  console.log("Frontend regression checks passed");
} finally {
  await vite.close();
}
