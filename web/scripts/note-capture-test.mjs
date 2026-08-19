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

function closeTo(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < 0.0001, message);
}

try {
  const { useStudioStore } = await vite.ssrLoadModule("/src/store/index.ts");
  const { noteCapture } = await vite.ssrLoadModule(
    "/src/capture/NoteCapture.ts",
  );
  const { audioScheduler } = await vite.ssrLoadModule(
    "/src/audio/AudioScheduler.ts",
  );
  let schedulerBeat;
  const previewed = [];
  const released = [];
  audioScheduler.getCurrentBeat = () => schedulerBeat;
  audioScheduler.previewNoteOn = (pitch, velocity) => {
    previewed.push({ pitch, velocity });
  };
  audioScheduler.previewNoteOff = (pitch) => {
    released.push(pitch);
  };
  audioScheduler.previewAllNotesOff = () => {};
  audioScheduler.stop = () => {};

  let state = useStudioStore.getState();
  state.clearMelody();
  state.setCurrentBeat(0);
  state.setNoteInputMode("record");
  state.setRecordingState("idle");
  schedulerBeat = 0;
  noteCapture.noteOn(60, 91);
  noteCapture.noteOff(60);
  state = useStudioStore.getState();
  assert.equal(state.melody.notes.length, 0);
  assert.deepEqual(previewed, [{ pitch: 60, velocity: 91 }]);
  assert.deepEqual(released, [60]);

  state.setNoteInputMode("step");
  state.setCurrentBeat(0.37);
  schedulerBeat = 0.37;
  noteCapture.noteOn(62, 83);
  noteCapture.noteOff(62);
  state = useStudioStore.getState();
  assert.deepEqual(state.melody.notes[0], {
    pitch: 62,
    start: 0.25,
    duration: 1,
    velocity: 83,
  });
  assert.equal(state.currentBeat, 1.25);

  state.clearMelody();
  state.setNoteInputMode("record");
  state.setRecordingState("recording");
  schedulerBeat = 2.13;
  noteCapture.noteOn(64, 72);
  noteCapture.finishTake(3.63);
  state = useStudioStore.getState();
  assert.equal(state.melody.notes.length, 1);
  closeTo(state.melody.notes[0].start, 2.13, "recorded onset must stay raw");
  closeTo(
    state.melody.notes[0].duration,
    1.5,
    "stop must preserve the held duration",
  );

  state.clearMelody();
  schedulerBeat = 4;
  noteCapture.noteOn(67, 40);
  schedulerBeat = 4.5;
  noteCapture.noteOn(67, 100);
  noteCapture.finishTake(5.25);
  state = useStudioStore.getState();
  assert.equal(state.melody.notes.length, 2);
  assert.deepEqual(
    state.melody.notes.map(({ velocity }) => velocity),
    [40, 100],
  );
  closeTo(state.melody.notes[0].duration, 0.5, "retrigger must close first note");
  closeTo(state.melody.notes[1].duration, 0.75, "retrigger must open next note");

  state.clearMelody();
  schedulerBeat = 6;
  noteCapture.noteOn(60, 70);
  schedulerBeat = 6.25;
  noteCapture.noteOn(64, 80);
  noteCapture.finishTake(7);
  state = useStudioStore.getState();
  assert.equal(state.melody.notes.length, 2);
  closeTo(state.melody.notes[0].duration, 1, "first polyphonic note duration");
  closeTo(
    state.melody.notes[1].duration,
    0.75,
    "second polyphonic note duration",
  );
  assert.equal(noteCapture.openNoteCount, 0);
  state.setRecordingState("idle");

  console.log("Shared note capture checks passed");
} finally {
  await vite.close();
}
