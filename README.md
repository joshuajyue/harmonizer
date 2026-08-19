# HarmonAIzer

Give it a melody. It writes the other voices.

Not a pitch-shifter. The commodity part of a harmonizer — timbre-preserving
resynthesis — is a solved, downloadable problem. The unsolved part, and the whole
point of this project, is the musical decision: **which notes should the
accompanying voices actually sing.** Commercial harmonizers are dumb here. They
lock to a scale, stack a fixed interval, and have no notion of harmonic function
or voice leading.

## Structure

| Path         | Owns                                                                 |
| ------------ | -------------------------------------------------------------------- |
| `contracts/` | The frozen API contract. `types.ts` and `schema.py` are mirrors — change together. |
| `ml/`        | Harmony engines (rule-based and learned) plus the evaluation harness. |
| `backend/`   | FastAPI service and the audio synthesis / transcription pipeline.     |
| `web/`       | React + TypeScript studio UI.                                         |

## Design commitments

**Engines return voiced parts, not chord labels.** `ml/engines/base.py` defines
one interface: melody in, fully voiced SATB out. v1 emitted a chord symbol per
beat and delegated the actual notes to a fixed renderer, which meant voice leading
was structurally unlearnable. Voice leading is the problem, so it lives inside the
engine where it can be optimized and measured.

**Every engine is scored by the same harness.** A rule engine and a neural engine
are directly comparable on identical held-out data and identical metrics
(`ml/eval/`). v1 had no such harness, so "the rules beat the model" was a vibe
rather than a measurement — and the per-beat accuracy it did report was inflated
by unmasked padding.

**Defects are surfaced, not hidden.** Parallel fifths, voice crossings, unresolved
leading tones and spacing errors come back in the API response as `violations[]`
and are drawn on the timeline. Showing *why* a harmonization is weak is something
no commercial tool bothers to do.

**This is a post-hoc harmonizer, not a real-time one.** That is a deliberate
scope boundary, and the measurements say it is the right one.

| engine         | 1 note  | 16 notes | per note |
| -------------- | ------- | -------- | -------- |
| `rules`        | 1.05 ms | 52.7 ms  | ~3.7 ms  |
| `fixed_thirds` | 0.37 ms | 5.8 ms   | ~0.37 ms |

The offline engine is not too slow for live use — at 1-4 ms per note it fits
inside a 10 ms budget comfortably. It is *non-causal*. `rules` runs Viterbi over
the whole melody, and its entire value is revising early chord choices once it
knows where the phrase cadences. It cannot emit chord 1 before it has seen chord
16, at any speed.

So a live harmonizer is a different engine, not a faster one: causal beam search
with a one-sonority lookback, which the `HarmonyEngine` interface already admits
as a future implementation. Worth noting that causality forbids *planning*, not
*voice leading* — avoiding parallel fifths only requires knowing the previous
sonority, so a real-time engine could still be far better than `fixed_thirds`,
which keeps no state at all. Pursuing both at once would compromise the offline
engine for nothing.

## Where it stands

Same 8-bar melody, same harness, two engines:

```
rules          0 violations   I V65 I ii7 V I V I IV V I V65 I ii7 V I
fixed_thirds  26 violations   IV64 vi64 I64 vi64 viio64 vi64 V64 ...
                              (11 parallel fifths + 15 parallel octaves)
```

`fixed_thirds` is the commercial-harmonizer floor: for each melody note it emits
voices a fixed diatonic third, fifth and octave below. Both of its failure modes
are structural rather than incidental. The bass is always the melody an octave
down, so it never chooses a root and lands on inversion after inversion; and the
voices sit at a fixed scale distance, so every melodic leap moves them all in
lockstep and manufactures parallels. You cannot write that algorithm without
them.

That is the number the learned engine has to beat — not just on defect count,
which the rule engine already drives to zero, but on musical interest while
staying clean.

## Running it

```bash
docker compose up --build
```

Frontend on <http://localhost:5173>, API on <http://localhost:8000>. nginx
reverse-proxies `/api/*` to the backend, so there is no CORS setup.

For development, see `backend/README.md` and `web/README.md`.

## History

v1 lives on the `main` branch and is kept only as a post-mortem. Its failure
modes are documented in `ml/eval/REPORT.md`; briefly, the learned model lost to
its own rule engine because the representation destroyed the signal — absolute
pitch classes against tonic-relative labels, a 7-triad vocabulary that could not
represent the training corpus, unmasked padding in the loss, and no transition
model at all.
