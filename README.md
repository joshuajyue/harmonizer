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
