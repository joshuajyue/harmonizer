# Project status — 2026-08-19

A snapshot for picking this up cold. Written at the end of a long unattended
build session; assumes you remember nothing.

## Where it stands

Working end to end. Record or draw a melody, harmonize it with any of eight
engines, see the voices and their voice-leading defects on a piano roll, render
to audio, export MIDI.

| area | lines | state |
| ---- | ----- | ----- |
| `contracts/` | 1,099 | stable, guarded by a drift test |
| `ml/` | 35,500 | Bach engines + eval harness complete; jazz reharm complete |
| `backend/` | 3,983 | complete, reviewed, hardened |
| `web/` | 8,619 | working, several open bugs (below) |

## Running it

Servers are **not** currently running. Start from the repository root — the
backend uses absolute imports and fails cryptically if launched from inside
`backend/`.

```bash
HARMONIZER_ENABLE_DEV_ENGINE=1 backend/.venv/bin/uvicorn backend.main:app --port 8000 --reload
cd web && npm run dev
```

Use `127.0.0.1`, not `localhost` — it resolves to IPv6 on this machine and hits
the wrong server. Add `--reload-dir backend --reload-dir ml` to pick up engine
changes without a restart.

Tests: `backend/.venv/bin/python -m pytest backend/tests contracts` and
`ml/.venv/bin/python -m pytest ml/tests contracts`. The ML suite takes ~3
minutes. `python contracts/test_contract_sync.py` must always exit 0.

## The headline result

From `ml/eval/REPORT.md`, on 61 held-out Bach chorales, every engine scored by
identical detectors:

| | voice-leading errors / 100 chords | style distance from Bach |
| --- | --- | --- |
| `fixed_thirds` (commercial-harmonizer floor) | 159.84 | 0.421 |
| `rules` | 0.03 | 0.202 (3.6x noise floor) |
| `neural` | 10.47 | 0.060 (at noise floor) |
| `neural_vl` | 0.12 | 0.073 (1.3x) |

The two systems fail in *orthogonal* ways. The rule engine is a flawless
contrapuntist with a narrow harmonic imagination; the learned model has Bach's
harmonic vocabulary and writes parallel fifths at twenty times his rate.
`neural_vl` — the model's harmonic choice under a voice-leading veto — keeps both
halves. So the answer to "can a learned model beat a strong rule engine" is: only
as a composition of the two, and only once the metrics could tell the difference.

The single most valuable thing in the harness is the **Bach oracle**: Bach's own
four parts scored by our own detectors. It is what revealed that `rules` scoring
0.00 was *undershooting* — Bach scores 4.13, and a system cleaner than Bach is
rigid, not good.

Jazz reharm (`ml/reharm/REPORT.md`) reached a comparable nuance: sampling gives
genuine one-to-many variety (five draws differ in 22% of chord roots; the
deterministic engine differs in 0%), but sampling does **not** give adventure —
hand-written substitution rules are *more* chromatic than the learned sampler,
because a model of 1,170 standards puts its mass where the corpus does, and the
middle of the jazz corpus is a diatonic ii-V-I. The shipped engine samples from a
hybrid.

## Known open bugs

1. **Marquee delete leaves notes rendered.** Only reproduces with **both** A and
   B harmonized. Root cause is a four-way asymmetry: `duration` and rendering
   cover both slots, but only the active slot is editable
   (`drawNotes.ts:167`, `editable: slot === model.activeSlot`) and only the
   active slot is played. Deleting removes active-slot notes; inactive-slot
   notes stay drawn, and playback stops early. Dropping the translucent overlay
   in favour of showing one slot at a time is both the requested feature and the
   cleanest fix.
2. **Harmonization ignores melody start offset.** A melody starting at beat 4
   returns harmony starting at beat 0. Reproduced against the `rules` engine
   through the API. Also silently violates the contract invariant that soprano
   retains the input melody — `test_soprano_retains_the_input_melody` passes only
   because the canonical fixture happens to start at beat 0. Likely in the shared
   `ml/data/melody.py` grid conversion, in which case every engine has it.
3. **Loop toggle does nothing mid-playback.** `usePlayback.ts` passes
   `loopEnabled/loopStart/loopEnd` into `audioScheduler.start({...})` once and
   the scheduler keeps that snapshot. `metronomeEnabled` and tempo are passed the
   same way and likely fail the same way.
4. **Keyboard collisions.** Musical typing uses `a w s e d f t g y h u j k o l p ;`
   (`VirtualKeyboard.tsx:9-27`). `l` is both a note and the loop toggle. `h` is a
   latent second collision. Needs a reserved-key set with a startup assertion,
   not a one-off patch.
5. **Unarmed keyboard input regressed** — playing while not recording no longer
   places notes. Both modes are wanted behind an explicit toggle.
6. **Recording quantization truncates durations.** Should move onsets to the grid
   and preserve length.
7. **Bass preview voice is inaudible.** `synthVoice.ts` uses `bass: "sine"`; a
   pure sine at 82 Hz puts nothing in the band a laptop speaker can reproduce,
   and has no harmonics for the ear to infer the missing fundamental from. Needs
   a harmonically rich waveform. Note `alto`/`tenor` are already `triangle`.

## Ownership map

Enforced strictly all session; zero merge conflicts across 40+ commits.

| path | owner |
| ---- | ----- |
| `contracts/` | lead only — agents request changes rather than editing |
| `ml/` (except `reharm/`) | ml-harmony |
| `ml/reharm/` | jazz-reharm |
| `backend/`, `docker-compose.yml` | backend-api |
| `web/` | web-ui |

Engines self-register via `ml.engines.base.register()`; the backend discovers
them by scanning both `ml.engines` and `ml.reharm`. A new engine appears in the
API and the frontend's comparison UI with no changes to either — this has been
verified repeatedly and is the seam that made parallel work possible.

## Design commitments worth preserving

- **Engines return voiced parts, not chord labels.** v1 emitted a chord symbol
  per beat and delegated notes to a fixed renderer, which made voice leading
  structurally unlearnable.
- **Defects are surfaced, not hidden** — `violations[]` on the response, drawn on
  the timeline. Same principle drives `substitutionOf`/`substitutionKind` for
  reharmonization: explain the decision, do not just emit it.
- **Silent defaults are bugs.** `tempo` is required precisely because defaulting
  it renders everything at the wrong speed with no error. The same class of bug
  was later found in MIDI export (hard-coded 4/4) and fixed.
- **Post-hoc, not real-time.** Measured at 1-4 ms/note, so speed is not the
  constraint — Viterbi is non-causal and cannot emit chord 1 before seeing chord
  16. A live harmonizer is a *different engine*, not a faster one. Worth noting
  causality forbids planning but not voice leading, so a real-time engine could
  still be far better than `fixed_thirds`.

## Where this could go next

The rules engine has a structural ceiling it cannot cross: it writes one frozen
voicing per beat slot, so it **cannot emit a passing tone**. A literature review
traced most of the Bach-vs-rules activity gap to exactly this — diminution, not
harmony. Constrained stochastic elaboration on top of the rule skeleton is the
open thread with the clearest payoff.
