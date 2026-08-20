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

### Did the learned model become more musically active?

Yes, and it overshoots. Reported as a percentage of Bach, separating the two
things that were previously conflated — *harmonic rhythm* (how often the chord
changes) from *diminution* (inner-voice motion within a held harmony):

| engine | harmonic rhythm | diminution |
| --- | --- | --- |
| `fixed_thirds` | 89% | 54% |
| `rules` | **105%** | 65% |
| `neural` | 97% | 82% |
| `neural_refine` | 101% | 102% |
| `neural_vl` | 97% | **106%** |

**No engine has a harmonic-rhythm problem** — the rule engine slightly exceeds
Bach. The entire gap that motivated the "creativity, not rule-compliance"
reframe is diminution, and for the rule engine it is a representational ceiling
rather than conservatism: it freezes one inner-voice sonority per beat slot and
scores exactly 0.0 on off-beat inner-voice motion. It cannot decorate a held
harmony at any setting.

`neural_vl` closes that gap and overshoots to 106%. Whether overshooting Bach is
good is a matter of taste, and nothing in this harness can settle it.

Jazz reharm (`ml/reharm/REPORT.md`) reached a comparable nuance: sampling gives
genuine one-to-many variety (five draws differ in 22% of chord roots; the
deterministic engine differs in 0%), but sampling does **not** give adventure —
hand-written substitution rules are *more* chromatic than the learned sampler,
because a model of 1,170 standards puts its mass where the corpus does, and the
middle of the jazz corpus is a diatonic ii-V-I. The shipped engine samples from a
hybrid.

## Bugs found and fixed this session

All of these are now closed and regression-tested. Kept because the root causes
are worth knowing if any of them regress.

1. **Marquee delete left notes rendered.** A four-way asymmetry: `duration` and
   rendering covered both comparison slots, but only the active slot was editable
   and only the active slot was played. Fixed by dropping the translucent overlay
   in favour of showing one slot at a time, which was also the requested feature.
2. **Reharm engines doubled the melody start offset** — and the visible timing
   error was the smaller half. The rules engine reports chords in absolute time
   while the reharm skeleton rebased the melody to zero, so the
   melody-compatibility hard constraint, the thing that engine exists to enforce,
   was receiving the wrong melody notes for any tune not starting at beat 0. Now
   works in one frame throughout. Verified across offsets 0, 4, 7 and 16 on all
   seven engines.
3. **Loop toggle did nothing mid-playback** — loop config was passed into
   `audioScheduler.start()` once and snapshotted. Now has a live `setLoop()` path;
   tempo and meter got the same treatment.
4. **`l` was both a note and the loop toggle**, with `h` a latent second
   collision. Fixed structurally: one `CHARACTER_SHORTCUTS` table plus a startup
   assertion against `MUSICAL_TYPING_KEY_SET`, so the next shortcut anyone adds
   cannot silently collide. Loop moved to `/`, harmonize to `Cmd/Ctrl+Enter`.
5. **Unarmed keyboard input stopped placing notes**, a regression from the
   record-mode work. Both behaviours now exist behind an explicit Record/Place
   toggle.
6. **Recording quantization truncated durations.** Now moves onsets to the grid
   and preserves length.
7. **The bass preview voice was inaudible.** `synthVoice.ts` used `bass: "sine"`,
   and a pure sine at 82 Hz puts nothing in the band a laptop speaker can
   reproduce, with no harmonics for the ear to infer the missing fundamental from.
   Now sawtooth with register compensation, and the four voices are timbrally
   distinguishable.

## A pattern worth carrying forward

Three separate faults this session — the fixture's voicing defects, a register
collapse, and the offset bug above — were each caught by a **test asserting a
property**, and none of them by the metric suite. As `jazz-reharm` put it after
the last one: every evaluation melody starts at beat 0, so no number in its
report could ever have moved. Metrics measure what you thought to measure;
invariants catch what you did not.

The same lesson applied to the contract guard, which could not detect the exact
failure it existed to prevent until it was given self-tests that deliberately
break it.

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
