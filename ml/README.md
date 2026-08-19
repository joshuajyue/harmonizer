# `ml/` — harmony intelligence

The musical decision: given a melody, which notes should the accompanying voices
sing. Timbre-preserving resynthesis is a solved commodity; this is the part that
is not.

## Layout

```
theory/      music-theory primitives: key normalization, chord vocabulary,
             voice-leading detectors. Heavily unit-tested; everything else
             depends on them being right.
data/        Bach chorale corpus loading, quantization, splits; and the
             Melody <-> grid conversions the engines share.
engines/     HarmonyEngine implementations. Importing a module registers it.
             base.py is the shared interface (also imported by the backend).
nn/          The learned model: tokenisation and the network.
training/    Scripts that produce checked-in artefacts (priors, tuned weights,
             the model checkpoint). None of them run at request time.
experiments/ One-off measurements that produce numbers for the report. Imported
             by nothing; safe to delete once the report is read.
eval/        The evaluation harness and REPORT.md.
tests/       Unit tests, an engine contract suite every registered engine must
             pass, conformance tests against the shared API contract, and the
             detector checks against the lead's shared fixtures.
```

## Engines

| id              | learned | what it is |
|-----------------|---------|------------|
| `fixed_thirds`  | no      | Scale-locked parallel intervals — the commercial-harmonizer baseline, present as the floor of the comparison. |
| `rules`         | no      | Functional-harmony Viterbi over a full chord vocabulary, then a voice-leading Viterbi over actual SATB voicings. |
| `neural`        | yes     | Masked model over all four voices, tonic-relative, decoded by annealed blocked Gibbs. Kept unpolished as the scientific control. |
| `neural_vl`     | yes     | The same model, then each voice re-solved by Viterbi under the voice-leading rules. **The one to ship.** |
| `neural_refine` | yes     | The same model seeded with the rule engine's draft, acting as a reviser. A negative result, kept because it is informative. |

Results on the 61 held-out chorales (full detail in `eval/REPORT.md`):

|                | defects /100 | style JS from Bach | chords/piece | beats on I or V |
|----------------|--------------|--------------------|--------------|-----------------|
| `fixed_thirds` | 159.84       | 0.421              | 6.7          | 36.2%           |
| `rules`        | 0.03         | 0.202              | 9.4          | 68.6%           |
| `neural`       | 10.47        | **0.060**          | 12.6         | 52.9%           |
| `neural_vl`    | 0.12         | 0.071              | 11.7         | 53.2%           |
| **Bach**       | **3.73**     | **0.056**          | **14.5**     | **48.6%**       |

**Read every column against the Bach row, and do not read the first column as a
leaderboard.** Bach breaks his own voice-leading rules 3.73 times per 100 chords;
an engine that never breaks them is not better than Bach, it is stiffer. Defects
are a guardrail here, not the objective — the objective is stylistic fidelity and
harmonic interest.

So `rules` at 0.03 is *undershooting*, and columns 3 and 4 say why: it plays 9.4
distinct chords per piece against Bach's 14.5 and spends 68.6% of its beats on I
or V against Bach's 48.6%. It is clean because it is narrow.

`neural_vl` is also near zero — and is *not* narrow: 11.7 chords per piece, 53.2%
on I or V, and more voice motion than Bach. Same defect rate, opposite character.
The number to watch is not the defect rate, it is what the engine is willing to
play. Held-out Bach differs from training Bach by 0.056, so `neural`'s 0.060 is
not "close to Bach" — it is indistinguishable from Bach at this sample size.

Engines return **fully voiced parts**, not a chord label per beat. Chord labels
are metadata derived from the voices. That is the central correction to v1,
where a fixed renderer chose the notes and voice leading was therefore
structurally impossible.

## Running things

```bash
python -m ml.eval.run                    # score every engine, write eval/REPORT.md
python -m ml.eval.run --detect-key       # also run the realistic melody-only-key setting
python -m pytest ml/tests -q             # unit, engine contract, API conformance
python3 contracts/test_contract_sync.py  # the lead's TS/Pydantic drift guard

python -m ml.training.calibrate_rules    # refit chord priors from the training split
python -m ml.training.tune_rules         # tune rule weights against the harness
python -m ml.training.train_neural       # train the shipped checkpoint
python -m ml.training.train_neural --ablation   # the representation experiment
python -m ml.experiments.v1_postmortem   # the autopsy: one factor at a time
```

Everything runs on a laptop CPU. Training the shipped model takes roughly half
an hour; harmonizing one chorale takes a fraction of a second for `rules` and
about a second for `neural`.

## Reading the results

Start at [`eval/REPORT.md`](eval/REPORT.md). Two things to know before reading
any number in it:

* **The Bach oracle row is the calibration.** It is Bach's own four parts pushed
  through the identical detectors. A defect rate is only interpretable relative
  to it, and it is not zero — Bach crosses voices and overlaps parts freely.
* **Chord agreement with Bach is not the headline.** It is reported because it
  is the number v1 optimised, and because watching it move independently of the
  quality metrics is the argument against it.
* **Neither is the defect rate.** It is a guardrail against degenerating into
  parallel thirds. An engine below Bach's rate is not winning; check what it gave
  up to get there.

## The shared contract

`contracts/` is owned by the lead and is **read-only here**. Two guards keep the
two sides honest, and they check different things:

* `contracts/test_contract_sync.py` (the lead's) proves `types.ts` and
  `schema.py` mirror each other.
* `ml/tests/test_contract_conformance.py` (mine) proves the engines actually
  satisfy the Pydantic side.

`contracts/examples/` carries a canonical melody and a harmonization with three
deliberate, independently verified voice-leading defects. Those are the most
valuable tests here, because they are the only ones whose expected answer was
written by someone else: every other check of the detectors was written by the
same person who wrote the detectors, so a shared misconception would pass
silently. `tests/test_shared_fixtures.py` asserts the detectors' findings and
the fixture's declared `violations` agree **exactly, in both directions**, and
it is driven off the fixture rather than hardcoded beats, so a re-voicing needs
no edit here.

It also sweeps every successive pair of sonorities and every voice pair *by
hand*, from raw pitch arithmetic, and fails if it finds a perfect parallel the
fixture does not declare. That check trusts neither the detector nor the
fixture's labels, so it still fires if the two share a mistake — which they did
once: an earlier revision of the fixture carried five unintended parallels while
declaring none, and this is the check that catches that class of error.

The one worth knowing about: `Chord.quality` is a bare `str` in the schema, so
Pydantic will accept `"min11"` and the UI will render something unrecognisable.
The conformance suite parses the quality strings the contract's field
description enumerates and asserts the whole chord vocabulary in
`theory/chords.py` is a subset — currently 10 of the 14 documented values, with
`maj6`, `min6`, `sus2` and `sus4` unused by these engines. Adding a quality here
without the contract knowing about it fails the build rather than shipping a
blank chord symbol.

Response-side fields are populated explicitly rather than left to their
defaults. The contract guarantees the frontend never has to null-check
`chord.extensions` or `response.violations`; a default that happens to produce
the right value is not the same as populating it, and stops being right the
moment the default changes. These engines write common-practice chorale
harmony, so `extensions` is always `[]` and the substitution-provenance fields
are always `None` — `ml/reharm/` is the engine that fills them in.

## Why it is built this way

v1 is gone — deleted from this branch and preserved only in git history on
`main`, where it can be read with `git show main:backend/model.py` and friends.
Nothing here imports it, inherits from it or reuses its structure. It matters
only as an autopsy, because its BiLSTM lost to its own rule engine, and the four
reasons why are the specification for everything above:

1. **Padding polluted the loss — but truncation was the real cost.** The padding
   bug is real (`chord_padding[:, 0] = 1`, no mask in the loss or in
   `evaluate()`), but measured against the corpus it fired on 1 of 368 chorales,
   0.07% of the training grid. The same code path's truncation to 32 quarter
   notes hid **41.2% of the corpus**. Here sequences carry an explicit `valid`
   mask *and* pieces are kept whole. Worth +28% in the reconstruction.
2. **The representation deleted the signal.** Inputs were absolute pitch classes;
   targets were tonic-relative scale degrees. The network had to induce the
   tonic across all twelve transpositions from ~400 chorales given a single
   `is_minor` bit — while the rule engine was handed `key.tonic` for free.
   Everything here is tonic-relative by construction. Worth **+68%** in v1's
   setting — the dominant term. But `--ablation` shows it is worth only 1.5%
   perplexity for *this* architecture, because v2 predicts pitches from pitches
   and the target is already in the input's frame. The real fix was never "use
   tonic-relative pitch"; it was "make the target expressible from the input".
3. **The label space could not represent the data.** Seven diatonic triads, with
   every seventh chord, secondary dominant and borrowed chord projected onto the
   nearest one by a `+1/-1` pitch-class vote — which preserves the chord root
   only 84.5% of the time and the exact chord only 68.9%. Here the model predicts
   the notes themselves and chord labels are derived afterwards, so there is no
   label space to be too small.
4. **There was no transition model.** Independent per-beat softmax and argmax,
   against a rule engine that had an explicit functional grammar. Here decoding
   is iterative and bidirectional: the model revises its own output.

Each of those four claims was checked against the source before being believed,
and one of them turned out to be wrong: see `eval/REPORT.md`. The reconstruction
that established the numbers lives in `experiments/v1_postmortem.py`, quarantined
away from anything the product imports.

The ordering of the work was deliberate — a genuinely strong rules baseline
first, then the evaluation harness, and only then the model. A weak baseline or
an untrustworthy metric would have made any conclusion about the model
worthless, which is precisely what happened before.
