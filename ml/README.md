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
eval/        The evaluation harness and REPORT.md.
tests/       Unit tests, plus a contract suite every registered engine must pass.
```

## Engines

| id              | learned | what it is |
|-----------------|---------|------------|
| `fixed_thirds`  | no      | Scale-locked parallel intervals — the commercial-harmonizer baseline, present as the floor of the comparison. |
| `rules`         | no      | Functional-harmony Viterbi over a full chord vocabulary, then a voice-leading Viterbi over actual SATB voicings. |
| `neural`        | yes     | Masked model over all four voices, tonic-relative, decoded by annealed blocked Gibbs. |
| `neural_refine` | yes     | The same model seeded with the rule engine's draft, acting as a reviser. |

Engines return **fully voiced parts**, not a chord label per beat. Chord labels
are metadata derived from the voices. That is the central correction to v1,
where a fixed renderer chose the notes and voice leading was therefore
structurally impossible.

## Running things

```bash
python -m ml.eval.run                    # score every engine, write eval/REPORT.md
python -m ml.eval.run --detect-key       # also run the realistic melody-only-key setting
python -m pytest ml/tests -q             # unit + engine contract tests

python -m ml.training.calibrate_rules    # refit chord priors from the training split
python -m ml.training.tune_rules         # tune rule weights against the harness
python -m ml.training.train_neural       # train the shipped checkpoint
python -m ml.training.train_neural --ablation   # the representation experiment
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

## Why it is built this way

v1's BiLSTM lost to v1's own rule engine. Four concrete reasons, each verified
against `git show main:backend/`, and each closed structurally here:

1. **Padding polluted the loss.** v1 padded every piece to 32 quarter notes and
   labelled the padding as tonic, with no mask in the loss or in `evaluate()`.
   Here sequences carry an explicit `valid` mask and pieces are kept whole.
2. **The representation deleted the signal.** Inputs were absolute pitch classes;
   targets were tonic-relative scale degrees. The network had to induce the
   tonic across all twelve transpositions from ~400 chorales given a single
   `is_minor` bit — while the rule engine was handed `key.tonic` for free.
   Everything here is tonic-relative by construction. `--ablation` measures
   exactly what that was worth.
3. **The label space could not represent the data.** Seven diatonic triads, with
   every seventh chord, secondary dominant and borrowed chord projected onto the
   nearest one by a `+1/-1` pitch-class vote. Here the model predicts the notes
   themselves and chord labels are derived afterwards, so there is no label
   space to be too small.
4. **There was no transition model.** Independent per-beat softmax and argmax,
   against a rule engine that had an explicit functional grammar. Here decoding
   is iterative and bidirectional: the model revises its own output.

The ordering of the work was deliberate — a genuinely strong rules baseline
first, then the evaluation harness, and only then the model. A weak baseline or
an untrustworthy metric would have made any conclusion about the model
worthless, which is precisely what happened in v1.
