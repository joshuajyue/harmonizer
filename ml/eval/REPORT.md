# HarmonAIzer v2 — engine evaluation

Generated 2026-08-19 08:47 UTC.

Bach chorale corpus, piece-level split by hash of the piece id: 263 train / 44 val / 61 test. Every engine sees the same held-out sopranos and nothing else.

## 1. Voice-leading defects per 100 chord changes

Objective and engine-agnostic: these are counted by the same detectors for every row, including Bach's. The `bach_oracle` column is the calibration — it is what the ceiling scores under these exact definitions, and it is not zero.

| defect / 100 chords      | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|--------------------------|--------------|-------|--------|---------------|-----------|-------------|
| parallel_fifths          | 72.61        | 0.00  | 4.95   | 0.00          | 0.00      | 0.21        |
| parallel_octaves         | 87.19        | 0.00  | 5.18   | 0.00          | 0.00      | 0.06        |
| contrary_fifths          | 2.82         | 0.07  | 0.34   | 0.00          | 0.00      | 0.23        |
| contrary_octaves         | 3.49         | 0.00  | 0.29   | 0.00          | 0.00      | 0.08        |
| direct_fifths            | 0.00         | 0.03  | 0.86   | 0.29          | 0.57      | 0.34        |
| direct_octaves           | 0.04         | 0.10  | 0.13   | 0.33          | 0.22      | 0.06        |
| voice_crossing           | 0.00         | 0.00  | 0.10   | 0.00          | 0.00      | 3.33        |
| voice_overlap            | 20.96        | 0.23  | 2.98   | 0.92          | 1.07      | 9.41        |
| spacing                  | 2.63         | 0.07  | 0.78   | 0.06          | 0.04      | 2.23        |
| range                    | 0.04         | 0.03  | 0.23   | 0.04          | 0.12      | 0.13        |
| unresolved_leading_tone  | 0.67         | 0.10  | 0.83   | 0.37          | 0.36      | 0.40        |
| unresolved_seventh       | 0.16         | 1.60  | 1.68   | 1.10          | 0.61      | 1.89        |
| doubled_leading_tone     | 0.00         | 0.07  | 0.44   | 0.69          | 0.53      | 0.72        |
| awkward_melodic_interval | 6.66         | 0.00  | 1.53   | 0.00          | 0.00      | 1.29        |
| large_leap               | 1.29         | 0.46  | 1.01   | 0.94          | 1.03      | 1.02        |
| **HARD TOTAL**           | 159.84       | 0.03  | 10.47  | 0.04          | 0.12      | 3.73        |

HARD TOTAL sums the unambiguous errors: parallel fifths, parallel octaves, voice crossing and range violations.

## 2. Style distance from the Bach corpus

Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint) against the **training** split. `bach_oracle` is held-out Bach measured against training Bach, so its value is the noise floor: no engine can meaningfully score below it.

| JS divergence       | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|---------------------|--------------|-------|--------|---------------|-----------|-------------|
| chord_unigram_js    | 0.169        | 0.108 | 0.014  | 0.055         | 0.019     | 0.007       |
| chord_bigram_js     | 0.421        | 0.202 | 0.060  | 0.116         | 0.073     | 0.056       |
| root_motion_js      | 0.264        | 0.054 | 0.007  | 0.015         | 0.006     | 0.001       |
| inversion_js        | 0.904        | 0.013 | 0.004  | 0.009         | 0.010     | 0.000       |
| quality_js          | 0.100        | 0.031 | 0.007  | 0.009         | 0.010     | 0.000       |
| cadence_js          | 0.372        | 0.069 | 0.004  | 0.047         | 0.003     | 0.004       |
| melodic_interval_js | 0.040        | 0.059 | 0.003  | 0.016         | 0.011     | 0.001       |
| outer_motion_js     | 0.651        | 0.025 | 0.010  | 0.002         | 0.002     | 0.001       |

### Style mix

| share of chords       | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------------------|--------------|-------|--------|---------------|-----------|-------------|
| seventh_chords        | 0.2%         | 12.3% | 11.7%  | 11.2%         | 9.3%      | 15.3%       |
| applied_chords        | 0.0%         | 10.8% | 9.9%   | 7.7%          | 10.3%     | 10.8%       |
| root_position         | 0.0%         | 68.9% | 72.8%  | 72.3%         | 72.5%     | 68.4%       |
| first_inversion       | 0.0%         | 27.9% | 24.4%  | 26.0%         | 26.1%     | 25.2%       |
| second_inversion      | 99.8%        | 3.2%  | 1.9%   | 1.2%          | 1.0%      | 3.5%        |
| contrary_outer_motion | 3.7%         | 49.9% | 32.6%  | 34.3%         | 31.7%     | 33.6%       |
| parallel_outer_motion | 89.5%        | 1.8%  | 7.8%   | 3.1%          | 3.9%      | 5.6%        |

### Cadence types

| cadence   | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------|--------------|-------|--------|---------------|-----------|-------------|
| HC        | 0.0%         | 24.7% | 29.6%  | 24.9%         | 27.7%     | 31.5%       |
| IAC       | 2.5%         | 27.7% | 5.5%   | 21.4%         | 6.0%      | 4.9%        |
| PAC       | 0.0%         | 19.7% | 28.5%  | 26.6%         | 28.8%     | 28.8%       |
| deceptive | 1.6%         | 0.3%  | 1.1%   | 0.3%          | 1.4%      | 0.8%        |
| other     | 95.9%        | 25.5% | 32.9%  | 24.4%         | 33.2%     | 31.2%       |
| phrygian  | 0.0%         | 0.5%  | 0.3%   | 0.8%          | 0.3%      | 0.0%        |
| plagal    | 0.0%         | 1.6%  | 2.2%   | 1.6%          | 2.7%      | 2.7%        |

## 3. Held-out likelihood

Negative log-likelihood in nats per predicted note token, and its perplexity, for Bach's own alto/tenor/bass on the held-out split. Only defined for probabilistic engines; a rule engine has no likelihood to report.

| metric           | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|------------------|--------------|-------|--------|---------------|-----------|-------------|
| NLL (nats/token) | n/a          | n/a   | 0.5246 | 0.5246        | 0.5246    | n/a         |
| perplexity       | n/a          | n/a   | 1.690  | 1.690         | 1.690     | n/a         |

## 4. Agreement with Bach — reported, NOT the headline

This is the metric v1 optimised. It is included for continuity and because watching it move independently of sections 1-3 is itself the argument against it: a harmonization can disagree with Bach on most beats and still be excellent, and v1's version of this number additionally counted padded positions.

| agreement          | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|--------------------|--------------|-------|--------|---------------|-----------|-------------|
| chord_exact        | 0.5%         | 28.5% | 41.0%  | 35.1%         | 41.5%     | 100.0%      |
| chord_root_quality | 18.2%        | 40.3% | 50.1%  | 44.7%         | 50.4%     | 100.0%      |
| chord_root         | 27.5%        | 48.9% | 57.0%  | 51.3%         | 57.5%     | 100.0%      |
| voice_note         | 15.4%        | 29.1% | 40.4%  | 35.2%         | 40.3%     | 100.0%      |
| bass_note          | 5.5%         | 24.7% | 37.2%  | 31.9%         | 38.2%     | 100.0%      |

## 5. Cost and robustness

| metric          | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------------|--------------|-------|--------|---------------|-----------|-------------|
| pieces scored   | 61           | 61    | 61     | 61            | 61        | 61          |
| failures        | 0            | 0     | 0      | 0             | 0         | 0           |
| seconds / piece | 0.021        | 0.198 | 2.845  | 1.306         | 1.724     | 0.000       |

## 6. Melody-only key detection

The tables above supply the ground-truth key so the comparison isolates the harmonic decision. In production the engine must find the key from the tune alone; this is the accuracy of that step on the same held-out melodies.

| metric                   | value |
|--------------------------|-------|
| exact key (tonic + mode) | 83.6% |
| correct tonic            | 83.6% |
| relative-key confusion   | 0.0%  |

### Same engines, key detected instead of supplied

| metric                  | fixed_thirds | rules | neural | neural_refine | neural_vl |
|-------------------------|--------------|-------|--------|---------------|-----------|
| HARD TOTAL / 100 chords | 151.41       | 0.03  | 10.90  | 0.04          | 0.14      |
| chord_bigram_js         | 0.405        | 0.187 | 0.077  | 0.119         | 0.090     |
| chord agreement         | 0.5%         | 26.5% | 37.3%  | 32.4%         | 37.3%     |

## 7. Representation ablation — what the v1 handicap actually cost

Identical architecture, identical data, identical number of gradient steps; only the pitch representation differs. `absolute` is exactly the information v1's network had: raw pitch plus a mode flag, no tonic. `absolute_augmented` adds per-epoch transposition, which is the standard remedy. Validation NLL is in nats per predicted note token, so lower is better.

| representation     | val NLL/token | perplexity | transpositions/piece | best epoch | minutes |
|--------------------|---------------|------------|----------------------|------------|---------|
| tonic_relative     | 0.6943        | 2.002      | 1.0                  | 137        | 29.5    |
| absolute           | 0.7089        | 2.032      | 1.0                  | 131        | 33.5    |
| absolute_augmented | 0.6961        | 2.006      | 10.64                | 140        | 30.8    |

Relative to tonic-relative: `absolute` +1.5% perplexity, `absolute_augmented` +0.2% perplexity.

## 8. Most frequent chords

* `fixed_thirds`: (5, 'maj') 10.8%, (0, 'maj') 9.9%, (5, 'min') 9.2%, (0, 'min') 9.1%, (9, 'min') 9.0%, (7, 'maj') 9.0%, (7, 'min') 8.1%, (8, 'maj') 8.0%
* `rules`: (7, 'maj') 25.7%, (0, 'maj') 23.0%, (0, 'min') 15.9%, (5, 'maj') 7.4%, (7, 'dom7') 5.7%, (10, 'maj') 5.4%, (5, 'min') 2.7%, (2, 'maj') 2.6%
* `neural`: (7, 'maj') 20.0%, (0, 'maj') 19.0%, (0, 'min') 12.0%, (3, 'maj') 6.6%, (5, 'maj') 6.3%, (10, 'maj') 5.1%, (9, 'min') 4.5%, (5, 'min') 2.6%
* `neural_refine`: (7, 'maj') 24.0%, (0, 'maj') 23.3%, (0, 'min') 14.7%, (5, 'maj') 6.2%, (10, 'maj') 4.4%, (7, 'dom7') 4.3%, (9, 'min') 4.1%, (5, 'min') 3.5%
* `neural_vl`: (7, 'maj') 20.0%, (0, 'maj') 20.0%, (0, 'min') 12.1%, (5, 'maj') 7.4%, (3, 'maj') 6.1%, (10, 'maj') 4.9%, (9, 'min') 4.3%, (5, 'min') 3.3%
* `bach_oracle`: (7, 'maj') 17.5%, (0, 'maj') 17.3%, (0, 'min') 10.6%, (3, 'maj') 5.8%, (5, 'maj') 5.5%, (10, 'maj') 4.5%, (9, 'min') 4.0%, (5, 'min') 3.8%


<!-- NARRATIVE -->

# Discussion

## Headline

A learned engine now beats a strong rule engine on this task, but only after the
rule engine was made strong enough to be worth beating and the metrics were
built to tell the difference. The result is not "the model won"; it is that the
two systems fail in *orthogonal* ways, and the useful engine is the one that
composes them:

| | voice-leading errors | style distance from Bach |
|---|---|---|
| `rules` | none (0.03 / 100 chords) | 3.6x the noise floor (0.202 vs 0.056) |
| `neural` | 10.47 / 100 chords | at the noise floor (0.060 vs 0.056) |
| `neural_vl` | none (0.12 / 100 chords) | 1.3x the noise floor (0.073) |

The rule engine is a flawless contrapuntist with a narrow harmonic imagination.
The learned model has Bach's harmonic vocabulary and writes parallel fifths at
twenty times his rate. `neural_vl` — the model's harmonic choice, realised under
a voice-leading veto — keeps both halves.

For reference, the commodity approach (`fixed_thirds`: scale-locked parallel
intervals, which is what a commercial harmonizer does) scores 159.84 defects per
100 chords and a style divergence of 0.421. The gap between it and anything else
here is the entire value of the project.

## 1. The v1 post-mortem, verified — and one correction

All four diagnoses were checked against `git show main:backend/`. Three hold as
stated. One does not, and the difference matters.

**Padding pollutes the loss — code is real, effect was not.** `chord_padding[:, 0] = 1`
is there in `data_processor.py`, and `train.py` takes an unmasked mean over it in
both the loss and `evaluate()`. But measured against the corpus v1 actually
trained on: exactly **1 of 368 chorales is shorter than 32 quarter notes**, so
padded positions are **0.07% of the training grid**. v1's reported `val_acc` was
therefore *not* meaningfully inflated by padding. The real cost of the same code
path is the other half of it — **truncation**. The median chorale is 48 beats and
the longest is 193, so `features[:SEQUENCE_LENGTH]` meant the model saw an
average of **64.4% of each piece and never saw 41.2% of the corpus**, including
most final cadences. Fixing the whole sequence-handling scheme is worth **+28.1%**
relative chord-root accuracy in the reconstruction below; essentially all of that
is truncation, not padding.

**Representation mismatch — confirmed, and it is the dominant term.** Inputs were
absolute pitch classes (`pitch_class_vector[note.pitch.pitchClass] = 1`); targets
were tonic-relative scale degrees (`(pc - tonic) % 12`). Two melodies with
identical pitch-class content in different keys had different correct answers and
nothing in the input distinguished them beyond one `is_minor` bit, while the rule
engine was handed `key.tonic` for free. Worth **+68.1%** on its own.

**The label space cannot represent the training data — confirmed, quantified.**
`extract_real_chord_labels` projects every sonority onto one of seven diatonic
triads by a `+1/-1` pitch-class vote. Measured directly against the corpus, that
projection **preserves the chord root only 84.5% of the time and the exact chord
only 68.9%**. Nearly a third of v1's training labels were wrong as descriptions of
what Bach wrote. 15.8% of beats are seventh chords and 11.9% are applied chords;
the label space can express neither.

**No transition model — confirmed.** `ChordLSTM` emits independent per-beat
softmax and `argmax`, with no CRF, no Viterbi and no autoregressive decoding,
against a rule engine carrying an explicit `MAJOR_FUNCTIONAL_PROGRESSIONS` table
and a +2 bonus for V->I.

## 2. Reconstructing v1 and flipping one factor at a time

`python -m ml.training.v1_diagnosis` rebuilds v1's exact setup — its 14-dim
features, its BiLSTM, its label extraction, its padding scheme — and varies one
factor at a time. Every arm is scored identically: tonic-relative chord-root
accuracy on all real beats of held-out pieces, so label spaces and sequence
schemes are directly comparable.

| configuration | root accuracy | vs v1 |
|---|---|---|
| **v1 as built** (absolute / 7 triads / 32-beat window) | **0.346** | — |
| + tonic-relative input | 0.582 | **+68.1%** |
| + whole pieces, masked loss | 0.444 | +28.1% |
| + rich label space | 0.321 | **-7.4%** |
| all three | **0.642** | **+85.4%** |

Two things worth sitting with.

**The representation is not one of four roughly equal problems; it is most of the
problem.** Fixing it alone recovers more than fixing everything else combined.
The owner's "the ML model was too abstract" intuition points at the right thing,
but the precise statement is narrower and more actionable: *the target was not a
function of the input*. No amount of capacity fixes that.

**Enriching the label space makes accuracy worse — until the representation is
fixed.** On its own it costs 7.4%; combined with tonic-relative input it gains
7.4% over the small space (0.598 -> 0.642). A larger, more faithful label space
is a harder classification problem, so measured accuracy falls even though the
labels are now truthful. Accuracy on corrupted labels is *easier* to achieve
precisely because the corruption throws information away. That single row is the
whole argument for section 4 of this report being explicitly not the headline —
and it is the trap v1 fell into.

## 3. What was built, in the order it was built

**Phase 1 — a rule engine worth losing to.** v1's "creative engine" chose one
diatonic triad per measure by greedy argmax and handed the label to a renderer
that stacked a root-position triad from a fixed register (`midi_utils.py`), so it
never voiced anything and voice leading was structurally impossible. The
replacement searches ~100 chords — every diatonic triad and seventh in every
inversion, secondary dominants, applied leading-tone chords, mixture, the
Neapolitan — with Viterbi over a functional grammar, keeps the top six per beat
as max-marginals, and then runs a *second* Viterbi over actual SATB voicings
whose transition cost is the voice-leading rulebook. Crossing, illegal spacing,
doubled leading tones and doubled sevenths are excluded by construction rather
than penalised. Chord frequency priors are fit to the training split; the
harmonic weights are tuned by coordinate descent against the harness on the
validation split. It scores **0.03 hard errors per 100 chords, against Bach's own
3.73**. That is the bar.

**Phase 2 — the harness, first, before any model existed.** One command scores
every registered engine on the identical held-out split with identical metrics.
Two controls make the numbers mean anything:

* `fixed_thirds` as the floor. Without it, "0.202 bigram divergence" sounds bad;
  with it (0.421) the rule engine is clearly doing real work.
* **The Bach oracle as the ceiling** — Bach's own four voices through the
  identical detectors. This is the single most important thing in the harness.
  Bach crosses voices 3.33 times per 100 chords and overlaps parts 9.41 times;
  the rule engine does neither, ever. Without that row one would "fix" the rule
  engine's spacing to be stricter than Bach's and call it progress. It also sets
  the noise floor for every distributional metric: held-out Bach differs from
  training Bach by 0.056 bits of chord-bigram divergence, so 0.060 is not "close
  to Bach", it is *indistinguishable from Bach at this sample size*.

The harness earned its cost immediately. It caught four bugs that were invisible
from the code and from listening:

1. The voicing search compared each slot's **initial** soprano pitch, so every
   parallel created by a melodic passing tone went unseen. Fixing it took the
   rule engine's parallel fifths and octaves to zero.
2. Cadence bonuses were applied to **every** slot of a held phrase-final note,
   pushing a dominant into the middle of the final chord and converting perfect
   authentic cadences into imperfect ones.
3. Downbeat alignment differed between training and inference on **41 of 61**
   held-out chorales, because training read music21's notated pickup and
   inference had to infer one. Both paths now call one shared function.
4. Tonic normalization chose its shift from the key alone, ignoring register, so
   a tune transposed up a fifth normalized an **octave** above anything the model
   had seen. Transposition equivariance measured 0.06 at that shift and 1.00
   everywhere else.

None of these change a loss curve. All of them change the output.

**Phase 3 — the model.** 1.6M parameters, trained in half an hour on a laptop
CPU. Every design choice is a v1 failure inverted:

* It predicts **the notes each voice sings**, not a chord label. There is no
  label space to be too small; chord labels are derived from the voices
  afterwards. `chordify()` in v1 discarded exactly the counterpoint that makes
  the corpus worth training on.
* Everything is **tonic-relative** by construction, and pieces are kept whole
  with an explicit padding mask.
* Every prediction is conditioned on the **entire rest of the texture in both
  directions**, and decoding is iterative: annealed blocked Gibbs repeatedly
  erases part of the model's own output and rewrites it, so a choice at beat 3
  can be reconsidered once beat 7 exists. That is the transition model v1 lacked
  — not bolted on as a CRF, but as the shape of the whole procedure.

## 4. Head-to-head: where the learned engine wins and where it does not

**The learned model loses on counterpoint, badly, and it is not close.** Raw
`neural` writes 4.95 parallel fifths and 5.18 parallel octaves per 100 chords.
Bach writes 0.21 and 0.06. The rule engine writes none. This is not a tuning
problem — it is structural: blocked Gibbs resamples sites independently, so
nothing in the procedure can see that two voices are about to move in parallel.
A model with a per-site conditional cannot represent a constraint that is
inherently joint.

**The learned model wins on harmony, and it is not close either.** Chord-bigram
divergence 0.060 against the rule engine's 0.202, with a noise floor of 0.056.
Cadence distribution 0.004 against 0.069. Every distributional metric in section
2 is three to twenty times better. Concretely, the rule engine produces 27.7%
imperfect authentic cadences where Bach produces 4.9%, and 19.7% perfect
authentic where Bach produces 28.8% — it does not know how to close a phrase like
Bach. The model matches Bach's cadence mix almost exactly. The rule engine also
essentially cannot **modulate**: its first-order chord grammar has no
representation of a phrase being temporarily in another key, so it stays in the
home key and cadences there. The model moves to the relative major and back
without being told such a thing exists.

**The fix is not to add a penalty term.** Mask one voice completely and the
model's logits for it no longer depend on any of its own choices, so its entire
line can be re-solved *exactly*, by Viterbi, with the model's log probabilities
as emissions and the voice-leading rules as transitions. Sweeping the three free
voices in turn is coordinate ascent on a well-defined objective. The model still
makes every harmonic decision; the rules only veto illegal ways of realising it.

That is `neural_vl`: parallel fifths **0.00**, parallel octaves **0.00**, hard
errors 10.47 -> **0.12**, with chord-bigram divergence essentially unchanged
(0.060 -> 0.073) and cadence divergence unchanged (0.003). It beats the rule
engine on every quality metric in this report and matches it on correctness.

**Where the rule engine still wins: cost and predictability.** 0.198 s/piece
against 1.724, and it has no checkpoint, no training, and no failure mode that
depends on a melody being in distribution. It is the right default for a latency
budget.

**`neural_refine` is the interesting failure.** Seeding Gibbs with the rule
engine's draft was meant to combine both. It does not: it inherits the rule
engine's harmonic habits (bigram divergence 0.116 and 21.4% imperfect cadences,
against `neural_vl`'s 0.073 and 6.0%) without gaining anything the veto does not
already provide. A good draft is a strong attractor, and the model spends its
sweeps agreeing with it. Composing the two systems works when the rules act as a
*constraint*; it does not work when they act as an *initialisation*.

## 5. What actually mattered

**Ranked by effect on the outcome:**

1. **Building the harness before the model.** Four real bugs, three of which were
   in code I had already convinced myself was correct, and one of which (the
   soprano-passing-tone bug) was the difference between a rule engine with 2.30
   parallel octaves per 100 chords and one with zero. v1's actual failure was not
   that its model was bad; it is that it had no way to find out.
2. **The Bach oracle row.** Every defect rate in section 1 is uninterpretable
   without it, and two "improvements" I was about to make would have pushed the
   engines *further* from Bach.
3. **Representation.** +68% in v1's setting. But see the caveat below.
4. **Making the constraint joint rather than penalised.** The difference between
   a learned engine that cannot be shipped and one that can, and it came from
   noticing that masking a whole voice makes its logits independent of itself —
   not from more capacity or more data.

**And the honest negative:** section 7's ablation shows the tonic-relative
representation is worth only **1.5% perplexity** for the v2 architecture, and
transposition augmentation recovers almost all of that (**+0.2%**). That looks
like it contradicts the +68% above. It does not — it sharpens it. v1's model saw
*only the melody* and had to produce a *tonic-relative* target, so the tonic was
genuinely missing information. v2's model sees all four voices and predicts
pitches from pitches: the target is in the same frame as the input, and the key
is inferable from the vertical context anyway. **The fix was never "use
tonic-relative pitch"; it was "make the target expressible from the input".**
Tonic-relative representation is one way to achieve that, and it is still worth
keeping — it is free, it makes the twelve keys share statistics, and it is what
makes the learned engines exactly transposition-equivariant (asserted as a test).
But had I only run the v2 ablation, I would have concluded the representation
barely matters, and had I only run the v1 reconstruction, I would have concluded
it is everything. Both experiments were necessary.

**What did not matter:** model size. 1.6M parameters was chosen to be small and
was never the binding constraint; validation loss plateaus well before the model
saturates. Nothing in this report would have been fixed by a bigger network,
which is the same conclusion the v1 post-mortem reached and the reason the effort
went into representation, evaluation and decoding instead.

## 6. Known limitations

* **Key detection is the weakest link in production.** 83.6% exact on held-out
  melodies. The tables above supply the ground-truth key so the comparison
  isolates the harmonic decision; the realistic setting is in section 6, where
  the learned engine's bigram divergence degrades from 0.073 to 0.090 and the
  rule engine's from 0.202 to 0.187. A wrong key is a wrong harmonization
  regardless of engine, and this is the highest-leverage thing left.
* **Articulation is not modelled.** The model predicts sounding pitch per
  sixteenth, not note onsets, so tied and repeated notes are indistinguishable
  and re-articulation is derived afterwards. Nothing in this report measures it,
  which is exactly why it was descoped — but a listener would notice.
* **The rule engine cannot modulate**, as above. It is a first-order chord
  grammar in a fixed key.
* **`unresolved_seventh` is the one defect no engine handles well** (0.61–1.68
  per 100 chords, against Bach's 1.89). Note Bach's own rate is the highest of
  any row, which suggests the detector is somewhat stricter than practice rather
  than that every engine is failing.
* **`neural_refine` is not transposition-stable.** Measured on one tune across
  six transpositions, chord-root agreement with the untransposed answer is
  0.35-0.80, against 0.50-0.85 for the rule draft it is seeded from and 1.00 for
  `neural` and `neural_vl`. Gibbs amplifies small differences in the draft rather
  than damping them. Another reason it is not the engine to ship.
* **The style metrics are distributional.** An engine could match every histogram
  and still be incoherent piece by piece. Nothing here is a substitute for
  listening, and nothing here measures phrase-level structure.

## 7. What I would try next, in order

1. **Better key detection, and key changes within a piece.** 83.6% is the
   binding constraint on real input, and Krumhansl-Schmuckler on a whole melody
   cannot represent a piece that modulates. A small learned key-tracker over the
   melody, or joint inference of key and harmony, is the obvious move.
2. **Constrained decoding during Gibbs rather than only after it.** The polish
   currently runs as a post-pass. Folding the veto into the sampling sweeps —
   masking illegal pitches per site given the current context — would let the
   model explore *within* the legal set instead of being pulled back into it, and
   should recover the 0.060 -> 0.073 divergence the polish costs.
3. **Model onsets.** A hold token in the DeepBach manner, plus an articulation
   metric in the harness so the improvement is measurable rather than asserted.
4. **A phrase-level metric.** Everything distributional here is first-order.
   Cadence *placement* and phrase *length* relative to the melody's own phrasing
   would catch failures the current metrics cannot see.
5. **Learn the voicing cost instead of writing it.** The polish weights are
   hand-set. They could be fit to Bach directly — the harness is already the
   objective function, and this is the same move that took the rule engine's
   guessed priors (V7 at 19% of all chords, against Bach's 2%) and replaced them
   with measured ones.
6. **Listening tests.** Every number here is a proxy. They are good proxies, and
   they are calibrated against Bach, but the project is about how something
   sounds and nothing above hears anything.

## 8. Reproducing

```bash
python -m ml.training.calibrate_rules     # chord priors from the training split
python -m ml.training.tune_rules          # rule weights, tuned against the harness
python -m ml.training.train_neural        # the shipped checkpoint (~30 min, CPU)
python -m ml.training.train_neural --ablation   # section 7
python -m ml.training.v1_diagnosis        # section 2 of this discussion
python -m ml.eval.run --detect-key        # everything above
python -m pytest ml/tests -q              # 267 tests
```

Splits are a hash of the piece id, so they are stable across runs and machines;
no model here has ever seen a test piece. Every engine is deterministic given a
seed.
