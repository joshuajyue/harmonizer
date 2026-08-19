# HarmonAIzer v2 — engine evaluation

Generated 2026-08-19 10:45 UTC.

Bach chorale corpus, piece-level split by hash of the piece id: 263 train / 44 val / 61 test. Every engine sees the same held-out sopranos and nothing else.

## 1. Every engine against Bach, on the axes that matter

**The headline is distance from Bach, not a defect leaderboard.** Voice-leading defects are a guardrail that stops an engine degenerating into parallel thirds; they are not the objective. Bach himself breaks these rules, and an engine that never does is not better than Bach — it is stiffer than Bach. Every column below is therefore read against the `bach_oracle` row, and *under*-shooting is as much a miss as overshooting.

| metric                     | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|----------------------------|--------------|-------|--------|---------------|-----------|-------------|
| STRUCTURAL defects / piece | 0.443        | 0.115 | 0.164  | 0.246         | 0.131     | 0.066       |
| HARD defects /100 chords   | 159.84       | 0.03  | 10.37  | 0.04          | 0.12      | 0.40        |
| SOFT defects /100 chords   | 38.7         | 2.7   | 11.0   | 4.7           | 5.0       | 21.0        |
| chord-bigram JS from Bach  | 0.421        | 0.202 | 0.060  | 0.116         | 0.071     | 0.056       |
| cadence JS from Bach       | 0.372        | 0.069 | 0.004  | 0.047         | 0.003     | 0.004       |
| distinct chords / piece    | 6.7          | 9.4   | 12.6   | 10.6          | 11.7      | 14.5        |
| share of beats on I or V   | 36.2         | 68.6  | 52.9   | 64.5          | 53.2      | 48.6        |
| chord changes /100 beats   | 69.5         | 82.4  | 75.9   | 79.1          | 76.2      | 78.2        |
| voice moves /100 beats     | 80.8         | 97.1  | 122.1  | 152.1         | 157.7     | 149.3       |

The three defect rows are tiered by **audibility**, not by textbook tradition, and are never summed. STRUCTURAL is the category a listener notices instantly — a piece that never resolves, a phrase closing somewhere impossible — and it must stay near Bach's own ~0.02 whatever else an engine is doing. HARD is the classic audible errors. SOFT is real to a theorist and largely invisible to a listener; Bach breaks all of them, so drifting upward there in exchange for more interesting harmony is a fair trade.

The bottom four rows are what a defect count cannot see at all. An engine reaches zero defects either by realising a full harmonic vocabulary carefully, or by narrowing the vocabulary until nothing can go wrong. Those look identical in the defect rows and completely different below them.

## 2. Defects by tier

Objective and engine-agnostic: counted by the same detectors for every row, including Bach's. Read as a guardrail — the question is whether an engine is *materially worse* than the oracle, not whether it is lowest.

Structural defects are per PIECE; everything else is per 100 chord changes. The two units are never added together. `half_cadence_ending` is reported and not scored: Bach ends 9.2% of his chorales on a root-position V, so it is idiomatic — but an engine doing it half the time is broken, and this is the only way to see that.

| defect                             | unit        | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|------------------------------------|-------------|--------------|-------|--------|---------------|-----------|-------------|
| no_tonal_closure                   | % of pieces | 37.7%        | 0.0%  | 1.6%   | 3.3%          | 1.6%      | 1.6%        |
| implausible_phrase_cadence         | % of pieces | 3.3%         | 9.8%  | 14.8%  | 19.7%         | 11.5%     | 4.9%        |
| key_not_established                | % of pieces | 3.3%         | 1.6%  | 0.0%   | 1.6%          | 0.0%      | 0.0%        |
| **STRUCTURAL / piece**             | per piece   | 0.443        | 0.115 | 0.164  | 0.246         | 0.131     | 0.066       |
| half_cadence_ending (not a defect) | % of pieces | 0.0%         | 0.0%  | 9.8%   | 1.6%          | 6.6%      | 11.5%       |
| parallel_fifths                    | /100 chords | 72.61        | 0.00  | 4.95   | 0.00          | 0.00      | 0.21        |
| parallel_octaves                   | /100 chords | 87.19        | 0.00  | 5.18   | 0.00          | 0.00      | 0.06        |
| range                              | /100 chords | 0.04         | 0.03  | 0.23   | 0.04          | 0.12      | 0.13        |
| **HARD TOTAL**                     | /100 chords | 159.84       | 0.03  | 10.37  | 0.04          | 0.12      | 0.40        |
| voice_crossing                     | /100 chords | 0.00         | 0.00  | 0.10   | 0.00          | 0.00      | 3.33        |
| voice_overlap                      | /100 chords | 20.96        | 0.23  | 2.98   | 0.92          | 1.32      | 9.41        |
| spacing                            | /100 chords | 2.63         | 0.07  | 0.78   | 0.06          | 0.08      | 2.23        |
| direct_fifths                      | /100 chords | 0.00         | 0.03  | 0.86   | 0.29          | 0.54      | 0.34        |
| direct_octaves                     | /100 chords | 0.04         | 0.10  | 0.13   | 0.33          | 0.22      | 0.06        |
| contrary_fifths                    | /100 chords | 2.82         | 0.07  | 0.34   | 0.00          | 0.02      | 0.23        |
| contrary_octaves                   | /100 chords | 3.49         | 0.00  | 0.29   | 0.00          | 0.06      | 0.08        |
| unresolved_leading_tone            | /100 chords | 0.67         | 0.10  | 0.83   | 0.37          | 0.38      | 0.40        |
| unresolved_seventh                 | /100 chords | 0.16         | 1.60  | 1.68   | 1.10          | 0.46      | 1.89        |
| doubled_leading_tone               | /100 chords | 0.00         | 0.07  | 0.44   | 0.69          | 0.48      | 0.72        |
| awkward_melodic_interval           | /100 chords | 6.66         | 0.00  | 1.53   | 0.00          | 0.28      | 1.29        |
| large_leap                         | /100 chords | 1.29         | 0.46  | 1.01   | 0.94          | 1.14      | 1.02        |
| **SOFT TOTAL**                     | /100 chords | 38.71        | 2.71  | 10.96  | 4.70          | 5.00      | 21.00       |

## 3. Style distance from the Bach corpus

Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint) against the **training** split. `bach_oracle` is held-out Bach measured against training Bach, so its value is the noise floor: no engine can meaningfully score below it.

| JS divergence       | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|---------------------|--------------|-------|--------|---------------|-----------|-------------|
| chord_unigram_js    | 0.169        | 0.108 | 0.014  | 0.055         | 0.021     | 0.007       |
| chord_bigram_js     | 0.421        | 0.202 | 0.060  | 0.116         | 0.071     | 0.056       |
| root_motion_js      | 0.264        | 0.054 | 0.007  | 0.015         | 0.006     | 0.001       |
| inversion_js        | 0.904        | 0.013 | 0.004  | 0.009         | 0.011     | 0.000       |
| quality_js          | 0.100        | 0.031 | 0.007  | 0.009         | 0.011     | 0.000       |
| cadence_js          | 0.372        | 0.069 | 0.004  | 0.047         | 0.003     | 0.004       |
| melodic_interval_js | 0.040        | 0.059 | 0.003  | 0.016         | 0.011     | 0.001       |
| outer_motion_js     | 0.651        | 0.025 | 0.010  | 0.002         | 0.002     | 0.001       |

### Style mix

| share of chords       | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------------------|--------------|-------|--------|---------------|-----------|-------------|
| seventh_chords        | 0.2%         | 12.3% | 11.7%  | 11.2%         | 9.1%      | 15.3%       |
| applied_chords        | 0.0%         | 10.8% | 9.9%   | 7.7%          | 10.1%     | 10.8%       |
| root_position         | 0.0%         | 68.9% | 72.8%  | 72.3%         | 72.3%     | 68.4%       |
| first_inversion       | 0.0%         | 27.9% | 24.4%  | 26.0%         | 26.4%     | 25.2%       |
| second_inversion      | 99.8%        | 3.2%  | 1.9%   | 1.2%          | 0.9%      | 3.5%        |
| contrary_outer_motion | 3.7%         | 49.9% | 32.6%  | 34.3%         | 31.5%     | 33.6%       |
| parallel_outer_motion | 89.5%        | 1.8%  | 7.8%   | 3.1%          | 3.8%      | 5.6%        |

### Cadence types

| cadence   | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------|--------------|-------|--------|---------------|-----------|-------------|
| HC        | 0.0%         | 24.7% | 29.6%  | 24.9%         | 27.9%     | 31.5%       |
| IAC       | 2.5%         | 27.7% | 5.5%   | 21.4%         | 6.0%      | 4.9%        |
| PAC       | 0.0%         | 19.7% | 28.5%  | 26.6%         | 29.0%     | 28.8%       |
| deceptive | 1.6%         | 0.3%  | 1.1%   | 0.3%          | 1.1%      | 0.8%        |
| other     | 95.9%        | 25.5% | 32.9%  | 24.4%         | 33.4%     | 31.2%       |
| phrygian  | 0.0%         | 0.5%  | 0.3%   | 0.8%          | 0.3%      | 0.0%        |
| plagal    | 0.0%         | 1.6%  | 2.2%   | 1.6%          | 2.2%      | 2.7%        |

## 4. Held-out likelihood

Negative log-likelihood in nats per predicted note token, and its perplexity, for Bach's own alto/tenor/bass on the held-out split. Only defined for probabilistic engines; a rule engine has no likelihood to report.

| metric           | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|------------------|--------------|-------|--------|---------------|-----------|-------------|
| NLL (nats/token) | n/a          | n/a   | 0.5246 | 0.5246        | 0.5246    | n/a         |
| perplexity       | n/a          | n/a   | 1.690  | 1.690         | 1.690     | n/a         |

## 5. Agreement with Bach — reported, NOT the headline

This is the metric v1 optimised. It is included for continuity and because watching it move independently of sections 1-3 is itself the argument against it: a harmonization can disagree with Bach on most beats and still be excellent, and v1's version of this number additionally counted padded positions.

| agreement          | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|--------------------|--------------|-------|--------|---------------|-----------|-------------|
| chord_exact        | 0.5%         | 28.5% | 41.0%  | 35.1%         | 41.6%     | 100.0%      |
| chord_root_quality | 18.2%        | 40.3% | 50.1%  | 44.7%         | 50.6%     | 100.0%      |
| chord_root         | 27.5%        | 48.9% | 57.0%  | 51.3%         | 57.6%     | 100.0%      |
| voice_note         | 15.4%        | 29.1% | 40.4%  | 35.2%         | 40.5%     | 100.0%      |
| bass_note          | 5.5%         | 24.7% | 37.2%  | 31.9%         | 38.5%     | 100.0%      |

## 6. Cost and robustness

| metric          | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------------|--------------|-------|--------|---------------|-----------|-------------|
| pieces scored   | 61           | 61    | 61     | 61            | 61        | 61          |
| failures        | 0            | 0     | 0      | 0             | 0         | 0           |
| seconds / piece | 0.018        | 0.189 | 1.788  | 0.872         | 0.881     | 0.000       |

## 7. Melody-only key detection

The tables above supply the ground-truth key so the comparison isolates the harmonic decision. In production the engine must find the key from the tune alone; this is the accuracy of that step on the same held-out melodies.

| metric                   | value |
|--------------------------|-------|
| exact key (tonic + mode) | 83.6% |
| correct tonic            | 83.6% |
| relative-key confusion   | 0.0%  |

### Same engines, key detected instead of supplied

| metric                  | fixed_thirds | rules | neural | neural_refine | neural_vl |
|-------------------------|--------------|-------|--------|---------------|-----------|
| HARD TOTAL / 100 chords | 151.41       | 0.03  | 10.72  | 0.04          | 0.22      |
| chord_bigram_js         | 0.405        | 0.187 | 0.077  | 0.119         | 0.088     |
| chord agreement         | 0.5%         | 26.5% | 37.3%  | 32.4%         | 37.3%     |

## 8. Representation ablation — what the v1 handicap actually cost

Identical architecture, identical data, identical number of gradient steps; only the pitch representation differs. `absolute` is exactly the information v1's network had: raw pitch plus a mode flag, no tonic. `absolute_augmented` adds per-epoch transposition, which is the standard remedy. Validation NLL is in nats per predicted note token, so lower is better.

| representation     | val NLL/token | perplexity | transpositions/piece | best epoch | minutes |
|--------------------|---------------|------------|----------------------|------------|---------|
| tonic_relative     | 0.6943        | 2.002      | 1.0                  | 137        | 29.5    |
| absolute           | 0.7089        | 2.032      | 1.0                  | 131        | 33.5    |
| absolute_augmented | 0.6961        | 2.006      | 10.64                | 140        | 30.8    |

Relative to tonic-relative: `absolute` +1.5% perplexity, `absolute_augmented` +0.2% perplexity.

## 9. Most frequent chords

* `fixed_thirds`: (5, 'maj') 10.8%, (0, 'maj') 9.9%, (5, 'min') 9.2%, (0, 'min') 9.1%, (9, 'min') 9.0%, (7, 'maj') 9.0%, (7, 'min') 8.1%, (8, 'maj') 8.0%
* `rules`: (7, 'maj') 25.7%, (0, 'maj') 23.0%, (0, 'min') 15.9%, (5, 'maj') 7.4%, (7, 'dom7') 5.7%, (10, 'maj') 5.4%, (5, 'min') 2.7%, (2, 'maj') 2.6%
* `neural`: (7, 'maj') 20.0%, (0, 'maj') 19.0%, (0, 'min') 12.0%, (3, 'maj') 6.6%, (5, 'maj') 6.3%, (10, 'maj') 5.1%, (9, 'min') 4.5%, (5, 'min') 2.6%
* `neural_refine`: (7, 'maj') 24.0%, (0, 'maj') 23.3%, (0, 'min') 14.7%, (5, 'maj') 6.2%, (10, 'maj') 4.4%, (7, 'dom7') 4.3%, (9, 'min') 4.1%, (5, 'min') 3.5%
* `neural_vl`: (7, 'maj') 20.2%, (0, 'maj') 20.0%, (0, 'min') 12.3%, (5, 'maj') 7.1%, (3, 'maj') 6.0%, (10, 'maj') 4.8%, (9, 'min') 4.3%, (5, 'min') 3.3%
* `bach_oracle`: (7, 'maj') 17.5%, (0, 'maj') 17.3%, (0, 'min') 10.6%, (3, 'maj') 5.8%, (5, 'maj') 5.5%, (10, 'maj') 4.5%, (9, 'min') 4.0%, (5, 'min') 3.8%


<!-- NARRATIVE -->

# Discussion

## Headline

**The objective is not zero defects.** Bach breaks his own voice-leading rules
3.73 times per 100 chord changes. An engine that never breaks them is not better
than Bach; it is stiffer than Bach. So defects are treated here as a *guardrail*
— they stop an engine degenerating into parallel thirds — and the objective is
stylistic fidelity and harmonic interest, measured against the oracle.

Read section 1 that way and the ranking inverts against the obvious one:

| | structural /piece | hard /100 | style vs Bach | vocabulary | verdict |
|---|---|---|---|---|---|
| `rules` | 0.115 | **0.03** | 0.202 (floor 0.056) | 9.4 (Bach 14.5) | undershoots — safe and narrow |
| `neural` | 0.164 | 10.37 | **0.060** | 12.6 | closest to Bach's harmony, but writes real parallels |
| `neural_vl` | 0.131 | 0.12 | 0.071 | 11.7 | clean *and* wide — **the one to ship** |
| `fixed_thirds` | 0.443 | 159.84 | 0.421 | 6.7 | the commodity floor |
| *Bach* | *0.066* | *0.40* | *0.056* | *14.5* | *the calibration* |

**The rule engine's 0.03 HARD is not a win, it is a diagnosis.** It reaches near-zero
by narrowing what it is willing to play: 9.4 distinct chords per piece against
Bach's 14.5, and 68.6% of its beats sitting on I or V against Bach's 48.6%. Its
harmony is *safe*, and the defect column cannot see that. Rows 4-7 of section 1
exist because of this.

**`neural_vl`'s 0.12 is a different thing entirely, and this distinction is the
main finding.** It is also near zero, but it is not narrow: 11.7 chords per
piece, 53.2% on I or V, and 157.7 voice moves per 100 beats against Bach's
149.3 — slightly *more* textural activity than Bach. Its cleanliness is bought by
realising a wide vocabulary carefully, not by refusing to leave I and V. Two
engines with almost the same defect rate, for completely opposite reasons.

So: a low defect count is only evidence of stiffness when it comes with a
narrowed vocabulary. The number to watch is not the defect rate, it is what the
engine is willing to play.

For reference, the commodity approach — scale-locked parallel intervals, which is
what a commercial harmonizer does — scores 159.84 defects per 100 chords, a style
divergence of 0.421, and 6.7 distinct chords per piece. The gap between it and
anything else here is the entire value of the project.

## 0a. The defect taxonomy had a hole big enough to drive a piece through

Every detector in `voicing.py` compares two sonorities: parallels, crossings,
spacing, tendency tones, melodic intervals. That is the whole of classical
voice-leading pedagogy, and it means **a harmonization that ends on the dominant
scored zero defects**. It breaks no voice-leading rule. It is also audibly wrong
in a way that no parallel fifth ever is.

`ml/theory/structure.py` adds the missing category, reasoning over the chord
sequence and phrase structure rather than over adjacent chords:
`no_tonal_closure`, `implausible_phrase_cadence`, `key_not_established`.

**The obvious version of this rule is wrong, and the oracle caught it.** Written
from theory — "a piece must end on the tonic" — it flags **15% of Bach** as
catastrophically broken. He closes ~8% of chorales on a root-position V, rising
to 12.8% in minor keys. Checking the pitch content of all 22 such pieces, only
one has a mistaken key label: the rest are Phrygian half cadences, and they are
idiomatic. So the plausible-ending tables are *measured* from the training split
and split by mode (`ml/training/calibrate_cadences.py`), and ending on V is
reported as `half_cadence_ending` at info severity — visible, with Bach's own
9.2% beside it as the yardstick — rather than scored as an error. The same
calibration turned up a smaller surprise: Bach closes phrases on viio often
enough that flagging *that* would also have been inventing an error.

Calibrated this way Bach scores **0.066 structural defects per piece**, and the
first thing the new tier found was this:

| | `no_tonal_closure` | verdict |
|---|---|---|
| `fixed_thirds` | **37.7% of pieces** | fails to resolve on more than a third of everything it writes |
| `neural_refine` | 3.3% | |
| `neural`, `neural_vl`, *Bach* | 1.6% | |
| `rules` | 0.0% | |

The commodity baseline does not close 37.7% of its harmonizations, and under the
old taxonomy that was **completely invisible** — it scored 159.84 on defects for
entirely unrelated reasons. A catastrophic, listener-obvious error was being
missed by a metric suite of fifteen detectors.

## 0b. Severity re-tiered by audibility, not by tradition

The old single "hard" bucket summed parallel fifths with voice crossings. A
listener notices a parallel octave immediately and essentially never notices that
the tenor briefly sat above the alto — and Bach crosses voices 3.33 times per 100
chords, which was **89% of his entire old "hard" score of 3.73**. The headline
number was mostly measuring the least audible thing in it.

Three tiers now, reported separately and never summed:

* **STRUCTURAL** (per piece) — wrong at the level of the whole piece. Must stay
  near Bach's 0.066 whatever else an engine does.
* **HARD** (/100 chords) — parallels and range. Bach: 0.40, not 3.73.
* **SOFT** (/100 chords) — crossing, overlap, spacing, direct perfects, tendency
  tones. Bach: **21.0**.

That last number is the argument for the whole re-tiering. *Every* engine here
scores far below Bach on SOFT — `rules` 2.7, `neural_vl` 5.0, `neural` 11.0
against his 21.0. They are all substantially more "correct" than Bach precisely
where correctness is inaudible. Under a single summed score that looked like
quality; tiered, it reads as what it is.

Severity maps onto the contract's three levels so the UI can rank them:
structural and hard are `error`, soft is `warning` or `info`, and
`half_cadence_ending` is `info` but always surfaced.

## 0c. What changed when the objective was reframed

This report originally led with a defect-rate leaderboard, on which the rule
engine won. That framing was wrong, and correcting it changed three things.

**It changed which engine is recommended.** Under "fewest defects", `rules` wins
and `neural` looks like a failure at 10.47. Under "closest to Bach, subject to a
defect budget", `neural_vl` wins outright — it is nearer Bach than `rules` on
every style axis (chord-bigram 0.071 vs 0.202, cadence 0.003 vs 0.069) *and*
nearer on vocabulary and activity, while staying well inside the guardrail.

**It produced a measurement that resolves the tension rather than trading it
off.** `ml/experiments/defect_style_tradeoff.py` sweeps the one knob that
balances the model against the rulebook. The result is that there is essentially
no trade-off to make:

| rule weight | hard defects | chord-bigram JS | chords/piece |
|---|---|---|---|
| 0.0 (model alone) | 8.56 | 0.075 | 12.5 |
| 0.15 | 0.42 | **0.071** | 12.8 |
| 0.5 (shipped) | 0.33 | 0.072 | **13.2** |
| 1.0 | 0.33 | 0.073 | 12.9 |
| 2.0 | 0.47 | 0.080 | 13.0 |
| *Bach* | *1.96* | *0.095* | *15.8* |

Going from no constraint to a light one removes **every** parallel fifth and
octave while *improving* style divergence and *increasing* chord variety. The
constraint is nearly free because of where it acts: it vetoes how a chord is
voiced, never which chord is chosen. Anything from 0.15 to 1.0 is within noise,
so the setting is not load-bearing; 0.5 ships.

**It located the rule engine's actual problem, which is not what it looked
like.** The engine is not written too safely — its chord priors are fitted to
Bach's own frequencies. It is *decoded* too safely: taking the single best path
through a first-order chain is mode-seeking, and the mode of a distribution over
progressions is blander than samples from it. Adding Gumbel perturbation to the
emission scores (`temperature > 0`, which the engine previously accepted and
silently ignored) confirms it:

| rules temperature | chords/piece | share on I or V | hard defects |
|---|---|---|---|
| 0.0 | 10.4 | 67.4% | 0.00 |
| 0.6 | 12.7 | 66.9% | 0.00 |
| 1.0 | 14.5 | 62.0% | 0.00 |
| 1.5 | **17.0** | 53.7% | 0.00 |
| *Bach* | *15.8* | *46.3%* | *1.96* |

Sampling the path posterior instead of its mode recovers Bach-like harmonic
variety at zero cost in defects — the voicing search still enforces those. But
chord-bigram divergence stays at 0.15-0.17 throughout, so sampling fixes the
engine's *vocabulary breadth* and not its *grammar*. The remaining gap is the
hand-written transition table, which is exactly the part a learned model
replaces. That is the clearest single argument in this report for the learned
engine existing at all.

## 1. The v1 post-mortem, verified — and one correction

v1 no longer exists on this branch; it survives only in git history on `main`.
All four diagnoses were checked there (`git show main:backend/model.py`,
`train.py`, `data_processor.py`, `midi_utils.py`) before being believed. Three
hold as stated. One does not, and the difference matters.

**Padding pollutes the loss — code is real, effect was not.** `chord_padding[:, 0] = 1`
is there in `main:backend/data_processor.py`, and `main:backend/train.py` takes
an unmasked mean over it in both the loss and `evaluate()`. But measured against the corpus v1 actually
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

`python -m ml.experiments.v1_postmortem` reconstructs that exact configuration —
the 14-dim features, the BiLSTM, the label projection, the 32-beat window — and
varies one factor at a time. It is a control group quarantined in
`ml/experiments/`, not a dependency: nothing in the product imports it, and it
can be deleted the moment this section has been read. Every arm is scored identically: tonic-relative chord-root
accuracy on all real beats of held-out pieces, so label spaces and sequence
schemes are directly comparable.

| configuration | root accuracy | vs v1 |
|---|---|---|
| **v1 as built** (absolute / 7 triads / 32-beat window) | **0.346** | — |
| + tonic-relative input | 0.582 | **+68.1%** |
| + whole pieces, masked loss | 0.444 | +28.1% |
| + rich label space | 0.321 | **-7.4%** |
| all three | **0.642** | **+85.4%** |

Re-run independently after the module was relocated and its symbols renamed,
seven of the eight arms reproduce to three decimal places and one
(`absolute/v1_triads/masked`) moves by 0.012, which is CPU thread-count
nondeterminism in LSTM training rather than anything meaningful. The headline
0.346 -> 0.642 is identical across runs. Treat single-arm differences below
about 0.02 as noise; every effect discussed below is five to twenty times that.

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
that stacked a root-position triad from a fixed register
(`main:backend/midi_utils.py`), so it never voiced anything and voice leading was
structurally impossible. The
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

## 4. Head-to-head, read against the oracle

**The learned model wins on harmony, decisively.** Chord-bigram divergence 0.060
against the rule engine's 0.202, with a noise floor of 0.056 — held-out Bach
measured against training Bach. `neural` is not "close to Bach"; at this sample
size it is indistinguishable from Bach. Cadence distribution 0.004 against 0.069.
Concretely, the rule engine produces 27.7% imperfect authentic cadences where
Bach produces 4.9%, and 19.7% perfect authentic where Bach produces 28.8% — it
does not know how to close a phrase like Bach. The learned engines match Bach's
cadence mix almost exactly.

**And on harmonic breadth.** 12.6 distinct chords per piece against the rule
engine's 9.4 (Bach: 14.5), and 52.9% of beats on I or V against 68.6% (Bach:
48.6%). The rule engine also essentially cannot **modulate**: a first-order chord
grammar has no representation of a phrase being temporarily in another key, so it
stays home and cadences there. The model moves to the relative major and back
without being told such a thing exists.

**The learned model loses on counterpoint, and it is not close.** Raw `neural`
writes 4.95 parallel fifths and 5.18 parallel octaves per 100 chords; Bach writes
0.21 and 0.06. At 10.47 hard defects against Bach's 3.73 it is roughly three
times the oracle, which is outside any sensible guardrail. This is structural,
not a tuning problem: blocked Gibbs resamples sites independently, so nothing in
the procedure can see that two voices are about to move in parallel. A model with
a per-site conditional cannot represent a constraint that is inherently joint.

**The fix is not a penalty term.** Mask one voice completely and the model's
logits for it no longer depend on any of its own choices, so its entire line can
be re-solved *exactly*, by Viterbi, with the model's log probabilities as
emissions and the voice-leading rules as transitions. Sweeping the three free
voices in turn is coordinate ascent on a well-defined objective. The model still
makes every harmonic decision; the rules only veto illegal ways of realising it.

That is `neural_vl`: parallel fifths **0.00**, parallel octaves **0.00**, hard
defects 10.47 -> **0.12** — and, critically, *no loss of harmonic breadth*. It
keeps 11.7 chords per piece and 53.2% on I or V, and it moves its voices more
than Bach does (157.7 per 100 beats against 149.3). It is the only engine here
that is simultaneously inside the defect guardrail and close to Bach on every
style axis.

**Is `neural_vl` at 0.12 defects also undershooting?** No, and the distinction is
the point of section 1. Undershooting matters because of what causes it. The rule
engine's near-zero comes from refusing to leave I and V; `neural_vl`'s comes from
voicing a wide vocabulary carefully. Same number, opposite meaning, and only rows
4-7 of section 1 can tell them apart. If the choice were between an engine at
Bach's 3.73 with `neural`'s style and one at 0.12 with the same style, the second
is better — being cleaner than Bach is only a fault when it is *bought* with
blandness, and here it is not.

**Where the rule engine still wins: cost and predictability.** 0.19 s/piece
against 1.20, no checkpoint, no training, and no failure mode that depends on the
melody being in distribution. It is the right default for a latency budget, and
with `temperature` raised it reaches Bach-like chord variety (section 0), though
not Bach-like grammar.

**`neural_refine` is the interesting failure.** Seeding Gibbs with the rule
engine's draft was meant to combine both. It does not: it inherits the rule
engine's harmonic habits — bigram divergence 0.116, 64.5% of beats on I or V,
21.4% imperfect cadences — without gaining anything the veto does not already
provide. A good draft is a strong attractor, and the model spends its sweeps
agreeing with it. Composing the two systems works when the rules act as a
*constraint*; it fails when they act as an *initialisation*.

## 5. What actually mattered

**Ranked by effect on the outcome:**

1. **Choosing the right objective — and it took an outside correction.** For most
   of this work the target was "fewest defects", on which the rule engine wins
   and the learned engine looks like a failure. It was the wrong target: it
   rewards being stiffer than Bach. The moment defects became a guardrail and
   distance-from-the-oracle became the objective, the recommended engine changed,
   and a knob I had assumed was a trade-off turned out to cost nothing. No amount
   of modelling would have found this, because the model was being scored
   correctly against the wrong question.
2. **The Bach oracle row.** Every number in section 1 is uninterpretable without
   it, and it is what makes "the rule engine is cleaner than Bach" legible as a
   diagnosis rather than a victory. Two "improvements" I was part-way into would
   have pushed the engines *further* from Bach.
3. **Measuring vocabulary and activity, not just correctness.** Rows 4-7 of
   section 1 are what separate two engines with near-identical defect rates and
   opposite characters. Without them, `rules` at 0.03 and `neural_vl` at 0.12
   look like the same result; with them they are not remotely the same.
4. **Building the harness before the model.** Four real bugs, three in code I had
   already convinced myself was correct, one of which (the soprano-passing-tone
   bug) was the difference between a rule engine with 2.30 parallel octaves per
   100 chords and one with zero. v1's actual failure was not that its model was
   bad; it is that it had no way to find out.
5. **Representation.** +68% in v1's setting. But see the caveat below.
6. **Making the constraint joint rather than penalised.** The difference between
   a learned engine that cannot be shipped and one that can, and it came from
   noticing that masking a whole voice makes its logits independent of itself —
   not from more capacity or more data.

**And two honest negatives.**

Section 8's ablation shows the tonic-relative representation is worth only **1.5%
perplexity** for the v2 architecture, and transposition augmentation recovers
almost all of that (**+0.2%**). That looks like it contradicts the +68% above. It
does not — it sharpens it. v1's model saw *only the melody* and had to produce a
*tonic-relative* target, so the tonic was genuinely missing information. v2's
model sees all four voices and predicts pitches from pitches: the target is in
the same frame as the input, and the key is inferable from the vertical context
anyway. **The fix was never "use tonic-relative pitch"; it was "make the target
expressible from the input".** Tonic-relative representation is one way to achieve
that, and it is still worth keeping — it is free, it makes the twelve keys share
statistics, and it is what makes the learned engines exactly
transposition-equivariant. But had I only run the v2 ablation I would have
concluded the representation barely matters, and had I only run the v1
reconstruction I would have concluded it is everything. Both experiments were
necessary.

The second: **I optimised the rule engine toward the wrong objective for most of
its development.** `ml/training/tune_rules.py` minimises a weighted sum in which
voice-leading defects carry the largest single weight, and it duly drove them to
zero — past Bach, into the narrowness section 1 now measures. The style
divergences were in the objective too, which is the only reason the engine is not
worse, but nothing in it rewarded harmonic breadth. The objective should have
included distance from the oracle's chord variety and I/V share from the start.
That the harness could measure the resulting stiffness the moment the right
columns were added is the argument for having built it first; that it took an
outside correction to look at those columns is the argument against trusting any
single scalar, including one I designed.

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
* **Phrase cadences are the one structural axis every engine is worse at than
  Bach.** `implausible_phrase_cadence` fires on 4.9% of his pieces and on
  9.8-19.7% of the engines'. Piece endings and key establishment are essentially
  solved (0-3.3%); closing *internal* phrases somewhere Bach would is not. It is
  the clearest remaining structural target and it is invisible in any
  chord-distribution metric, because a wrong chord at a fermata is one beat in
  fifty.
* **`neural_refine` is not transposition-stable.** Measured on one tune across
  six transpositions, chord-root agreement with the untransposed answer is
  0.35-0.80, against 0.50-0.85 for the rule draft it is seeded from and 1.00 for
  `neural` and `neural_vl`. Gibbs amplifies small differences in the draft rather
  than damping them. Another reason it is not the engine to ship.
* **The style metrics are distributional.** An engine could match every histogram
  and still be incoherent piece by piece. Nothing here is a substitute for
  listening, and nothing here measures phrase-level structure.

## 7. What I would try next, in order

1. **Retune the rule engine against the reframed objective.** Its tuner still
   minimises a scalar dominated by defect rate, which is what drove it to 0.03
   and 9.4 chords per piece. Adding distance from the oracle's chord variety and
   I/V share to that objective — and letting the defect term saturate once it is
   at or below Bach's rate rather than rewarding zero — should widen it
   substantially at no cost, since section 0 shows the variety is reachable with
   defects still at 0.00.
2. **Better key detection, and key changes within a piece.** 83.6% is the binding
   constraint on real input, and Krumhansl-Schmuckler on a whole melody cannot
   represent a piece that modulates. A small learned key-tracker over the melody,
   or joint inference of key and harmony, is the obvious move.
3. **Phrase-cadence awareness.** Every engine is 2-4x Bach on
   `implausible_phrase_cadence` while matching him on piece endings. The rule
   engine's cadence bonuses fire on inferred phrase ends but only bias the chord
   choice; the learned model gets a phrase-end feature but nothing that
   *constrains* what may close a phrase. A hard restriction to the measured
   plausible set at phrase-final beats would likely close this at no cost, in the
   same way the voice-leading veto did.
4. **Constrained decoding during Gibbs rather than only after it.** The polish
   currently runs as a post-pass. Folding the veto into the sampling sweeps —
   masking illegal pitches per site given the current context — would let the
   model explore *within* the legal set instead of being pulled back into it, and
   would let it explore within the legal set rather than be pulled back into it. The polish currently costs only 0.060 -> 0.071 in divergence, so the headroom is small — but section 0 suggests it may be negative, i.e. free.
5. **Model onsets.** A hold token in the DeepBach manner, plus an articulation
   metric in the harness so the improvement is measurable rather than asserted.
6. **A phrase-level metric.** Everything distributional here is first-order.
   Cadence *placement* and phrase *length* relative to the melody's own phrasing
   would catch failures the current metrics cannot see.
7. **Learn the voicing cost instead of writing it.** The polish weights are
   hand-set. They could be fit to Bach directly — the harness is already the
   objective function, and this is the same move that took the rule engine's
   guessed priors (V7 at 19% of all chords, against Bach's 2%) and replaced them
   with measured ones.
8. **Listening tests.** Every number here is a proxy. They are good proxies, and
   they are calibrated against Bach, but the project is about how something
   sounds and nothing above hears anything.

## 8. Reproducing

```bash
python -m ml.training.calibrate_rules     # chord priors from the training split
python -m ml.training.calibrate_cadences  # structural cadence tables
python -m ml.training.tune_rules          # rule weights, tuned against the harness
python -m ml.training.train_neural        # the shipped checkpoint (~30 min, CPU)
python -m ml.training.train_neural --ablation   # section 7
python -m ml.experiments.v1_postmortem    # section 2 of this discussion
python -m ml.eval.run --detect-key        # everything above
python -m pytest ml/tests -q              # 267 tests
```

Splits are a hash of the piece id, so they are stable across runs and machines;
no model here has ever seen a test piece. Every engine is deterministic given a
seed.
