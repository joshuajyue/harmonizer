# HarmonAIzer v2 — engine evaluation

Generated 2026-08-19 07:43 UTC.

Bach chorale corpus, piece-level split by hash of the piece id: 263 train / 44 val / 61 test. Every engine sees the same held-out sopranos and nothing else.

## 1. Voice-leading defects per 100 chord changes

Objective and engine-agnostic: these are counted by the same detectors for every row, including Bach's. The `bach_oracle` column is the calibration — it is what the ceiling scores under these exact definitions, and it is not zero.

| defect / 100 chords      | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|--------------------------|--------------|-------|--------|---------------|-----------|-------------|
| parallel_fifths          | 72.61        | 0.00  | 4.90   | 0.00          | 0.00      | 0.21        |
| parallel_octaves         | 87.19        | 0.00  | 5.29   | 0.00          | 0.00      | 0.06        |
| contrary_fifths          | 2.82         | 0.00  | 0.31   | 0.00          | 0.00      | 0.23        |
| contrary_octaves         | 3.49         | 0.00  | 0.23   | 0.02          | 0.00      | 0.08        |
| direct_fifths            | 0.00         | 0.00  | 0.67   | 0.28          | 0.39      | 0.34        |
| direct_octaves           | 0.04         | 0.13  | 0.16   | 0.38          | 0.20      | 0.06        |
| voice_crossing           | 0.00         | 0.00  | 0.26   | 0.00          | 0.00      | 3.33        |
| voice_overlap            | 20.96        | 0.16  | 3.71   | 2.72          | 1.56      | 9.41        |
| spacing                  | 2.63         | 0.03  | 0.23   | 0.06          | 0.00      | 2.23        |
| range                    | 0.04         | 0.03  | 0.03   | 0.02          | 0.02      | 0.13        |
| unresolved_leading_tone  | 0.67         | 0.10  | 0.83   | 0.45          | 0.57      | 0.40        |
| unresolved_seventh       | 0.16         | 1.48  | 1.66   | 0.64          | 0.87      | 1.89        |
| doubled_leading_tone     | 0.00         | 0.07  | 0.41   | 0.49          | 0.35      | 0.72        |
| awkward_melodic_interval | 6.66         | 0.00  | 1.50   | 0.00          | 0.00      | 1.29        |
| large_leap               | 1.29         | 0.43  | 1.06   | 0.62          | 0.99      | 1.02        |
| **HARD TOTAL**           | 159.84       | 0.03  | 10.47  | 0.02          | 0.02      | 3.73        |

HARD TOTAL sums the unambiguous errors: parallel fifths, parallel octaves, voice crossing and range violations.

## 2. Style distance from the Bach corpus

Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint) against the **training** split. `bach_oracle` is held-out Bach measured against training Bach, so its value is the noise floor: no engine can meaningfully score below it.

| JS divergence       | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|---------------------|--------------|-------|--------|---------------|-----------|-------------|
| chord_unigram_js    | 0.169        | 0.108 | 0.020  | 0.055         | 0.021     | 0.007       |
| chord_bigram_js     | 0.421        | 0.199 | 0.060  | 0.113         | 0.067     | 0.056       |
| root_motion_js      | 0.264        | 0.050 | 0.008  | 0.022         | 0.009     | 0.001       |
| inversion_js        | 0.904        | 0.016 | 0.005  | 0.010         | 0.010     | 0.000       |
| quality_js          | 0.100        | 0.030 | 0.008  | 0.011         | 0.011     | 0.000       |
| cadence_js          | 0.372        | 0.092 | 0.004  | 0.055         | 0.007     | 0.004       |
| melodic_interval_js | 0.040        | 0.055 | 0.003  | 0.015         | 0.010     | 0.001       |
| outer_motion_js     | 0.651        | 0.029 | 0.008  | 0.004         | 0.001     | 0.001       |

### Style mix

| share of chords       | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------------------|--------------|-------|--------|---------------|-----------|-------------|
| seventh_chords        | 0.2%         | 12.2% | 10.9%  | 11.2%         | 9.0%      | 15.3%       |
| applied_chords        | 0.0%         | 10.0% | 8.2%   | 8.0%          | 8.6%      | 10.8%       |
| root_position         | 0.0%         | 68.8% | 72.2%  | 72.8%         | 73.4%     | 68.4%       |
| first_inversion       | 0.0%         | 30.0% | 25.3%  | 25.7%         | 25.0%     | 25.2%       |
| second_inversion      | 99.8%        | 1.2%  | 1.8%   | 1.1%          | 1.2%      | 3.5%        |
| contrary_outer_motion | 3.7%         | 51.8% | 32.9%  | 34.4%         | 31.1%     | 33.6%       |
| parallel_outer_motion | 89.5%        | 2.4%  | 7.8%   | 2.5%          | 4.9%      | 5.6%        |

### Cadence types

| cadence   | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------|--------------|-------|--------|---------------|-----------|-------------|
| HC        | 0.0%         | 23.6% | 26.6%  | 23.0%         | 23.6%     | 31.5%       |
| IAC       | 2.5%         | 31.8% | 6.0%   | 23.8%         | 5.2%      | 4.9%        |
| PAC       | 0.0%         | 18.1% | 28.5%  | 26.0%         | 28.5%     | 28.8%       |
| deceptive | 1.6%         | 0.3%  | 1.9%   | 0.8%          | 1.9%      | 0.8%        |
| other     | 95.9%        | 23.0% | 33.7%  | 23.6%         | 36.2%     | 31.2%       |
| phrygian  | 0.0%         | 1.1%  | 0.3%   | 0.5%          | 0.5%      | 0.0%        |
| plagal    | 0.0%         | 2.2%  | 3.0%   | 2.2%          | 4.1%      | 2.7%        |

## 3. Held-out likelihood

Negative log-likelihood in nats per predicted note token, and its perplexity, for Bach's own alto/tenor/bass on the held-out split. Only defined for probabilistic engines; a rule engine has no likelihood to report.

| metric           | fixed_thirds | rules | neural  | neural_refine | neural_vl | bach_oracle |
|------------------|--------------|-------|---------|---------------|-----------|-------------|
| NLL (nats/token) | n/a          | n/a   | -0.5123 | -0.5123       | -0.5123   | n/a         |
| perplexity       | n/a          | n/a   | 0.599   | 0.599         | 0.599     | n/a         |

## 4. Agreement with Bach — reported, NOT the headline

This is the metric v1 optimised. It is included for continuity and because watching it move independently of sections 1-3 is itself the argument against it: a harmonization can disagree with Bach on most beats and still be excellent, and v1's version of this number additionally counted padded positions.

| agreement          | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|--------------------|--------------|-------|--------|---------------|-----------|-------------|
| chord_exact        | 0.5%         | 28.9% | 41.1%  | 35.0%         | 41.9%     | 100.0%      |
| chord_root_quality | 18.2%        | 40.4% | 50.0%  | 44.0%         | 50.2%     | 100.0%      |
| chord_root         | 27.5%        | 49.0% | 56.3%  | 51.4%         | 56.5%     | 100.0%      |
| voice_note         | 15.4%        | 29.6% | 40.2%  | 35.2%         | 39.9%     | 100.0%      |
| bass_note          | 5.5%         | 25.1% | 36.9%  | 32.1%         | 37.2%     | 100.0%      |

## 5. Cost and robustness

| metric          | fixed_thirds | rules | neural | neural_refine | neural_vl | bach_oracle |
|-----------------|--------------|-------|--------|---------------|-----------|-------------|
| pieces scored   | 61           | 61    | 61     | 61            | 61        | 61          |
| failures        | 0            | 0     | 0      | 0             | 0         | 0           |
| seconds / piece | 0.021        | 0.225 | 3.713  | 1.540         | 1.006     | 0.000       |

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
| HARD TOTAL / 100 chords | 151.41       | 0.03  | 11.22  | 0.02          | 0.06      |
| chord_bigram_js         | 0.405        | 0.184 | 0.073  | 0.118         | 0.079     |
| chord agreement         | 0.5%         | 26.8% | 37.1%  | 33.1%         | 37.7%     |

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
* `rules`: (7, 'maj') 24.9%, (0, 'maj') 22.7%, (0, 'min') 17.2%, (5, 'maj') 7.0%, (7, 'dom7') 5.7%, (10, 'maj') 5.4%, (5, 'min') 2.8%, (2, 'maj') 2.4%
* `neural`: (0, 'maj') 21.1%, (7, 'maj') 19.7%, (0, 'min') 10.8%, (3, 'maj') 8.3%, (5, 'maj') 5.4%, (10, 'maj') 5.2%, (9, 'min') 3.9%, (5, 'min') 3.1%
* `neural_refine`: (0, 'maj') 24.5%, (7, 'maj') 22.7%, (0, 'min') 15.3%, (5, 'maj') 5.5%, (10, 'maj') 5.0%, (7, 'dom7') 4.5%, (9, 'min') 3.8%, (5, 'min') 3.6%
* `neural_vl`: (0, 'maj') 20.2%, (7, 'maj') 19.7%, (0, 'min') 11.6%, (3, 'maj') 6.7%, (5, 'maj') 5.8%, (10, 'maj') 5.2%, (9, 'min') 4.9%, (5, 'min') 4.5%
* `bach_oracle`: (7, 'maj') 17.5%, (0, 'maj') 17.3%, (0, 'min') 10.6%, (3, 'maj') 5.8%, (5, 'maj') 5.5%, (10, 'maj') 4.5%, (9, 'min') 4.0%, (5, 'min') 3.8%


<!-- NARRATIVE -->
