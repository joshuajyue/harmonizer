"""Voice-leading-constrained polish for the learned engine.

The head-to-head result this exists to fix: the learned model has Bach's
harmonic vocabulary — its chord-transition distribution is closer to the corpus
than the rule engine's, and closer than held-out Bach is to training Bach — but
it writes parallel fifths and octaves at roughly twenty times Bach's rate.
Blocked Gibbs resamples sites independently, so nothing in the procedure can see
that two voices are about to move in parallel.

The fix is not to add a penalty term and hope. Mask ONE voice completely and the
model's logits for that voice no longer depend on any of its own choices, so its
entire line can be re-chosen exactly, by Viterbi, with the model's log
probabilities as emissions and the voice-leading rulebook as transitions. The
other three voices are fixed, so every parallel, crossing and overlap involving
them is decidable. Sweeping the three free voices in turn is coordinate ascent
on a well-defined objective.

The model still makes every harmonic decision. The rules only veto illegal ways
of realising it.
"""

from __future__ import annotations

import numpy as np

from ..theory.voicing import MAX_SPACING, VOICE_RANGES, SOPRANO, ALTO, TENOR, BASS

#: Cost of a parallel perfect consonance. Large enough to be a veto in practice
#: but finite, so the search still returns something when a melody leaves no
#: legal option at all.
PARALLEL_COST = 24.0
CONTRARY_PERFECT_COST = 3.0
CROSSING_COST = 18.0
OVERLAP_COST = 1.2
SPACING_COST = 2.5
RANGE_COST = 12.0
LEAP_COST = 0.16
AWKWARD_LEAP_COST = 4.0
OCTAVE_LEAP_COST = 5.0

#: How much a rule costs relative to one nat of model log-probability. Raising
#: this buys correctness at the price of style; the value is chosen on the
#: validation split.
RULE_WEIGHT = 1.0

_NEIGHBOURS = {ALTO: (SOPRANO, TENOR), TENOR: (ALTO, BASS), BASS: (TENOR, None)}
_AWKWARD = np.array([6, 10, 11, 13, 14])


def _melodic_costs(pitches: np.ndarray) -> np.ndarray:
    """(P, P) cost of moving between two pitches within one voice."""
    delta = np.abs(pitches[None, :] - pitches[:, None])
    cost = np.where(delta > 2, LEAP_COST * (delta - 2), 0.0)
    cost += np.where(np.isin(delta, _AWKWARD), AWKWARD_LEAP_COST, 0.0)
    cost += np.where(delta > 12, OCTAVE_LEAP_COST, 0.0)
    return cost.astype(np.float64)


def _static_costs(pitches: np.ndarray, fixed: np.ndarray, voice: int) -> np.ndarray:
    """(T, P) cost of each pitch at each step, from the vertical alone."""
    length = fixed.shape[1]
    cost = np.zeros((length, pitches.shape[0]), dtype=np.float64)
    # The model's alphabet is deliberately wider than the singable range so a
    # user melody outside the corpus can still be harmonized; the search has to
    # be told not to use that headroom unless it must.
    low, high = VOICE_RANGES[voice]
    out_of_range = RANGE_COST * (np.maximum(0, low - pitches) + np.maximum(0, pitches - high))
    cost += out_of_range[None, :]
    upper, lower = _NEIGHBOURS[voice]
    for t in range(length):
        if upper is not None:
            above = fixed[upper, t]
            if above >= 0:
                cost[t] += np.where(pitches > above, CROSSING_COST, 0.0)
                limit = MAX_SPACING.get((upper, voice), 12)
                cost[t] += np.where(above - pitches > limit, SPACING_COST, 0.0)
        if lower is not None:
            below = fixed[lower, t]
            if below >= 0:
                cost[t] += np.where(pitches < below, CROSSING_COST, 0.0)
                limit = MAX_SPACING.get((voice, lower), 12)
                cost[t] += np.where(pitches - below > limit, SPACING_COST, 0.0)
    return cost


def _parallel_costs(pitches: np.ndarray, fixed: np.ndarray, voice: int, t: int) -> np.ndarray | None:
    """(P, P) parallel/overlap cost for moving from step t-1 to step t.

    Returns None when no other voice moves, in which case no parallel is
    possible and the caller can reuse the precomputed melodic matrix — which is
    the common case on a sixteenth grid and most of the speed.
    """
    movers = [w for w in range(4) if w != voice and fixed[w, t] != fixed[w, t - 1] and fixed[w, t] >= 0 and fixed[w, t - 1] >= 0]
    if not movers:
        return None

    size = pitches.shape[0]
    cost = np.zeros((size, size), dtype=np.float64)
    moved = (pitches[None, :] != pitches[:, None])
    direction = np.sign(pitches[None, :] - pitches[:, None])

    for w in movers:
        before, after = int(fixed[w, t - 1]), int(fixed[w, t])
        other_direction = np.sign(after - before)
        previous_interval = np.abs(pitches - before) % 12
        current_interval = np.abs(pitches - after) % 12

        was_fifth, now_fifth = previous_interval == 7, current_interval == 7
        was_octave, now_octave = previous_interval == 0, current_interval == 0
        perfect = (was_fifth[:, None] & now_fifth[None, :]) | (was_octave[:, None] & now_octave[None, :])
        same_direction = direction == other_direction
        cost += np.where(moved & perfect & same_direction, PARALLEL_COST, 0.0)
        cost += np.where(moved & perfect & ~same_direction, CONTRARY_PERFECT_COST, 0.0)

    upper, lower = _NEIGHBOURS[voice]
    if upper is not None and fixed[upper, t - 1] >= 0:
        cost += np.where(pitches[None, :] > fixed[upper, t - 1], OVERLAP_COST, 0.0)
    if lower is not None and fixed[lower, t - 1] >= 0:
        cost += np.where(pitches[None, :] < fixed[lower, t - 1], OVERLAP_COST, 0.0)
    return cost


def polish_voice(
    log_probs: np.ndarray,
    pitches: np.ndarray,
    fixed: np.ndarray,
    voice: int,
    *,
    active: np.ndarray,
    rule_weight: float = RULE_WEIGHT,
) -> np.ndarray:
    """Re-choose one voice's whole line by Viterbi.

    `log_probs` is (T, P) from the model with this voice fully masked, so it does
    not depend on the voice's own current notes and the search is exact.
    `fixed` is (4, T) current pitches; row `voice` is ignored. `active` marks the
    steps this voice actually sings.

    Returns the chosen pitches, one per step (unchanged where inactive).
    """
    length, size = log_probs.shape
    melodic = _melodic_costs(pitches)
    static = _static_costs(pitches, fixed, voice) * rule_weight
    emission = -log_probs + static

    steps = np.nonzero(active)[0]
    if steps.size == 0:
        return fixed[voice].copy()

    dp = emission[steps[0]].copy()
    back = np.zeros((steps.size, size), dtype=np.int32)
    for index in range(1, steps.size):
        t = steps[index]
        previous_t = steps[index - 1]
        transition = melodic.copy()
        if t == previous_t + 1:
            extra = _parallel_costs(pitches, fixed, voice, t)
            if extra is not None:
                transition = transition + extra
        total = dp[:, None] + transition * rule_weight
        choice = np.argmin(total, axis=0)
        dp = total[choice, np.arange(size)] + emission[t]
        back[index] = choice

    out = fixed[voice].copy()
    cursor = int(np.argmin(dp))
    for index in range(steps.size - 1, -1, -1):
        out[steps[index]] = pitches[cursor]
        cursor = int(back[index][cursor])
    return out
