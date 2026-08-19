"""Tests for the voice-leading-constrained polish.

This is the step that takes the learned engine from 4.9 parallel fifths per 100
chords to zero, so it has to actually be a veto and not merely a preference.
"""

import numpy as np
import pytest

from ml.engines._polish import PARALLEL_COST, polish_voice
from ml.theory.voicing import ALTO, BASS, TENOR, VOICE_RANGES


def uniform_log_probs(length: int, pitches: np.ndarray) -> np.ndarray:
    return np.full((length, pitches.shape[0]), -np.log(pitches.shape[0]))


def prefer(log_probs: np.ndarray, pitches: np.ndarray, choices, strength: float = 6.0) -> np.ndarray:
    """Make the model strongly want a specific pitch at each step."""
    out = log_probs.copy()
    for t, pitch in enumerate(choices):
        out[t, int(np.nonzero(pitches == pitch)[0][0])] += strength
    return out


class TestParallelVeto:
    def test_refuses_parallel_fifths_when_an_alternative_exists(self):
        # Bass moves C3 -> D3. The model wants the tenor on G3 then A3, which is
        # a textbook parallel fifth. A legal alternative must be chosen instead.
        pitches = np.arange(55, 70)
        fixed = np.array([
            [72, 74],   # soprano
            [67, 69],   # alto
            [-1, -1],   # tenor: the voice being re-solved
            [48, 50],   # bass
        ])
        log_probs = prefer(uniform_log_probs(2, pitches), pitches, [55, 57], strength=3.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(2, dtype=bool))
        assert not (chosen[0] == 55 and chosen[1] == 57)

    def test_allows_a_fifth_that_is_not_parallel(self):
        # Bass holds, so the tenor may sit on a fifth above it at both steps.
        pitches = np.arange(55, 70)
        fixed = np.array([[72, 74], [67, 69], [-1, -1], [48, 48]])
        log_probs = prefer(uniform_log_probs(2, pitches), pitches, [55, 55], strength=6.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(2, dtype=bool))
        assert list(chosen) == [55, 55]

    def test_refuses_parallel_octaves(self):
        pitches = np.arange(48, 65)
        fixed = np.array([[72, 74], [67, 69], [-1, -1], [48, 50]])
        log_probs = prefer(uniform_log_probs(2, pitches), pitches, [60, 62], strength=3.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(2, dtype=bool))
        assert not (chosen[0] == 60 and chosen[1] == 62)

    def test_an_overwhelming_preference_still_loses_to_a_parallel(self):
        """The veto must be strong enough to survive a confident model."""
        pitches = np.arange(55, 70)
        fixed = np.array([[72, 74], [67, 69], [-1, -1], [48, 50]])
        log_probs = prefer(uniform_log_probs(2, pitches), pitches, [55, 57], strength=PARALLEL_COST - 6.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(2, dtype=bool))
        assert not (chosen[0] == 55 and chosen[1] == 57)

    def test_oblique_motion_is_never_a_parallel(self):
        pitches = np.arange(55, 70)
        fixed = np.array([[72, 72], [67, 67], [-1, -1], [48, 48]])
        log_probs = prefer(uniform_log_probs(2, pitches), pitches, [60, 62], strength=6.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(2, dtype=bool))
        assert list(chosen) == [60, 62]


class TestVerticalConstraints:
    def test_avoids_crossing_above_the_alto(self):
        pitches = np.arange(48, 75)
        fixed = np.array([[72], [64], [-1], [48]])
        log_probs = prefer(uniform_log_probs(1, pitches), pitches, [70], strength=6.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(1, dtype=bool))
        assert chosen[0] <= 64

    def test_avoids_crossing_below_the_bass(self):
        pitches = np.arange(48, 70)
        fixed = np.array([[72], [67], [-1], [60]])
        log_probs = prefer(uniform_log_probs(1, pitches), pitches, [50], strength=6.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(1, dtype=bool))
        assert chosen[0] >= 60

    def test_stays_inside_the_voice_range(self):
        low, high = VOICE_RANGES[ALTO]
        pitches = np.arange(low - 6, high + 7)
        fixed = np.array([[81], [-1], [60], [48]])
        log_probs = prefer(uniform_log_probs(1, pitches), pitches, [high + 5], strength=6.0)
        chosen = polish_voice(log_probs, pitches, fixed, ALTO, active=np.ones(1, dtype=bool))
        assert low <= chosen[0] <= high

    def test_respects_spacing_to_the_voice_above(self):
        pitches = np.arange(53, 75)
        fixed = np.array([[79], [-1], [55], [43]])
        log_probs = prefer(uniform_log_probs(1, pitches), pitches, [55], strength=2.0)
        chosen = polish_voice(log_probs, pitches, fixed, ALTO, active=np.ones(1, dtype=bool))
        assert 79 - chosen[0] <= 12


class TestBehaviour:
    def test_follows_the_model_when_nothing_is_illegal(self):
        pitches = np.arange(55, 70)
        fixed = np.array([[72, 72, 72], [69, 69, 69], [-1, -1, -1], [48, 48, 48]])
        wanted = [60, 62, 64]
        log_probs = prefer(uniform_log_probs(3, pitches), pitches, wanted, strength=8.0)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(3, dtype=bool))
        assert list(chosen) == wanted

    def test_inactive_steps_are_left_untouched(self):
        pitches = np.arange(55, 70)
        fixed = np.array([[72, -1, 72], [69, -1, 69], [60, -1, 62], [48, -1, 48]])
        active = np.array([True, False, True])
        log_probs = uniform_log_probs(3, pitches)
        chosen = polish_voice(log_probs, pitches, fixed, TENOR, active=active)
        assert chosen[1] == fixed[TENOR, 1]

    def test_is_deterministic(self):
        rng = np.random.default_rng(3)
        pitches = np.arange(48, 66)
        fixed = np.array([
            rng.integers(66, 78, 12), rng.integers(58, 70, 12),
            np.full(12, -1), rng.integers(40, 55, 12),
        ])
        log_probs = rng.normal(size=(12, pitches.shape[0]))
        first = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(12, dtype=bool))
        second = polish_voice(log_probs, pitches, fixed, TENOR, active=np.ones(12, dtype=bool))
        assert np.array_equal(first, second)

    def test_returns_one_pitch_per_step(self):
        pitches = np.arange(40, 60)
        fixed = np.array([[72] * 5, [67] * 5, [60] * 5, [-1] * 5])
        chosen = polish_voice(uniform_log_probs(5, pitches), pitches, fixed, BASS, active=np.ones(5, dtype=bool))
        assert chosen.shape == (5,)
        assert all(pitch in pitches for pitch in chosen)

    def test_empty_active_returns_the_input_line(self):
        pitches = np.arange(48, 60)
        fixed = np.array([[72], [67], [55], [-1]])
        chosen = polish_voice(uniform_log_probs(1, pitches), pitches, fixed, BASS, active=np.zeros(1, dtype=bool))
        assert list(chosen) == list(fixed[BASS])
