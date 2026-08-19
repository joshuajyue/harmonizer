"""Functional-harmony rule engine: the baseline the learned model has to beat.

This is a deliberate strengthening of v1's "creative engine", which chose one
diatonic triad per measure by greedy best-fit and then handed the label to a
renderer that stacked a root-position triad from a fixed register. That renderer
made voice leading structurally impossible, so the rule engine v1 lost to was
never actually voicing anything.

Two changes carry almost all of the improvement:

1. Chord choice is a Viterbi search over a rich vocabulary with a functional
   transition grammar, not a greedy per-measure argmax. Greedy cannot plan a
   cadence, and a measure-level decision cannot express harmonic rhythm.
2. The engine emits actual SATB voices, chosen by a second Viterbi whose
   transition cost *is* the voice-leading rulebook. Parallel fifths, unresolved
   tendency tones and bad spacing are priced into the search, so the output is
   optimised for the thing we then measure.

A weak baseline would make the head-to-head worthless, so this is built to be
genuinely hard to beat.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Sequence

import numpy as np

from contracts.schema import Chord, KeySignature, Melody, Violation

from ..data.corpus import REST, STEP
from ..data.melody import MelodyGrid, detect_melody_key, grid_to_voices, melody_to_grid, select_voices
from ..theory.chords import ChordLabel, build_vocabulary
from ..theory.pitch import Key
from ..theory.voicing import (
    ALTO,
    BASS,
    MAX_SPACING,
    PREFERRED_RANGES,
    SOPRANO,
    TENOR,
    VOICE_NAMES,
    VOICE_RANGES,
    analyze_texture,
    dominant_target,
    texture_from_voices,
)
from .base import Harmonization, HarmonyEngine, register

# ---------------------------------------------------------------------------
# Harmonic scoring
# ---------------------------------------------------------------------------

#: How strongly a melody note argues for or against a chord, by metric weight.
CHORD_TONE_BONUS = 2.2
NEIGHBOUR_PENALTY = {0: -0.05, 1: -0.15, 2: -0.9, 3: -1.4}
FOREIGN_PENALTY = {0: -0.8, 1: -1.2, 2: -2.6, 3: -3.4}

#: Prior on each chord, tonic-relative, before any melody evidence. Estimated
#: from the training split by `python -m ml.training.calibrate_rules`: how often
#: Bach reaches for a chord is a measurable fact, not a judgement call, and
#: hand-set numbers had V7 at 19% of all chords where Bach uses it at 2%. The
#: functional grammar below stays hand-written — only the frequencies are fit.
from ._rule_priors import INVERSION_PRIOR as _FITTED_INVERSION_PRIOR
from ._rule_priors import MAJOR_PRIOR as _MAJOR_PRIOR
from ._rule_priors import MINOR_PRIOR as _MINOR_PRIOR

UNKNOWN_CHORD_PRIOR = -2.0

#: Penalty for each inversion. Second inversion is a special case in tonal
#: practice, not a free choice, so it starts deep in the hole and is redeemed
#: only by the cadential-6-4 transition bonus.
INVERSION_PRIOR = dict(_FITTED_INVERSION_PRIOR)
#: The fitted marginal is not a strong enough prior for the six-four. Every
#: inversion is a free alternative to the search — same pitch classes, smoother
#: bass — so the prior has to counteract a systematic pull the marginal
#: frequency does not know about. A six-four is a specific contrapuntal event
#: (cadential, passing, pedal), not a voicing convenience.
INVERSION_PRIOR[2] = -2.9
APPLIED_PRIOR = -1.4
#: Sevenths match more melody notes than triads simply by having more notes, so
#: the emission term structurally prefers them; the fitted unigram priors alone
#: cannot correct a bias that scales with how much melody is in the slot.
SEVENTH_PENALTY = -0.35
#: A cadential six-four sits on a metrically STRONG beat and resolves on a
#: weaker one. Anywhere else it needs a stepwise bass to justify itself, which
#: this engine does not attempt, so it is priced out.
WEAK_BEAT_64_PENALTY = -2.6
#: Cadences want root position in both chords; an inverted cadence turns what
#: should be a PAC into an IAC.
CADENCE_INVERSION_PENALTY = -1.5

FUNCTION_MATRIX = {
    ("T", "T"): 0.15, ("T", "PD"): 1.0, ("T", "D"): 0.8, ("T", "A"): 0.55,
    ("PD", "T"): -0.15, ("PD", "PD"): 0.2, ("PD", "D"): 1.5, ("PD", "A"): 0.45,
    ("D", "T"): 1.5, ("D", "PD"): -1.6, ("D", "D"): 0.3, ("D", "A"): -0.6,
    ("A", "T"): 0.0, ("A", "PD"): 0.0, ("A", "D"): 0.0, ("A", "A"): 0.0,
}

#: Bonus by root motion in semitones (from -> to), the classic strength order.
ROOT_MOTION = {
    0: 0.0, 1: -0.5, 2: 0.6, 3: -0.2, 4: -0.35, 5: 1.0,
    6: -0.4, 7: 0.2, 8: 0.15, 9: 0.5, 10: 0.3, 11: -0.5,
}

CADENCE_TONIC_BONUS = 2.0
CADENCE_DOMINANT_BONUS = 1.2
PRE_CADENCE_DOMINANT_BONUS = 0.9
FINAL_TONIC_BONUS = 5.0
OPENING_TONIC_BONUS = 1.4

# ---------------------------------------------------------------------------
# Voice-leading costs
# ---------------------------------------------------------------------------

PARALLEL_PERFECT_COST = 9.0
CONTRARY_PERFECT_COST = 2.2
DIRECT_PERFECT_COST = 1.8
OVERLAP_COST = 1.1
UNRESOLVED_LT_COST = 3.0
FRUSTRATED_LT_COST = 0.7
UNRESOLVED_SEVENTH_COST = 3.0
UPPER_LEAP_COST = 0.22
BASS_LEAP_COST = 0.09
AWKWARD_MELODIC_COST = 3.5
OVER_OCTAVE_LEAP_COST = 4.5
COMMON_TONE_BONUS = -0.32
CONTRARY_BASS_BONUS = -0.42
ALL_SIMILAR_COST = 0.35

DOUBLE_THIRD_MAJOR_COST = 0.7
DOUBLE_THIRD_MINOR_COST = 0.25
DOUBLE_FIFTH_COST = 0.35
MISSING_FIFTH_COST = 0.25
TESSITURA_COST = 0.045
HARMONY_WEIGHT = 1.35

MAX_CHORD_CANDIDATES = 6
MAX_VOICINGS_PER_CHORD = 22


@dataclass(frozen=True)
class RuleConfig:
    """Tunable harmonic weights.

    Grouped into one object so `ml/training/tune_rules.py` can search them on
    the validation split with the eval harness as the objective, instead of me
    guessing numbers and calling the result a baseline. The voice-leading costs
    are deliberately NOT in here: they already drive parallel fifths, parallel
    octaves and range violations to zero, and tuning a metric that reads zero
    only risks overfitting it.
    """

    chord_tone_bonus: float = CHORD_TONE_BONUS
    seventh_penalty: float = SEVENTH_PENALTY
    applied_prior: float = APPLIED_PRIOR
    first_inversion_prior: float = INVERSION_PRIOR.get(1, -0.66)
    six_four_prior: float = INVERSION_PRIOR[2]
    third_inversion_prior: float = INVERSION_PRIOR.get(3, -2.09)
    weak_beat_64_penalty: float = WEAK_BEAT_64_PENALTY
    cadence_tonic_bonus: float = CADENCE_TONIC_BONUS
    cadence_dominant_bonus: float = CADENCE_DOMINANT_BONUS
    cadence_inversion_penalty: float = CADENCE_INVERSION_PENALTY
    pre_cadence_dominant_bonus: float = PRE_CADENCE_DOMINANT_BONUS
    root_position_dominant_bonus: float = 0.8
    harmony_weight: float = HARMONY_WEIGHT

    def inversion_prior(self, inversion: int) -> float:
        return {
            0: 0.0,
            1: self.first_inversion_prior,
            2: self.six_four_prior,
            3: self.third_inversion_prior,
        }.get(inversion, -1.0)


#: Weights found by `python -m ml.training.tune_rules`, which uses the eval
#: harness on the validation split as its objective. Checked in as data so the
#: engine stays deterministic and import-cheap, and so the numbers in the report
#: can be reproduced exactly.
_TUNED_CONFIG_PATH = Path(__file__).resolve().parent / "_rule_config.json"


def load_config(path: Path = _TUNED_CONFIG_PATH) -> RuleConfig:
    if not path.exists():
        return RuleConfig()
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return RuleConfig()
    known = {f.name for f in dataclass_fields(RuleConfig)}
    return RuleConfig(**{k: v for k, v in payload.items() if k in known})


DEFAULT_CONFIG = load_config()


def chord_function(chord: ChordLabel, mode: str) -> str:
    if chord.applied_to is not None:
        return "A"
    if dominant_target(chord) is not None:
        return "D"
    root = chord.relative_root
    if mode == "major":
        if root in (0, 9, 4):
            return "T"
        if root in (5, 2, 1, 8):
            return "PD"
    else:
        if root in (0, 3, 8):
            return "T"
        if root in (5, 2, 1, 10):
            return "PD"
    return "PD"


def is_six_four(chord: ChordLabel) -> bool:
    return chord.inversion == 2 and not chord.is_seventh


def is_cadential_64(chord: ChordLabel) -> bool:
    return chord.relative_root == 0 and chord.inversion == 2 and chord.quality in ("maj", "min")


def chord_prior(chord: ChordLabel, mode: str, config: RuleConfig = DEFAULT_CONFIG) -> float:
    table = _MINOR_PRIOR if mode == "minor" else _MAJOR_PRIOR
    base = table.get((chord.relative_root, chord.quality))
    if base is None:
        base = config.applied_prior if chord.applied_to is not None else UNKNOWN_CHORD_PRIOR
    if chord.is_seventh:
        base += config.seventh_penalty
    return base + config.inversion_prior(chord.inversion)


def transition_score(a: ChordLabel, b: ChordLabel, mode: str) -> float:
    if a.key() == b.key():
        return -0.15
    if a.relative_root == b.relative_root and a.quality == b.quality:
        return 0.35  # same chord, new inversion: free bass motion
    if a.relative_root == b.relative_root:
        return 0.15  # e.g. IV -> iv, I -> I7

    score = FUNCTION_MATRIX[(chord_function(a, mode), chord_function(b, mode))]
    score += ROOT_MOTION[(b.relative_root - a.relative_root) % 12]

    if a.applied_to is not None:
        score += 3.2 if b.relative_root == a.applied_to else -3.4

    target = dominant_target(a)
    if target is not None and a.applied_to is None:
        deceptive = 9 if mode == "major" else 8
        if b.relative_root == target:
            score += 2.4
        elif b.relative_root == deceptive and dominant_target(b) is None:
            score += 0.9  # deceptive cadence
        elif dominant_target(b) is not None:
            score += 0.6  # V -> V7, viio -> V etc.
        else:
            score -= 2.2

    if is_cadential_64(a):
        score += 2.2 if b.relative_root == 7 else -3.2
    if is_cadential_64(b) and chord_function(a, mode) == "PD":
        score += 0.7

    # A diatonic major triad a fifth above another degree tonicizes it without
    # any chromaticism — VII -> III in minor is the standard route to the
    # relative major and has to be reachable, or the engine can never leave the
    # home key.
    if a.quality == "maj" and a.applied_to is None and dominant_target(a) is None:
        if b.relative_root == (a.relative_root + 5) % 12 and b.quality in ("maj", "min"):
            score += 0.9

    if a.relative_root == 1 and a.quality == "maj":  # Neapolitan
        score += 2.4 if b.relative_root in (7, 11) else -1.2

    return score


def build_transition_matrix(vocab: Sequence[ChordLabel], mode: str) -> np.ndarray:
    n = len(vocab)
    matrix = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(vocab):
        for j, b in enumerate(vocab):
            matrix[i, j] = transition_score(a, b, mode)
    return matrix


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------


@dataclass
class Slot:
    """One harmonic decision point: a beat's worth of melody."""

    start: int
    stop: int
    strength: int
    #: 3 = downbeat, 2 = secondary strong beat (halfway through the measure),
    #: 1 = any other beat. `strength` alone cannot express this: on a
    #: quarter-note slot grid every slot start is already a beat, so a test on
    #: `strength < 2` is never true and every metric rule keyed to it is dead.
    metric_level: int = 1
    events: list[tuple[int, int, int]] = field(default_factory=list)  # (pitch, steps, strength)
    principal: int | None = None
    final: int | None = None
    lowest: int | None = None
    phrase_end: bool = False

    @property
    def is_rest(self) -> bool:
        return self.principal is None


def build_slots(grid: MelodyGrid) -> list[Slot]:
    """Cut the melody into beat-length harmonic slots."""
    slots: list[Slot] = []
    step_size = grid.steps_per_beat
    measure = grid.steps_per_measure
    offset = (measure - grid.pickup_steps) % measure
    half = measure // 2
    for start in range(0, grid.length, step_size):
        stop = min(start + step_size, grid.length)
        events: list[tuple[int, int, int]] = []
        run_pitch, run_start = None, start
        for t in range(start, stop):
            pitch = grid.pitches[t]
            new_event = pitch != run_pitch or (grid.onsets[t] and t > start)
            if new_event:
                if run_pitch not in (None, REST):
                    events.append((run_pitch, t - run_start, grid.beat_strength[run_start]))
                run_pitch, run_start = pitch, t
        if run_pitch not in (None, REST):
            events.append((run_pitch, stop - run_start, grid.beat_strength[run_start]))

        pitches = [p for p, _, _ in events]
        position = (start + offset) % measure
        if position == 0:
            metric_level = 3
        elif half and position == half:
            metric_level = 2
        else:
            metric_level = 1
        slots.append(Slot(
            start=start,
            stop=stop,
            strength=grid.beat_strength[start],
            metric_level=metric_level,
            events=events,
            principal=events[0][0] if events else None,
            final=events[-1][0] if events else None,
            lowest=min(pitches) if pitches else None,
            phrase_end=any(grid.phrase_end[t] for t in range(start, stop)),
        ))
    return slots


def emission_scores(
    slots: Sequence[Slot],
    vocab: Sequence[ChordLabel],
    key: Key,
    config: RuleConfig = DEFAULT_CONFIG,
) -> np.ndarray:
    """(n_slots, n_chords) fit of each chord to each slot's melody."""
    mode = key.mode
    priors = np.array([chord_prior(chord, mode, config) for chord in vocab], dtype=np.float32)
    members = [set(chord.pitch_classes) for chord in vocab]
    scores = np.tile(priors, (len(slots), 1))

    # A phrase-final note is long, so it spans several slots. The cadence chord
    # is struck at the START of that note and then held: applying the
    # pre-cadence dominant bonus before *every* slot of the run pushes a
    # dominant onto the middle of the final note and turns perfect authentic
    # cadences into imperfect ones.
    cadence_runs: list[tuple[int, int]] = []
    for index, slot in enumerate(slots):
        if not slot.phrase_end or slot.is_rest:
            continue
        if cadence_runs and cadence_runs[-1][1] == index - 1:
            cadence_runs[-1] = (cadence_runs[-1][0], index)
        else:
            cadence_runs.append((index, index))
    phrase_end_indices = {i for start, stop in cadence_runs for i in range(start, stop + 1)}
    cadence_onsets = {start for start, _ in cadence_runs}
    last_index = len(slots) - 1

    for slot_index, slot in enumerate(slots):
        if slot.is_rest:
            continue
        for pitch, steps, strength in slot.events:
            rel = key.to_relative(pitch % 12)
            weight = max(0.35, steps / max(1, slot.stop - slot.start))
            if strength >= 2:
                weight *= 1.5
            for chord_index, pcs in enumerate(members):
                if rel in pcs:
                    scores[slot_index, chord_index] += config.chord_tone_bonus * weight
                elif any((rel - pitch_class) % 12 in (1, 11) for pitch_class in pcs):
                    scores[slot_index, chord_index] += NEIGHBOUR_PENALTY[strength] * weight
                else:
                    scores[slot_index, chord_index] += FOREIGN_PENALTY[strength] * weight

    weak_slots = [i for i, slot in enumerate(slots) if slot.metric_level < 2 and not slot.is_rest]
    for chord_index, chord in enumerate(vocab):
        if is_six_four(chord):
            for slot_index in weak_slots:
                scores[slot_index, chord_index] += config.weak_beat_64_penalty
        root_position_tonic = chord.relative_root == 0 and chord.inversion == 0 and not chord.is_seventh
        root_position_dominant = chord.relative_root == 7 and chord.inversion == 0
        is_dominant = dominant_target(chord) == 0

        for slot_index in phrase_end_indices:
            if slot_index != last_index and slots[slot_index].strength < 2:
                continue
            if root_position_tonic:
                scores[slot_index, chord_index] += config.cadence_tonic_bonus
            elif root_position_dominant:
                scores[slot_index, chord_index] += config.cadence_dominant_bonus
            if chord.inversion > 0:
                scores[slot_index, chord_index] += config.cadence_inversion_penalty
            if chord.inversion == 2:
                scores[slot_index, chord_index] -= 1.6
            if chord.applied_to is not None:
                scores[slot_index, chord_index] -= 1.0
            if slot_index in cadence_onsets and slot_index > 0:
                if is_dominant:
                    scores[slot_index - 1, chord_index] += config.pre_cadence_dominant_bonus
                    if root_position_dominant:
                        scores[slot_index - 1, chord_index] += config.root_position_dominant_bonus
                    elif chord.relative_root == 11:
                        # A leading-tone chord before the tonic yields an
                        # imperfect cadence where Bach writes a perfect one.
                        scores[slot_index - 1, chord_index] -= config.root_position_dominant_bonus

        if slots and root_position_tonic:
            scores[0, chord_index] += OPENING_TONIC_BONUS
            scores[last_index, chord_index] += FINAL_TONIC_BONUS
        if chord.relative_root == 0 and chord.quality == "maj" and key.mode == "minor" and chord.inversion == 0:
            scores[last_index, chord_index] += FINAL_TONIC_BONUS - 1.0  # Picardy third
        if slots and chord.applied_to is not None:
            scores[0, chord_index] -= 1.5

    for slot_index, slot in enumerate(slots):
        if slot.is_rest:
            scores[slot_index, :] = 0.0
    return scores


def chord_max_marginals(emission: np.ndarray, transition: np.ndarray) -> np.ndarray:
    """Best total path score constrained to pass through each (slot, chord).

    Max-plus forward-backward. Using max-marginals rather than the single best
    path keeps genuinely competitive alternatives alive for the voicing search,
    so voice leading can overrule a marginally better chord.
    """
    n_slots, n_chords = emission.shape
    forward = np.zeros_like(emission)
    backward = np.zeros_like(emission)
    forward[0] = emission[0]
    for t in range(1, n_slots):
        forward[t] = emission[t] + (forward[t - 1][:, None] + transition).max(axis=0)
    backward[-1] = 0.0
    for t in range(n_slots - 2, -1, -1):
        backward[t] = (transition + (emission[t + 1] + backward[t + 1])[None, :]).max(axis=1)
    return forward + backward


# ---------------------------------------------------------------------------
# Voicing
# ---------------------------------------------------------------------------


def _pitches_in_range(pitch_classes: Sequence[int], low: int, high: int) -> list[int]:
    out = []
    for pitch_class in pitch_classes:
        pitch = low + ((pitch_class - low) % 12)
        while pitch <= high:
            out.append(pitch)
            pitch += 12
    return sorted(out)


def enumerate_voicings(
    chord: ChordLabel,
    key: Key,
    soprano: int,
    soprano_floor: int,
    *,
    relaxed: bool = False,
) -> list[tuple[tuple[int, int, int, int], float]]:
    """Every acceptable SATB voicing of `chord` under a fixed soprano.

    Returns ((S, A, T, B), static_cost). Voice crossing, illegal spacing, a
    doubled leading tone and a doubled seventh are excluded by construction, so
    the search can never even consider them.
    """
    absolute = chord.absolute_pitch_classes(key)
    root_pc, third_pc = absolute[0], absolute[1]
    fifth_pc = absolute[2] if len(absolute) > 2 else None
    seventh_pc = absolute[3] if len(absolute) > 3 else None
    bass_pc = absolute[chord.inversion]
    chord_pcs = set(absolute)

    lt = None
    target = dominant_target(chord)
    if target is not None:
        lt = key.to_absolute((target + 11) % 12)

    bass_low, bass_high = VOICE_RANGES[BASS]
    alto_low, alto_high = VOICE_RANGES[ALTO]
    tenor_low, tenor_high = VOICE_RANGES[TENOR]
    if relaxed:
        bass_low, bass_high = bass_low - 2, bass_high + 2
        alto_low, alto_high = alto_low - 2, alto_high + 2
        tenor_low, tenor_high = tenor_low - 2, tenor_high + 2

    alto_ceiling = min(soprano_floor, soprano)
    max_sa = MAX_SPACING[(SOPRANO, ALTO)] + (3 if relaxed else 0)
    max_at = MAX_SPACING[(ALTO, TENOR)] + (3 if relaxed else 0)
    max_tb = MAX_SPACING[(TENOR, BASS)] + (3 if relaxed else 0)

    soprano_pc = soprano % 12
    out: list[tuple[tuple[int, int, int, int], float]] = []

    for bass in _pitches_in_range([bass_pc], bass_low, bass_high):
        for tenor in _pitches_in_range(sorted(chord_pcs), max(tenor_low, bass), min(tenor_high, bass + max_tb)):
            if tenor < bass:
                continue
            for alto in _pitches_in_range(
                sorted(chord_pcs), max(alto_low, tenor), min(alto_high, alto_ceiling, tenor + max_at)
            ):
                if alto < tenor or soprano - alto > max_sa or alto > alto_ceiling:
                    continue

                voiced = [bass % 12, tenor % 12, alto % 12]
                present = set(voiced)
                if soprano_pc in chord_pcs:
                    present.add(soprano_pc)
                if root_pc not in present or third_pc not in present:
                    continue
                if seventh_pc is not None and seventh_pc not in present:
                    continue

                counts: dict[int, int] = {}
                for pitch_class in voiced:
                    counts[pitch_class] = counts.get(pitch_class, 0) + 1
                if soprano_pc in chord_pcs:
                    counts[soprano_pc] = counts.get(soprano_pc, 0) + 1
                if lt is not None and counts.get(lt, 0) > 1:
                    continue
                if seventh_pc is not None and counts.get(seventh_pc, 0) > 1:
                    continue

                cost = 0.0
                for pitch_class, count in counts.items():
                    if count < 2:
                        continue
                    extra = count - 1
                    if pitch_class == third_pc and pitch_class != root_pc:
                        cost += extra * (
                            DOUBLE_THIRD_MAJOR_COST if chord.quality in ("maj", "dom7", "maj7", "aug")
                            else DOUBLE_THIRD_MINOR_COST
                        )
                    elif pitch_class == fifth_pc:
                        cost += extra * DOUBLE_FIFTH_COST
                if fifth_pc is not None and fifth_pc not in present:
                    cost += MISSING_FIFTH_COST
                for voice, pitch in ((ALTO, alto), (TENOR, tenor), (BASS, bass)):
                    low, high = PREFERRED_RANGES[voice]
                    centre = (low + high) / 2
                    cost += TESSITURA_COST * abs(pitch - centre)
                    if pitch < low:
                        cost += 0.35 * (low - pitch)
                    elif pitch > high:
                        cost += 0.35 * (pitch - high)
                # Keep the upper three voices reasonably compact.
                cost += 0.03 * max(0, (soprano - tenor) - 14)
                out.append(((soprano, alto, tenor, bass), cost))

    out.sort(key=lambda item: item[1])
    return out[:MAX_VOICINGS_PER_CHORD]


_PAIRS = ((SOPRANO, ALTO), (SOPRANO, TENOR), (SOPRANO, BASS), (ALTO, TENOR), (ALTO, BASS), (TENOR, BASS))


def voice_leading_costs(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    """Pairwise voice-leading cost between two sets of voicings.

    `prev` is (Np, 4) and `curr` is (Nc, 4), both (S, A, T, B). Fully vectorised:
    the search evaluates a few hundred thousand voicing pairs per harmonization.
    """
    np_, nc = prev.shape[0], curr.shape[0]
    cost = np.zeros((np_, nc), dtype=np.float32)
    motion = curr[None, :, :].astype(np.int32) - prev[:, None, :].astype(np.int32)  # (Np, Nc, 4)
    moved = motion != 0

    for high, low in _PAIRS:
        p_hi, p_lo = prev[:, high].astype(np.int32), prev[:, low].astype(np.int32)
        c_hi, c_lo = curr[:, high].astype(np.int32), curr[:, low].astype(np.int32)
        prev_gap = np.abs(p_hi - p_lo)[:, None]
        curr_gap = np.abs(c_hi[None, :] - c_lo[None, :])
        both_move = moved[:, :, high] & moved[:, :, low]

        was_fifth = (prev_gap % 12 == 7)
        now_fifth = (curr_gap % 12 == 7)
        was_octave = (prev_gap % 12 == 0)
        now_octave = (curr_gap % 12 == 0)

        same_direction = (motion[:, :, high] > 0) == (motion[:, :, low] > 0)
        perfect_to_perfect = both_move & ((was_fifth & now_fifth) | (was_octave & now_octave))
        cost += np.where(perfect_to_perfect & same_direction, PARALLEL_PERFECT_COST, 0.0)
        cost += np.where(perfect_to_perfect & ~same_direction, CONTRARY_PERFECT_COST, 0.0)

    # Direct fifths/octaves in the outer voices, with the soprano leaping.
    s_motion, b_motion = motion[:, :, SOPRANO], motion[:, :, BASS]
    similar = (s_motion > 0) == (b_motion > 0)
    similar &= (s_motion != 0) & (b_motion != 0)
    soprano_leaps = np.abs(s_motion) > 2
    outer_gap = np.abs(curr[None, :, SOPRANO].astype(np.int32) - curr[None, :, BASS].astype(np.int32))
    arrives_perfect = (outer_gap % 12 == 7) | (outer_gap % 12 == 0)
    cost += np.where(similar & soprano_leaps & arrives_perfect, DIRECT_PERFECT_COST, 0.0)

    # Voice overlap: a voice moving past where its neighbour just was.
    for upper, lower in ((SOPRANO, ALTO), (ALTO, TENOR), (TENOR, BASS)):
        cost += np.where(curr[None, :, lower].astype(np.int32) > prev[:, None, upper].astype(np.int32), OVERLAP_COST, 0.0)
        cost += np.where(curr[None, :, upper].astype(np.int32) < prev[:, None, lower].astype(np.int32), OVERLAP_COST, 0.0)

    # Melodic behaviour of the generated voices.
    for voice in (ALTO, TENOR, BASS):
        delta = np.abs(motion[:, :, voice])
        per_semitone = BASS_LEAP_COST if voice == BASS else UPPER_LEAP_COST
        cost += np.where(delta > 2, per_semitone * (delta - 2), 0.0)
        cost += np.where(np.isin(delta, (6, 10, 11, 13, 14)), AWKWARD_MELODIC_COST, 0.0)
        cost += np.where(delta > 12, OVER_OCTAVE_LEAP_COST, 0.0)
        cost += np.where(delta == 0, COMMON_TONE_BONUS, 0.0)

    contrary_bass = ((s_motion > 0) != (b_motion > 0)) & (s_motion != 0) & (b_motion != 0)
    cost += np.where(contrary_bass, CONTRARY_BASS_BONUS, 0.0)

    directions = np.sign(motion)
    all_same = (directions[:, :, SOPRANO] != 0)
    for voice in (ALTO, TENOR, BASS):
        all_same &= directions[:, :, voice] == directions[:, :, SOPRANO]
    cost += np.where(all_same, ALL_SIMILAR_COST, 0.0)

    return cost


def tendency_costs(
    prev: np.ndarray,
    curr: np.ndarray,
    prev_chord: ChordLabel,
    curr_chord: ChordLabel,
    key: Key,
) -> np.ndarray:
    """Extra cost for abandoning a leading tone or a chordal seventh."""
    np_, nc = prev.shape[0], curr.shape[0]
    cost = np.zeros((np_, nc), dtype=np.float32)
    motion = curr[None, :, :].astype(np.int32) - prev[:, None, :].astype(np.int32)

    target = dominant_target(prev_chord)
    if target is not None and curr_chord.relative_root == target and prev_chord.key() != curr_chord.key():
        lt_abs = key.to_absolute((target + 11) % 12)
        tonic_abs = key.to_absolute(target)
        fifth_abs = key.to_absolute((target + 7) % 12)
        for voice in range(4):
            holds_lt = (prev[:, voice] % 12 == lt_abs)[:, None]
            resolves = (motion[:, :, voice] == 1) & (curr[None, :, voice] % 12 == tonic_abs)
            if voice in (ALTO, TENOR):
                frustrated = (curr[None, :, voice] % 12 == fifth_abs) & (np.abs(motion[:, :, voice]) <= 4)
                cost += np.where(holds_lt & ~resolves & frustrated, FRUSTRATED_LT_COST, 0.0)
                cost += np.where(holds_lt & ~resolves & ~frustrated, UNRESOLVED_LT_COST, 0.0)
            else:
                cost += np.where(holds_lt & ~resolves, UNRESOLVED_LT_COST, 0.0)

    seventh = prev_chord.seventh_relative_pc
    if seventh is not None and prev_chord.key() != curr_chord.key():
        seventh_abs = key.to_absolute(seventh)
        carried = curr_chord.seventh_relative_pc
        carried_abs = key.to_absolute(carried) if carried is not None else None
        for voice in range(4):
            holds = (prev[:, voice] % 12 == seventh_abs)[:, None]
            falls = (motion[:, :, voice] == -1) | (motion[:, :, voice] == -2)
            if carried_abs is not None:
                falls = falls | ((motion[:, :, voice] == 0) & (curr[None, :, voice] % 12 == carried_abs))
            cost += np.where(holds & ~falls, UNRESOLVED_SEVENTH_COST, 0.0)

    return cost


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


@dataclass
class _State:
    chord_index: int
    voicing: tuple[int, int, int, int]


class RuleHarmonyEngine(HarmonyEngine):
    """Functional harmony + voice-leading search. No learned parameters."""

    id = "rules"
    name = "Functional Harmony (rules)"
    description = (
        "Viterbi chord search over a full functional vocabulary — sevenths, inversions, "
        "secondary dominants, mixture — followed by a voice-leading Viterbi that voices "
        "actual SATB parts. Deterministic."
    )
    learned = False

    def __init__(
        self,
        *,
        config: RuleConfig = DEFAULT_CONFIG,
        max_chord_candidates: int = MAX_CHORD_CANDIDATES,
    ) -> None:
        self.config = config
        self.max_chord_candidates = max_chord_candidates
        self._vocab_cache: dict[str, tuple[list[ChordLabel], np.ndarray]] = {}

    def _vocabulary(self, mode: str) -> tuple[list[ChordLabel], np.ndarray]:
        if mode not in self._vocab_cache:
            vocab = build_vocabulary(mode)
            self._vocab_cache[mode] = (vocab, build_transition_matrix(vocab, mode))
        return self._vocab_cache[mode]

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        grid = melody_to_grid(melody)
        if melody.key is not None:
            key, confidence = Key(melody.key.tonic, melody.key.mode), melody.key.confidence or 1.0
        else:
            key, confidence = detect_melody_key(grid)

        if grid.length == 0:
            empty, names = select_voices([[], [], [], []], voice_count)
            return Harmonization(
                key=KeySignature(tonic=key.tonic, mode=key.mode, confidence=confidence),
                voices=grid_to_voices(empty, names=names),
            )

        slots = build_slots(grid)
        plan = self._plan(slots, key)
        lines = self._expand(plan, slots, grid)
        chords = self._chord_list(plan, slots, key)
        violations = self._violations(lines, plan, slots, grid, key)

        selected, names = select_voices(lines, voice_count)
        voices = grid_to_voices(selected, names=names)
        if voice_count > 4:
            voices = voices + self._extra_voices(lines, voice_count - 4)
        return Harmonization(
            key=KeySignature(tonic=key.tonic, mode=key.mode, confidence=confidence),
            voices=voices,
            chords=chords,
            violations=violations,
        )

    # -- search ------------------------------------------------------------

    def _plan(self, slots: Sequence[Slot], key: Key) -> list[_State | None]:
        vocab, transition = self._vocabulary(key.mode)
        emission = emission_scores(slots, vocab, key, self.config)
        marginals = chord_max_marginals(emission, transition)

        candidates: list[list[int]] = []
        for slot_index, slot in enumerate(slots):
            if slot.is_rest:
                candidates.append([])
                continue
            order = np.argsort(-marginals[slot_index])[: self.max_chord_candidates]
            candidates.append([int(i) for i in order])

        states: list[list[_State]] = []
        state_costs: list[np.ndarray] = []
        for slot_index, slot in enumerate(slots):
            slot_states: list[_State] = []
            costs: list[float] = []
            if not slot.is_rest:
                best = marginals[slot_index][candidates[slot_index]].max() if candidates[slot_index] else 0.0
                for chord_index in candidates[slot_index]:
                    chord = vocab[chord_index]
                    options = enumerate_voicings(chord, key, slot.principal, slot.lowest)
                    if not options:
                        options = enumerate_voicings(chord, key, slot.principal, slot.principal)
                    if not options:
                        options = enumerate_voicings(chord, key, slot.principal, slot.principal, relaxed=True)
                    harmonic_cost = self.config.harmony_weight * (best - marginals[slot_index][chord_index])
                    for voicing, static_cost in options:
                        slot_states.append(_State(chord_index, voicing))
                        costs.append(static_cost + harmonic_cost)
            states.append(slot_states)
            state_costs.append(np.array(costs, dtype=np.float32) if costs else np.zeros(0, dtype=np.float32))

        return self._viterbi(states, state_costs, slots, vocab, key)

    def _viterbi(
        self,
        states: list[list[_State]],
        state_costs: list[np.ndarray],
        slots: Sequence[Slot],
        vocab: Sequence[ChordLabel],
        key: Key,
    ) -> list[_State | None]:
        n = len(slots)
        dp: list[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in range(n)]
        back: list[np.ndarray] = [np.zeros(0, dtype=np.int32) for _ in range(n)]
        # Two views of each state: `incoming` has the slot's first melody pitch
        # in the soprano, `outgoing` its last. A voice-leading rule applies to
        # the motion that actually happens across the barline, which is from the
        # END of one slot's melody to the START of the next — comparing the two
        # slots' initial pitches silently misses every parallel that a melodic
        # passing tone creates.
        incoming = [
            np.array([s.voicing for s in slot_states], dtype=np.int32) if slot_states else np.zeros((0, 4), np.int32)
            for slot_states in states
        ]
        outgoing = []
        for slot, array in zip(slots, incoming):
            moved = array.copy()
            if len(moved) and slot.final is not None:
                moved[:, SOPRANO] = slot.final
            outgoing.append(moved)

        previous_index: int | None = None
        for index in range(n):
            if not states[index]:
                continue
            if previous_index is None:
                dp[index] = state_costs[index].copy()
                back[index] = np.full(len(states[index]), -1, dtype=np.int32)
                previous_index = index
                continue

            gap = slots[index].start - slots[previous_index].start
            decay = 1.0 if gap <= slots[index].stop - slots[index].start else 0.4
            cost = voice_leading_costs(outgoing[previous_index], incoming[index]) * decay

            prev_chords = np.array([s.chord_index for s in states[previous_index]])
            curr_chords = np.array([s.chord_index for s in states[index]])
            for prev_chord in np.unique(prev_chords):
                prev_mask = prev_chords == prev_chord
                for curr_chord in np.unique(curr_chords):
                    curr_mask = curr_chords == curr_chord
                    block = tendency_costs(
                        outgoing[previous_index][prev_mask], incoming[index][curr_mask],
                        vocab[int(prev_chord)], vocab[int(curr_chord)], key,
                    )
                    cost[np.ix_(prev_mask, curr_mask)] += block * decay

            total = dp[previous_index][:, None] + cost
            best_prev = np.argmin(total, axis=0)
            dp[index] = total[best_prev, np.arange(total.shape[1])] + state_costs[index]
            back[index] = best_prev.astype(np.int32)
            previous_index = index

        plan: list[_State | None] = [None] * n
        filled = [i for i in range(n) if states[i]]
        if not filled:
            return plan
        cursor = filled[-1]
        choice = int(np.argmin(dp[cursor]))
        for position in reversed(filled):
            plan[position] = states[position][choice]
            choice = int(back[position][choice]) if back[position][choice] >= 0 else 0
        return plan

    # -- output ------------------------------------------------------------

    def _expand(self, plan: Sequence[_State | None], slots: Sequence[Slot], grid: MelodyGrid) -> list[list[int]]:
        lines = [[REST] * grid.length for _ in range(4)]
        lines[SOPRANO] = list(grid.pitches)
        for state, slot in zip(plan, slots):
            if state is None:
                continue
            for t in range(slot.start, slot.stop):
                for voice in (ALTO, TENOR, BASS):
                    lines[voice][t] = state.voicing[voice]
        return lines

    def _chord_list(self, plan: Sequence[_State | None], slots: Sequence[Slot], key: Key) -> list[Chord]:
        vocab, _ = self._vocabulary(key.mode)
        out: list[Chord] = []
        for state, slot in zip(plan, slots):
            if state is None:
                continue
            label = vocab[state.chord_index]
            start = round(slot.start * STEP, 6)
            duration = round((slot.stop - slot.start) * STEP, 6)
            if out and out[-1].roman == label.roman(key.mode) and math.isclose(out[-1].start + out[-1].duration, start):
                out[-1] = out[-1].model_copy(update={"duration": round(out[-1].duration + duration, 6)})
                continue
            out.append(Chord(
                start=start,
                duration=duration,
                roman=label.roman(key.mode),
                root=label.absolute_root(key),
                quality=label.contract_quality(),
                inversion=label.inversion,
                secondaryOf=None if label.applied_to is None else key.to_absolute(label.applied_to),
                # Response-side fields are always populated, never left to a
                # default, so the UI never has to null-check. These engines write
                # common-practice chorale harmony: no upper extensions, and no
                # reharmonization, so there is no substitution to explain.
                extensions=[],
                substitutionOf=None,
                substitutionKind=None,
            ))
        return out

    def _violations(
        self,
        lines: Sequence[Sequence[int]],
        plan: Sequence[_State | None],
        slots: Sequence[Slot],
        grid: MelodyGrid,
        key: Key,
    ) -> list[Violation]:
        vocab, _ = self._vocabulary(key.mode)
        texture = texture_from_voices([[None if p == REST else p for p in line] for line in lines], step=STEP)
        per_step: list[ChordLabel | None] = [None] * grid.length
        for state, slot in zip(plan, slots):
            if state is None:
                continue
            for t in range(slot.start, slot.stop):
                per_step[t] = vocab[state.chord_index]
        return [
            Violation(
                kind=defect.kind,
                severity=defect.severity,
                start=defect.offset,
                voices=[VOICE_NAMES[v] for v in defect.voices if v < 4],
                message=defect.message,
            )
            for defect in analyze_texture(texture, key, per_step)
            if defect.severity != "info"
        ]

    def _extra_voices(self, lines: Sequence[Sequence[int]], count: int):
        """Octave doublings for voice counts above four."""
        from contracts.schema import Note, Voice

        extras = []
        for index in range(count):
            source = lines[BASS] if index % 2 == 0 else lines[SOPRANO]
            shift = -12 if index % 2 == 0 else 12
            doubled = [REST if p == REST else p + shift for p in source]
            voice = grid_to_voices([doubled])[0]
            extras.append(Voice(name="bass" if index % 2 == 0 else "soprano", notes=voice.notes))
        return extras


register(RuleHarmonyEngine())
