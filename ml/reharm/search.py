"""The lattice, and the two ways of walking it.

Both engines are built here, on purpose, out of the same parts. They share the
candidate space from `substitutions.py` and the hard constraints baked into it,
and differ in exactly one thing:

  * the **rule** engine takes the Viterbi argmax of a hand-written score;
  * the **stochastic** engine samples from a learned distribution over the same
    lattice, at a temperature.

Isolating the difference to that single axis is what makes the comparison in
REPORT.md mean anything. If the sampler wins it is because sampling wins, not
because it was given a bigger vocabulary or a weaker opponent.

Sampling is done by forward-filtering backward-sampling, not by greedy
left-to-right choice. The distinction is not academic. A greedy sampler picks a
tritone substitute in bar 3 and only discovers in bar 4 that nothing can follow
it, so it must either backtrack or emit something incoherent — the classic
failure of sampled harmony. FFBS computes exact backward messages first, so
every sample it draws is a sample from the *whole-sequence* distribution and is
coherent by construction. It is also the reason temperature behaves like a
musical dial rather than a noise knob: at T -> 0 it converges on the argmax
path, and at T = 1 it draws honestly from the model.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Protocol, Sequence

import numpy as np

from .chords import JazzChord, absorb_melody, is_dominant_resolution, resolves_down_semitone
from .data import ChordSpan, Progression
from .model import BOS, ChordNGram, Token, token_of
from .skeleton import Skeleton, Unit
from .substitutions import Candidate, Context, generate

NEG = -1e9


@dataclass(frozen=True)
class ReharmConfig:
    """Everything tunable, in one object, with the defaults stated once.

    `adventure` is the user-facing dial: it scales how much a substitution has
    to earn its place. At 0 the engine barely moves off the skeleton; at 1 it
    substitutes wherever the harmony and the melody allow.
    """

    adventure: float = 0.75
    substitution_cost: float = 1.4
    split_cost: float = 0.7
    melody_weight: float = 4.0
    resolution_bonus: float = 1.2
    unresolved_dominant_cost: float = 0.9
    broken_tritone_cost: float = 4.0
    ii_v_bonus: float = 0.6
    repeat_cost: float = 0.5
    cadence_bonus: float = 1.5
    #: How hard the result is pulled back toward the skeleton. This is not a
    #: scoring preference, it is the definition of the task: a chord language
    #: model asked for the most jazz-like continuation will happily write a
    #: different, perfectly idiomatic tune. Both engines pay it, so the
    #: comparison between them stays about search versus sampling.
    anchor_weight: float = 2.2
    #: Cost of substituting in two consecutive units. Reharmonization is a
    #: seasoning; a substitution in every bar reads as a modulation.
    consecutive_cost: float = 0.8
    model_weight: float = 1.0
    temperature: float = 1.0
    top_p: float = 0.95
    allow_coltrane: bool = False

    def effective_substitution_cost(self) -> float:
        return self.substitution_cost * (1.6 - 1.5 * max(0.0, min(1.0, self.adventure)))

    def effective_anchor(self) -> float:
        return self.anchor_weight * (1.4 - 1.2 * max(0.0, min(1.0, self.adventure)))


# ---------------------------------------------------------------------------
# Lattice
# ---------------------------------------------------------------------------


@dataclass
class Lattice:
    units: list[Unit]
    candidates: list[list[Candidate]]
    tonic: int
    mode: str

    def __len__(self) -> int:
        return len(self.units)


def build_lattice(skeleton: Skeleton, config: ReharmConfig = ReharmConfig()) -> Lattice:
    units = skeleton.units
    candidates: list[list[Candidate]] = []
    for index, unit in enumerate(units):
        context = Context(
            tonic=skeleton.tonic,
            mode=skeleton.mode,
            previous=units[index - 1].base if index else None,
            following=units[index + 1].base if index + 1 < len(units) else None,
            following2=units[index + 2].base if index + 2 < len(units) else None,
            is_last=index == len(units) - 1,
            allow_coltrane=config.allow_coltrane,
        )
        candidates.append(generate(unit, context))
    return Lattice(units=list(units), candidates=candidates, tonic=skeleton.tonic, mode=skeleton.mode)


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


class Scorer(Protocol):
    def emission(self, index: int, candidate: Candidate) -> float: ...

    def transition(self, index: int, previous: Candidate, candidate: Candidate) -> float: ...


#: Hand-written root-motion preferences, semitones from one root to the next.
#: Down a fifth first, then the semitone motions that the substitutions in this
#: package actually produce; standing still is discouraged.
ROOT_MOTION = {
    0: -0.45, 1: 0.15, 2: 0.25, 3: 0.0, 4: -0.1, 5: 0.55,
    6: -0.2, 7: 0.1, 8: 0.05, 9: 0.2, 10: 0.2, 11: 0.35,
}


@dataclass
class RuleScorer:
    """Hand-written functional scoring. No learned parameters, by design."""

    lattice: Lattice
    config: ReharmConfig = ReharmConfig()

    def emission(self, index: int, candidate: Candidate) -> float:
        unit = self.lattice.units[index]
        score = candidate.bonus
        score -= self.config.melody_weight * candidate.melody_penalty
        score -= self.config.effective_anchor() * anchor_distance(candidate, unit, self.lattice)
        if candidate.kind is not None and candidate.kind != "extension":
            score -= self.config.effective_substitution_cost()
        if len(candidate.chords) > 1:
            score -= self.config.split_cost
        score += self._internal(candidate)
        if unit.is_last:
            score += self._cadence(candidate)
        return score

    def transition(self, index: int, previous: Candidate, candidate: Candidate) -> float:
        return self._pair(previous.last, candidate.first) - consecutive_penalty(
            previous, candidate, self.config
        )

    def _internal(self, candidate: Candidate) -> float:
        return sum(
            self._pair(candidate.chords[i], candidate.chords[i + 1])
            for i in range(len(candidate.chords) - 1)
        )

    def _pair(self, previous: JazzChord, chord: JazzChord) -> float:
        score = ROOT_MOTION[(chord.root - previous.root) % 12]
        if previous.same_harmony(chord):
            score -= self.config.repeat_cost
        if previous.is_dominant:
            if is_dominant_resolution(previous, chord):
                score += self.config.resolution_bonus
            else:
                score -= self.config.unresolved_dominant_cost
            # A tritone substitute exists to move the bass down a semitone. If
            # it does not do that, it is not a substitution, it is a mistake.
            if previous.substitution_kind == "tritone" and not resolves_down_semitone(previous, chord):
                score -= self.config.broken_tritone_cost
        if previous.quality in ("min7", "halfdim7") and chord.is_dominant:
            if (chord.root - previous.root) % 12 == 5:
                score += self.config.ii_v_bonus
        return score

    def _cadence(self, candidate: Candidate) -> float:
        degree = (candidate.last.root - self.lattice.tonic) % 12
        third = 4 if self.lattice.mode == "major" else 3
        if degree == 0 and (candidate.last.root + third) % 12 in candidate.last.core_pcs:
            return self.config.cadence_bonus
        if degree == 0:
            return self.config.cadence_bonus * 0.6
        return 0.0


@dataclass
class ModelScorer:
    """Learned scoring: a jazz chord language model plus an anchor to the tune.

    The anchor is not a hack around a weak model, it is the definition of the
    task. A chord language model asked for the most jazz-like continuation will
    cheerfully write a *different tune*; reharmonization means staying attached
    to the one you were given. So the objective is "likely under real jazz
    harmony AND recognisably still this progression", and the balance between
    those is the `adventure` dial.
    """

    lattice: Lattice
    model: ChordNGram
    config: ReharmConfig = ReharmConfig()

    def emission(self, index: int, candidate: Candidate) -> float:
        unit = self.lattice.units[index]
        score = -self.config.melody_weight * candidate.melody_penalty
        score -= self.config.effective_anchor() * anchor_distance(candidate, unit, self.lattice)
        if len(candidate.chords) > 1:
            score -= self.config.split_cost * 0.5
        if candidate.kind is not None and candidate.kind != "extension":
            score -= self.config.effective_substitution_cost() * 0.35
        history = [BOS, BOS] if index == 0 else []
        for position in range(1, len(candidate.chords)):
            context = self._history(candidate.chords[:position], history)
            score += self.config.model_weight * self.model.log_probability(
                self._token(candidate.chords[position]), context
            )
        if index == 0:
            score += self.config.model_weight * self.model.log_probability(
                self._token(candidate.chords[0]), [BOS, BOS]
            )
        if unit.is_last:
            score += self._cadence(candidate)
        return score

    def transition(self, index: int, previous: Candidate, candidate: Candidate) -> float:
        history = [self._token(chord) for chord in previous.chords[-2:]]
        score = self.config.model_weight * self.model.log_probability(
            self._token(candidate.first), history
        )
        return score - consecutive_penalty(previous, candidate, self.config)

    def _token(self, chord: JazzChord) -> Token:
        return token_of(chord, self.lattice.tonic)

    def _history(self, chords: Sequence[JazzChord], prefix: Sequence[Token]) -> list[Token]:
        tokens = [*prefix, *(self._token(chord) for chord in chords)]
        return tokens[-2:]

    def _cadence(self, candidate: Candidate) -> float:
        degree = (candidate.last.root - self.lattice.tonic) % 12
        return self.config.cadence_bonus if degree == 0 else 0.0


@dataclass
class HybridScorer:
    """Learned syntax, hand-written appetite for colour, sampled for variety.

    The measurement that motivates this: on real jazz melodies the learned
    model produces the LOWEST style divergence of anything tested — it is more
    typical of the corpus than the human reference is — and the LOWEST
    chromaticism. That is not a bug in the model, it is what a model of the
    average of 1170 standards is *for*. The average of jazz is a diatonic
    ii-V-I.

    So the two halves are complementary rather than competing: the model knows
    what follows what, and the hand-written bonuses know that the point of a
    reharmonization is to be interesting. Sampling then supplies the one thing
    neither of them can, which is a different answer every time.
    """

    lattice: Lattice
    model: ChordNGram
    config: ReharmConfig = ReharmConfig()
    rule_weight: float = 0.9

    def __post_init__(self) -> None:
        self._model = ModelScorer(self.lattice, self.model, self.config)
        self._rules = RuleScorer(self.lattice, self.config)

    def emission(self, index: int, candidate: Candidate) -> float:
        learned = self._model.emission(index, candidate)
        unit = self.lattice.units[index]
        # Only the rule engine's *appetite* is borrowed — the candidate bonus
        # and its internal shape — not its anchor or melody terms, which the
        # learned scorer has already applied. Double-counting them would just
        # be the rule engine with extra steps.
        colour = candidate.bonus - (0.0 if candidate.kind is None else 0.0)
        if len(candidate.chords) > 1:
            colour += self._rules._internal(candidate)
        if unit.is_last:
            colour += self._rules._cadence(candidate)
        return learned + self.rule_weight * colour

    def transition(self, index: int, previous: Candidate, candidate: Candidate) -> float:
        learned = self._model.transition(index, previous, candidate)
        return learned + self.rule_weight * self._rules._pair(previous.last, candidate.first)


def _is_tonic_function(root: int, quality: str, lattice: Lattice) -> bool:
    degree = (root - lattice.tonic) % 12
    if lattice.mode == "major":
        return degree == 0 or (degree == 9 and quality in ("min", "min7", "min6"))
    return degree == 0 or (degree == 3 and quality in ("maj", "maj7", "maj6"))


def anchor_distance(candidate: Candidate, unit: Unit, lattice: Lattice) -> float:
    """How far a candidate sits from the chord it replaces.

    Pitch-class overlap with a root-identity term: a maj -> maj7 upgrade is
    nearly free, a relative substitution is cheap, an unrelated root is not.

    The extra clause protects the tonic. A tune is identified by where it comes
    to rest, and a reharmonizer that substitutes the tonic on a downbeat is not
    reharmonizing the tune, it is writing over it. This lives in the shared
    anchor rather than in either scorer because it belongs to the definition of
    the task, not to one strategy for solving it.
    """
    base_pcs = set(unit.base.core_pcs)
    worst = 0.0
    for chord in candidate.chords:
        pcs = set(chord.core_pcs)
        union = base_pcs | pcs
        overlap = len(base_pcs & pcs) / len(union) if union else 1.0
        distance = 1.0 - overlap
        if chord.root != unit.base.root:
            distance += 0.5
        worst = max(worst, distance)
    base_is_tonic = _is_tonic_function(unit.base.root, unit.base.quality, lattice)
    head_is_tonic = _is_tonic_function(candidate.first.root, candidate.first.quality, lattice)
    if base_is_tonic and not head_is_tonic:
        worst += 0.6 if unit.metric_level >= 3 else 0.35
    return worst


def consecutive_penalty(previous: Candidate, candidate: Candidate, config: ReharmConfig) -> float:
    """Cost of substituting in two units in a row."""
    substantive = {None, "extension"}
    if previous.kind in substantive or candidate.kind in substantive:
        return 0.0
    return config.consecutive_cost


# ---------------------------------------------------------------------------
# Walking the lattice
# ---------------------------------------------------------------------------


def _score_arrays(lattice: Lattice, scorer: Scorer) -> tuple[list[np.ndarray], list[np.ndarray]]:
    emissions = [
        np.array([scorer.emission(index, candidate) for candidate in candidates], dtype=np.float64)
        for index, candidates in enumerate(lattice.candidates)
    ]
    transitions: list[np.ndarray] = []
    for index in range(1, len(lattice.candidates)):
        previous = lattice.candidates[index - 1]
        current = lattice.candidates[index]
        matrix = np.array(
            [[scorer.transition(index, a, b) for b in current] for a in previous],
            dtype=np.float64,
        )
        transitions.append(matrix)
    return emissions, transitions


def viterbi(lattice: Lattice, scorer: Scorer) -> list[Candidate]:
    """Best path — the deterministic engine."""
    if not lattice.candidates:
        return []
    emissions, transitions = _score_arrays(lattice, scorer)
    best = emissions[0].copy()
    back: list[np.ndarray] = []
    for index in range(1, len(emissions)):
        total = best[:, None] + transitions[index - 1]
        choice = np.argmax(total, axis=0)
        back.append(choice)
        best = total[choice, np.arange(total.shape[1])] + emissions[index]

    path = [int(np.argmax(best))]
    for choice in reversed(back):
        path.append(int(choice[path[-1]]))
    path.reverse()
    return [lattice.candidates[index][choice] for index, choice in enumerate(path)]


def sample(
    lattice: Lattice,
    scorer: Scorer,
    *,
    temperature: float = 0.9,
    top_p: float = 0.92,
    seed: int | None = None,
) -> list[Candidate]:
    """Forward-filtering backward-sampling over the whole lattice.

    Exact sampling from P(path) proportional to exp(score(path) / T), so the
    result is coherent end to end rather than locally plausible and globally
    stuck. `top_p` truncates each conditional to its nucleus, which cuts the
    long tail of chords the model gives 0.1% to without touching the shape of
    the distribution where it matters.
    """
    if not lattice.candidates:
        return []
    if temperature <= 1e-6:
        return viterbi(lattice, scorer)

    rng = random.Random(seed)
    emissions, transitions = _score_arrays(lattice, scorer)
    scaled_emissions = [emission / temperature for emission in emissions]
    scaled_transitions = [transition / temperature for transition in transitions]

    # Backward messages: beta[t][i] = log sum over all completions after t.
    n = len(scaled_emissions)
    beta: list[np.ndarray] = [np.zeros(len(scaled_emissions[-1]))]
    for index in range(n - 1, 0, -1):
        payload = scaled_transitions[index - 1] + (scaled_emissions[index] + beta[0])[None, :]
        beta.insert(0, _logsumexp(payload, axis=1))

    path: list[int] = []
    for index in range(n):
        if index == 0:
            logits = scaled_emissions[0] + beta[0]
        else:
            logits = scaled_transitions[index - 1][path[-1]] + scaled_emissions[index] + beta[index]
        path.append(_sample_from(logits, top_p, rng))
    return [lattice.candidates[index][choice] for index, choice in enumerate(path)]


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    return (peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))).squeeze(axis)


def _sample_from(logits: np.ndarray, top_p: float, rng: random.Random) -> int:
    weights = np.exp(logits - np.max(logits))
    total = weights.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    probabilities = weights / total
    order = np.argsort(-probabilities, kind="stable")
    cumulative = np.cumsum(probabilities[order])
    keep = int(np.searchsorted(cumulative, min(1.0, max(1e-6, top_p))) + 1)
    keep = max(1, min(keep, len(order)))
    chosen = order[:keep]
    renormalized = probabilities[chosen] / probabilities[chosen].sum()
    draw = rng.random()
    running = 0.0
    for index, probability in zip(chosen, renormalized):
        running += probability
        if draw <= running:
            return int(index)
    return int(chosen[-1])


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class Reharmonization:
    """The chosen chords, with provenance, plus the skeleton they came from."""

    spans: list[ChordSpan]
    skeleton: Skeleton
    kinds: list[str] = field(default_factory=list)

    def progression(self, title: str = "reharm") -> Progression:
        return Progression(
            spans=list(self.spans),
            tonic=self.skeleton.tonic,
            mode=self.skeleton.mode,
            meter=self.skeleton.meter,
            title=title,
            source="reharm",
        )

    def summary(self) -> str:
        return " | ".join(
            span.chord.symbol() + (f"[{span.chord.substitution_kind}]" if span.chord.substitution_kind else "")
            for span in self.spans
        )


def realize(lattice: Lattice, path: Sequence[Candidate], skeleton: Skeleton) -> Reharmonization:
    """Turn a chosen path into timed chords, with melody-driven tensions.

    Tensions are added last and only where the melody supplies evidence for
    them: a b9 in the tune becomes a stated b9 in the chord rather than an
    accident the voicing has to dodge. This is the difference between a
    reharmonization that sounds intentional and one that sounds like a wrong
    note.
    """
    spans: list[ChordSpan] = []
    kinds: list[str] = []
    for unit, candidate in zip(lattice.units, path):
        for chord, start, duration in zip(candidate.chords, candidate.starts, candidate.durations):
            weighted = _weights_in(unit, start, duration)
            coloured = absorb_melody(chord, weighted)
            spans.append(ChordSpan(start, duration, coloured))
            if coloured.substitution_kind:
                kinds.append(coloured.substitution_kind)
    return Reharmonization(spans=spans, skeleton=skeleton, kinds=kinds)


def _weights_in(unit: Unit, start: float, duration: float) -> list[tuple[int, float]]:
    from .metrics import note_weight

    stop = start + duration
    totals: dict[int, float] = {}
    for note_start, pitch, note_duration in unit.melody:
        overlap = min(note_start + note_duration, stop) - max(note_start, start)
        if overlap <= 1e-6:
            continue
        totals[pitch % 12] = totals.get(pitch % 12, 0.0) + note_weight(max(note_start, start), overlap)
    return sorted(totals.items())
