"""The evaluation harness: score any registered engine on identical data.

v1's fatal methodological gap was that "the rules beat the model" was a vibe.
There was no held-out protocol, no metric that measured harmonization quality
rather than label agreement, and no calibration for how good "good" is. This
module is the fix, and it is deliberately engine-agnostic: a rule engine and a
neural engine go through the same code path, get the same input, and are scored
by the same analyser.

Protocol
--------
* Held-out test split, chosen by hash of the piece id, never seen in training.
* Each engine receives the soprano alone — exactly what a user uploads.
* The ground-truth key is supplied by default so the comparison isolates the
  harmonic decision; `--detect-key` reruns everything in the realistic setting
  where the engine must find the key itself, and the delta is reported.
* Style distributions are compared against the TRAIN split, so the Bach-oracle
  row shows the natural train/test divergence — the noise floor below which no
  divergence number is meaningful.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from contracts.schema import KeySignature, Melody

from ..data.corpus import REST, STEPS_PER_QUARTER, Chorale
from ..data.melody import chorale_to_melody, detect_melody_key, melody_to_grid, voices_to_grid
from ..engines.base import HarmonyEngine
from ..theory.pitch import Key
from .metrics import (
    AgreementCounts,
    DEFECT_KINDS,
    DefectCounts,
    HARD_DEFECTS,
    StyleCounts,
    collect_agreement,
    collect_defects,
    collect_style,
    js_divergence,
)


@dataclass
class EngineResult:
    engine_id: str
    name: str
    learned: bool
    pieces: int = 0
    failures: int = 0
    defects: DefectCounts = field(default_factory=DefectCounts)
    style: StyleCounts = field(default_factory=StyleCounts)
    agreement: AgreementCounts = field(default_factory=AgreementCounts)
    nll_nats: float = 0.0
    nll_tokens: int = 0
    seconds: float = 0.0

    @property
    def has_likelihood(self) -> bool:
        return self.nll_tokens > 0

    def perplexity(self) -> float | None:
        if not self.has_likelihood:
            return None
        import math

        return math.exp(self.nll_nats / self.nll_tokens)

    def nll_per_token(self) -> float | None:
        if not self.has_likelihood:
            return None
        return self.nll_nats / self.nll_tokens

    def divergences(self, reference: StyleCounts) -> dict[str, float]:
        return {
            "chord_unigram_js": js_divergence(self.style.chord_unigram, reference.chord_unigram),
            "chord_bigram_js": js_divergence(self.style.chord_bigram, reference.chord_bigram),
            "root_motion_js": js_divergence(self.style.root_motion, reference.root_motion),
            "inversion_js": js_divergence(self.style.inversion, reference.inversion),
            "quality_js": js_divergence(self.style.quality, reference.quality),
            "cadence_js": js_divergence(self.style.cadence, reference.cadence),
            "melodic_interval_js": js_divergence(self.style.melodic_interval, reference.melodic_interval),
            "outer_motion_js": js_divergence(self.style.outer_motion, reference.outer_motion),
        }

    def style_fractions(self) -> dict[str, float]:
        def fraction(counter: Counter, key) -> float:
            total = sum(counter.values())
            return counter.get(key, 0) / total if total else 0.0

        return {
            "seventh_chords": fraction(self.style.seventh_use, "seventh"),
            "applied_chords": fraction(self.style.applied_use, "applied"),
            "root_position": fraction(self.style.inversion, 0),
            "first_inversion": fraction(self.style.inversion, 1),
            "second_inversion": fraction(self.style.inversion, 2),
            "contrary_outer_motion": fraction(self.style.outer_motion, "contrary"),
            "parallel_outer_motion": fraction(self.style.outer_motion, "parallel"),
        }


def reference_style(chorales: Sequence[Chorale]) -> StyleCounts:
    """Bach's own style statistics, pooled over a split."""
    total = StyleCounts()
    for chorale in chorales:
        total.merge(collect_style(chorale.voices, chorale.key, phrase_ends=chorale.fermatas))
    return total


def reference_result(chorales: Sequence[Chorale], name: str = "bach_oracle") -> EngineResult:
    """Score Bach's own settings with the identical metric code."""
    result = EngineResult(engine_id=name, name="J. S. Bach (ground truth)", learned=False)
    for chorale in chorales:
        result.pieces += 1
        result.defects.merge(collect_defects(chorale.voices, chorale.key))
        result.style.merge(collect_style(chorale.voices, chorale.key, phrase_ends=chorale.fermatas))
        result.agreement.merge(collect_agreement(chorale.voices, chorale.voices, chorale.key))
    return result


def evaluate_engine(
    engine: HarmonyEngine,
    chorales: Sequence[Chorale],
    *,
    supply_key: bool = True,
    seed: int = 0,
    score_likelihood: bool = True,
    verbose: bool = False,
) -> EngineResult:
    """Run one engine over the held-out set and pool every metric."""
    result = EngineResult(engine_id=engine.id, name=engine.name, learned=engine.learned)

    for chorale in chorales:
        melody = chorale_to_melody(chorale)
        if not supply_key:
            melody = melody.model_copy(update={"key": None})

        started = time.perf_counter()
        try:
            harmonization = engine.harmonize(melody, voice_count=4, temperature=0.0, seed=seed)
        except Exception as error:  # a crash is a result, not an excuse to skip
            result.failures += 1
            if verbose:
                print(f"  ! {engine.id} failed on {chorale.id}: {error}")
            continue
        result.seconds += time.perf_counter() - started

        lines = voices_to_grid(harmonization.voices, length=chorale.length)
        while len(lines) < 4:
            lines.append([REST] * chorale.length)
        lines = lines[:4]
        # The soprano is the given melody; substitute if an engine drops it.
        if all(p == REST for p in lines[0]):
            lines[0] = list(chorale.voices[0])

        key = Key(harmonization.key.tonic, harmonization.key.mode) if supply_key else chorale.key
        result.pieces += 1
        result.defects.merge(collect_defects(lines, key))
        result.style.merge(collect_style(lines, key, phrase_ends=chorale.fermatas))
        result.agreement.merge(collect_agreement(lines, chorale.voices, chorale.key))

        if score_likelihood:
            scored = engine.log_likelihood(melody, _reference_voices(chorale))
            if scored is not None:
                nats, tokens = scored
                result.nll_nats += nats
                result.nll_tokens += tokens

    return result


def _reference_voices(chorale: Chorale):
    from ..data.melody import grid_to_voices

    return grid_to_voices(chorale.voices, onsets=chorale.onsets)


def key_detection_accuracy(chorales: Sequence[Chorale]) -> dict[str, float]:
    """How often melody-only key finding recovers the corpus key.

    Reported because the primary table supplies the key: this is the size of the
    handicap that removes.
    """
    exact = relative = tonic_only = 0
    for chorale in chorales:
        grid = melody_to_grid(chorale_to_melody(chorale))
        detected, _ = detect_melody_key(grid)
        if detected == chorale.key:
            exact += 1
        if detected.tonic == chorale.key.tonic:
            tonic_only += 1
        relative_tonic = (chorale.key.tonic + (3 if chorale.key.is_minor else 9)) % 12
        if detected.tonic == relative_tonic and detected.mode != chorale.key.mode:
            relative += 1
    total = max(1, len(chorales))
    return {
        "exact": exact / total,
        "same_tonic": tonic_only / total,
        "relative_key_confusion": relative / total,
        "pieces": len(chorales),
    }


def defect_table(results: Sequence[EngineResult]) -> list[tuple[str, list[float]]]:
    rows: list[tuple[str, list[float]]] = []
    for kind in DEFECT_KINDS:
        rows.append((kind, [result.defects.per_hundred(kind) for result in results]))
    rows.append(("HARD TOTAL", [result.defects.hard_error_rate() for result in results]))
    return rows
