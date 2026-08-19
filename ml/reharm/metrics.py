"""Jazz metrics. Deliberately NOT the chorale harness.

`ml/eval` and `ml/theory/voicing.py` encode chorale norms, and most of them are
actively wrong here: parallel fifths are an idiom (quartal voicings, planing),
voice crossing and wide spacing are normal, rootless voicings state neither root
nor fifth, and the Bach oracle is meaningless as a reference for a tritone
substitution. Reusing that harness would optimise this engine toward a target
nobody wants.

What is worth stealing from it is the *methodology*, which is oracle
calibration: score real music with your own metrics before you generate
anything, so you find out what the target actually looks like instead of
assuming. `oracle.py` does that with two corpora, and every threshold in this
package is set from those numbers rather than from taste.

Three families of measurement here:

1. **Melody compatibility** — a hard constraint. Every chord must support the
   melody sounding over it. This is the most common way automatic
   reharmonization fails and it is objectively checkable.
2. **Harmonic syntax** — does the result still resolve, still cadence, still
   mean something? A tritone sub that does not resolve down a semitone is not a
   tritone sub, it is a wrong chord.
3. **Distance from the original** — a reharmonization identical to the input is
   useless and one unrecognisably far from it is a different tune. The sweet
   spot is measurable, and it is the dial the product is actually built around.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .chords import (
    AVAILABLE_TENSION,
    CHORD_TONE,
    CONFLICT,
    SOFT_CONFLICT,
    STATED_TENSION,
    JazzChord,
    classify_melody_note,
    is_dominant_resolution,
    resolves_down_fifth,
    resolves_down_semitone,
)
from .data import ChordSpan, Progression

#: (start beat, midi pitch, duration in beats)
MelodyNote = tuple[float, int, float]

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
#: Jazz minor is a collection, not a scale: the raised 6 and 7 of melodic minor
#: are as native to a minor blues as the natural ones, so treating them as
#: chromatic would make every minor tune look wildly chromatic.
MINOR_COLLECTION = (0, 2, 3, 5, 7, 8, 9, 10, 11)


def diatonic_pcs(tonic: int, mode: str) -> frozenset[int]:
    scale = MINOR_COLLECTION if mode == "minor" else MAJOR_SCALE
    return frozenset((tonic + degree) % 12 for degree in scale)


def js_divergence(p: Mapping, q: Mapping, *, smoothing: float = 0.5) -> float:
    """Jensen-Shannon divergence in bits between two count distributions.

    Same definition as `ml/eval/metrics.py` so the numbers are directly
    comparable with the chorale side of the project; reimplemented rather than
    imported to keep this package independent of a module another workstream is
    actively editing.
    """
    keys = sorted(set(p) | set(q), key=repr)
    if not keys:
        return 0.0
    p_total = sum(p.get(k, 0) for k in keys) + smoothing * len(keys)
    q_total = sum(q.get(k, 0) for k in keys) + smoothing * len(keys)
    if p_total <= 0 or q_total <= 0:
        return 0.0
    divergence = 0.0
    for key in keys:
        pi = (p.get(key, 0) + smoothing) / p_total
        qi = (q.get(key, 0) + smoothing) / q_total
        mi = 0.5 * (pi + qi)
        if pi > 0:
            divergence += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            divergence += 0.5 * qi * math.log2(qi / mi)
    return max(0.0, min(1.0, divergence))


# ---------------------------------------------------------------------------
# Melody compatibility
# ---------------------------------------------------------------------------

#: A note landing on the beat carries more harmonic weight than one between
#: beats: jazz phrasing puts chord tones on strong beats and passing
#: dissonance in between, so an unweighted count would call every bebop line
#: a catastrophe.
ON_BEAT_EMPHASIS = 1.6
STRONG_BEAT_EMPHASIS = 2.0


def note_weight(start: float, duration: float, *, beats_per_bar: int = 4) -> float:
    """Harmonic weight of a melody note: how much it constrains the chord."""
    position = start % beats_per_bar
    on_beat = abs(position - round(position)) < 0.08
    strong = on_beat and (round(position) % 2 == 0)
    emphasis = STRONG_BEAT_EMPHASIS if strong else (ON_BEAT_EMPHASIS if on_beat else 1.0)
    return max(0.125, duration) * emphasis


@dataclass
class MelodyFit:
    """How a melody sits over a chord sequence, weighted by harmonic weight."""

    weight: float = 0.0
    by_verdict: Counter = field(default_factory=Counter)
    conflicts: list[tuple[float, int, JazzChord, str]] = field(default_factory=list)
    notes: int = 0
    uncovered: float = 0.0

    def merge(self, other: "MelodyFit") -> None:
        self.weight += other.weight
        self.by_verdict.update(other.by_verdict)
        self.conflicts.extend(other.conflicts)
        self.notes += other.notes
        self.uncovered += other.uncovered

    def rate(self, verdict: str) -> float:
        return self.by_verdict.get(verdict, 0.0) / self.weight if self.weight else 0.0

    @property
    def chord_tone_rate(self) -> float:
        return self.rate(CHORD_TONE)

    @property
    def tension_rate(self) -> float:
        return self.rate(STATED_TENSION) + self.rate(AVAILABLE_TENSION)

    @property
    def soft_conflict_rate(self) -> float:
        return self.rate(SOFT_CONFLICT)

    @property
    def hard_conflict_rate(self) -> float:
        return self.rate(CONFLICT)

    def as_dict(self) -> dict[str, float]:
        return {
            "chord_tone_rate": self.chord_tone_rate,
            "tension_rate": self.tension_rate,
            "soft_conflict_rate": self.soft_conflict_rate,
            "hard_conflict_rate": self.hard_conflict_rate,
            "notes": float(self.notes),
        }


def melody_fit(
    melody: Sequence[MelodyNote],
    progression: Progression,
    *,
    beats_per_bar: int | None = None,
) -> MelodyFit:
    """Classify every melody note against the chord sounding under it.

    Notes are split at chord boundaries: a note held across a chord change is
    judged against BOTH chords, which is the whole point — reharmonization
    changes what a held note means, and a metric that only looked at the chord
    at the note's onset would be blind to exactly the failure mode it exists to
    catch.
    """
    per_bar = beats_per_bar if beats_per_bar is not None else progression.meter[0]
    fit = MelodyFit()
    for start, pitch, duration in melody:
        fit.notes += 1
        for span, overlap in _overlapping_spans(progression.spans, start, duration):
            weight = note_weight(max(start, span.start), overlap, beats_per_bar=per_bar)
            verdict = classify_melody_note(span.chord, pitch % 12)
            fit.weight += weight
            fit.by_verdict[verdict.verdict] += weight
            if verdict.verdict in (CONFLICT, SOFT_CONFLICT):
                fit.conflicts.append((max(start, span.start), pitch, span.chord, verdict.verdict))
        covered = sum(overlap for _, overlap in _overlapping_spans(progression.spans, start, duration))
        fit.uncovered += max(0.0, duration - covered)
    return fit


def _overlapping_spans(
    spans: Sequence[ChordSpan], start: float, duration: float
) -> list[tuple[ChordSpan, float]]:
    stop = start + duration
    out: list[tuple[ChordSpan, float]] = []
    for span in spans:
        overlap = min(stop, span.stop) - max(start, span.start)
        if overlap > 1e-6:
            out.append((span, overlap))
    return out


def hard_conflicts(melody: Sequence[MelodyNote], progression: Progression) -> list[tuple[float, int, JazzChord]]:
    """Every melody note that a chord genuinely cannot support."""
    fit = melody_fit(melody, progression)
    return [(t, pitch, chord) for t, pitch, chord, verdict in fit.conflicts if verdict == CONFLICT]


# ---------------------------------------------------------------------------
# Harmonic syntax
# ---------------------------------------------------------------------------


@dataclass
class SyntaxCounts:
    """Countable facts about a chord sequence. Mergeable across a corpus."""

    chords: int = 0
    transitions: int = 0
    beats: float = 0.0
    sevenths: int = 0
    with_extensions: int = 0
    extension_total: int = 0
    dominants: int = 0
    dominants_resolved: int = 0
    dominants_down_fifth: int = 0
    dominants_down_semitone: int = 0
    ii_v: int = 0
    ii_v_i: int = 0
    nondiatonic_roots: int = 0
    chromatic_tones: int = 0
    total_tones: int = 0
    ends_on_tonic: int = 0
    progressions: int = 0
    #: key-relative (root, quality) and root-motion distributions
    vocabulary: Counter = field(default_factory=Counter)
    root_motion: Counter = field(default_factory=Counter)
    qualities: Counter = field(default_factory=Counter)

    def merge(self, other: "SyntaxCounts") -> None:
        for name in (
            "chords", "transitions", "sevenths", "with_extensions", "extension_total",
            "dominants", "dominants_resolved", "dominants_down_fifth", "dominants_down_semitone",
            "ii_v", "ii_v_i", "nondiatonic_roots", "chromatic_tones", "total_tones",
            "ends_on_tonic", "progressions",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.beats += other.beats
        self.vocabulary.update(other.vocabulary)
        self.root_motion.update(other.root_motion)
        self.qualities.update(other.qualities)

    def _ratio(self, numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "chords": float(self.chords),
            "mean_chord_beats": self._ratio(self.beats, self.chords),
            "seventh_rate": self._ratio(self.sevenths, self.chords),
            "extension_rate": self._ratio(self.with_extensions, self.chords),
            "mean_extensions": self._ratio(self.extension_total, self.chords),
            "dominant_rate": self._ratio(self.dominants, self.chords),
            "dominant_resolution_rate": self._ratio(self.dominants_resolved, self.dominants),
            "semitone_resolution_share": self._ratio(self.dominants_down_semitone, self.dominants_resolved),
            "ii_v_per_16_bars": self._ratio(self.ii_v * 64.0, self.beats),
            "ii_v_i_share": self._ratio(self.ii_v_i, self.ii_v),
            "nondiatonic_root_rate": self._ratio(self.nondiatonic_roots, self.chords),
            "chromatic_tone_rate": self._ratio(self.chromatic_tones, self.total_tones),
            "ends_on_tonic_rate": self._ratio(self.ends_on_tonic, self.progressions),
        }


def is_ii_of(a: JazzChord, b: JazzChord) -> bool:
    """Whether `a` is the related ii of dominant `b` (Dm7 before G7)."""
    return (
        a.quality in ("min7", "halfdim7", "min", "min6")
        and b.is_dominant
        and (b.root - a.root) % 12 == 5
    )


def collect_syntax(progression: Progression) -> SyntaxCounts:
    """Measure one chord sequence."""
    counts = SyntaxCounts(progressions=1)
    spans = progression.spans
    if not spans:
        return counts
    diatonic = diatonic_pcs(progression.tonic, progression.mode)

    for index, span in enumerate(spans):
        chord = span.chord
        counts.chords += 1
        counts.beats += span.duration
        counts.sevenths += 1 if chord.is_seventh else 0
        counts.with_extensions += 1 if chord.extensions else 0
        counts.extension_total += len(chord.extensions)
        counts.vocabulary[((chord.root - progression.tonic) % 12, chord.quality)] += 1
        counts.qualities[chord.quality] += 1
        if chord.root not in diatonic:
            counts.nondiatonic_roots += 1
        for pitch_class in chord.all_pcs:
            counts.total_tones += 1
            if pitch_class not in diatonic:
                counts.chromatic_tones += 1

        following = spans[index + 1].chord if index + 1 < len(spans) else None
        if following is not None:
            counts.transitions += 1
            counts.root_motion[(following.root - chord.root) % 12] += 1
        if chord.is_dominant:
            counts.dominants += 1
            if following is not None:
                if resolves_down_fifth(chord, following):
                    counts.dominants_resolved += 1
                    counts.dominants_down_fifth += 1
                elif resolves_down_semitone(chord, following):
                    counts.dominants_resolved += 1
                    counts.dominants_down_semitone += 1
        if following is not None and is_ii_of(chord, following):
            counts.ii_v += 1
            after = spans[index + 2].chord if index + 2 < len(spans) else None
            if after is not None and is_dominant_resolution(following, after):
                counts.ii_v_i += 1

    final = spans[-1].chord
    if (final.root - progression.tonic) % 12 in (0, 9 if progression.mode == "major" else 3):
        counts.ends_on_tonic += 1
    return counts


def collect_corpus_syntax(progressions: Iterable[Progression]) -> SyntaxCounts:
    total = SyntaxCounts()
    for progression in progressions:
        total.merge(collect_syntax(progression))
    return total


# ---------------------------------------------------------------------------
# Distance from the base progression
# ---------------------------------------------------------------------------


@dataclass
class DistanceMetrics:
    """How far a reharmonization travelled from the progression it started on."""

    changed_rate: float = 0.0
    root_change_rate: float = 0.0
    pc_distance: float = 0.0
    bass_change_rate: float = 0.0
    chord_ratio: float = 1.0
    base_chords: int = 0
    new_chords: int = 0

    def as_dict(self) -> dict[str, float]:
        return {
            "changed_rate": self.changed_rate,
            "root_change_rate": self.root_change_rate,
            "pc_distance": self.pc_distance,
            "bass_change_rate": self.bass_change_rate,
            "chord_ratio": self.chord_ratio,
        }


def _sample_points(base: Progression, other: Progression, *, step: float = 0.5) -> list[float]:
    end = max(base.duration, other.duration)
    count = max(1, int(round(end / step)))
    return [i * step + step / 2 for i in range(count)]


def distance(base: Progression, other: Progression, *, step: float = 0.5) -> DistanceMetrics:
    """Compare two progressions on a common time grid.

    Sampling in TIME rather than aligning chord lists is what makes this work
    when the reharmonization inserts chords: a ii-V inserted into one bar does
    not desynchronise everything after it, which an index-wise comparison would
    report as a total rewrite.
    """
    points = _sample_points(base, other, step=step)
    changed = roots = bass = 0
    overlap_total = 0.0
    counted = 0
    for point in points:
        a = base.chord_at(point)
        b = other.chord_at(point)
        if a is None or b is None:
            continue
        counted += 1
        if not a.same_harmony(b):
            changed += 1
        if a.root != b.root:
            roots += 1
        if a.bass_pc != b.bass_pc:
            bass += 1
        pcs_a, pcs_b = set(a.all_pcs), set(b.all_pcs)
        union = pcs_a | pcs_b
        overlap_total += 1.0 - (len(pcs_a & pcs_b) / len(union) if union else 1.0)
    if not counted:
        return DistanceMetrics(base_chords=len(base.spans), new_chords=len(other.spans))
    return DistanceMetrics(
        changed_rate=changed / counted,
        root_change_rate=roots / counted,
        pc_distance=overlap_total / counted,
        bass_change_rate=bass / counted,
        chord_ratio=len(other.spans) / len(base.spans) if base.spans else 1.0,
        base_chords=len(base.spans),
        new_chords=len(other.spans),
    )


# ---------------------------------------------------------------------------
# The headline score
# ---------------------------------------------------------------------------

#: Empirically calibrated from `oracle.py`. Comparing 163 tunes that appear in
#: both corpora — an iRealPro lead sheet against the changes a band actually
#: played on the same tune — the median ROOT change rate is 0.14 and the median
#: whole-chord change rate 0.34. So 0.14 is roughly what incidental
#: version-to-version variation looks like, and anything below it has not
#: reharmonized the tune so much as re-voiced it. The upper bound is a
#: judgement rather than a measurement: past about half the roots the harmonic
#: identity of the tune is gone, which is a different product.
DISTANCE_BAND = (0.15, 0.55)


def distance_reward(root_change_rate: float, band: tuple[float, float] = DISTANCE_BAND) -> float:
    """1.0 inside the sweet spot, falling off linearly outside it."""
    low, high = band
    if low <= root_change_rate <= high:
        return 1.0
    if root_change_rate < low:
        return max(0.0, root_change_rate / low) if low > 0 else 0.0
    return max(0.0, 1.0 - (root_change_rate - high) / max(1e-6, 1.0 - high))


@dataclass
class ReharmScore:
    """One number per axis, plus a headline that trades them off explicitly.

    The weights are a judgement call and are stated here rather than buried, so
    that disagreeing with them is easy.
    """

    melody: MelodyFit
    syntax: SyntaxCounts
    distance: DistanceMetrics

    @property
    def melody_penalty(self) -> float:
        return self.melody.hard_conflict_rate + 0.35 * self.melody.soft_conflict_rate

    @property
    def headline(self) -> float:
        syntax = self.syntax.as_dict()
        resolution = syntax["dominant_resolution_rate"]
        # The oracle measures 0.162 chromatic tones per chord tone on 1170 lead
        # sheets and 0.184 on the changes as played, so that — not zero and not
        # "as much as possible" — is what full marks for colour means.
        colour = min(1.0, syntax["chromatic_tone_rate"] / 0.18)
        return (
            0.40 * (1.0 - min(1.0, self.melody_penalty * 6.0))
            + 0.25 * resolution
            + 0.20 * distance_reward(self.distance.root_change_rate)
            + 0.15 * colour
        )

    def as_dict(self) -> dict[str, float]:
        out: dict[str, float] = {"headline": self.headline, "melody_penalty": self.melody_penalty}
        out.update(self.melody.as_dict())
        out.update(self.syntax.as_dict())
        out.update(self.distance.as_dict())
        return out


def score(melody: Sequence[MelodyNote], base: Progression, result: Progression) -> ReharmScore:
    return ReharmScore(
        melody=melody_fit(melody, result),
        syntax=collect_syntax(result),
        distance=distance(base, result),
    )
