"""Metrics for comparing harmonizations.

The design principle, straight out of the v1 post-mortem: per-beat agreement
with Bach's own chord labels is a *bad* headline metric. A harmonization can
disagree with Bach on most beats and still be excellent, and v1's reported
accuracy was additionally inflated by counting padded positions. So agreement is
computed here, reported, and explicitly demoted.

The numbers that carry weight are:
  * objective voice-leading defect rates, which are true or false regardless of
    taste, and
  * distributional distance from the Bach corpus, which measures style without
    demanding note-for-note imitation.

Every engine's output and Bach's own voices go through the *same* analyser, so
an engine cannot flatter itself by declaring chord labels its voices do not
actually realise.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..data.corpus import REST, STEPS_PER_QUARTER
from ..theory.chords import ChordLabel, analyze_chord
from ..theory.pitch import Key
from ..theory.voicing import (
    ALTO,
    BASS,
    SOPRANO,
    TENOR,
    VoicedTexture,
    analyze_texture,
    count_chord_changes,
    motion_type,
)

#: Defects counted in the headline voice-leading table, in reporting order.
DEFECT_KINDS = (
    "parallel_fifths",
    "parallel_octaves",
    "contrary_fifths",
    "contrary_octaves",
    "direct_fifths",
    "direct_octaves",
    "voice_crossing",
    "voice_overlap",
    "spacing",
    "range",
    "unresolved_leading_tone",
    "unresolved_seventh",
    "doubled_leading_tone",
    "awkward_melodic_interval",
    "large_leap",
)

#: Defects that are unambiguously errors rather than stylistic preferences.
HARD_DEFECTS = ("parallel_fifths", "parallel_octaves", "voice_crossing", "range")


def to_texture(lines: Sequence[Sequence[int]], step: float = 0.25) -> VoicedTexture:
    """Grid lines (with REST sentinels) -> a VoicedTexture (with None)."""
    length = max((len(line) for line in lines), default=0)
    grid = [
        tuple(None if t >= len(line) or line[t] == REST else int(line[t]) for line in lines)
        for t in range(length)
    ]
    return VoicedTexture(grid=grid, step=step)


# ---------------------------------------------------------------------------
# Harmonic analysis of a voiced texture
# ---------------------------------------------------------------------------


def beat_chords(lines: Sequence[Sequence[int]], key: Key, *, steps_per_beat: int = STEPS_PER_QUARTER) -> list[ChordLabel | None]:
    """Analyse one chord per beat from the sounding voices.

    Both the engine output and Bach's own four parts go through this, so the
    comparison is symmetric and no engine benefits from its own label choices.
    """
    length = max((len(line) for line in lines), default=0)
    out: list[ChordLabel | None] = []
    for start in range(0, length, steps_per_beat):
        pitches = [line[start] for line in lines if start < len(line) and line[start] != REST]
        out.append(analyze_chord(pitches, key) if pitches else None)
    return out


def step_chords(lines: Sequence[Sequence[int]], key: Key, *, steps_per_beat: int = STEPS_PER_QUARTER) -> list[ChordLabel | None]:
    """Per-grid-step chord context, by holding each beat's analysis."""
    per_beat = beat_chords(lines, key, steps_per_beat=steps_per_beat)
    length = max((len(line) for line in lines), default=0)
    return [per_beat[min(t // steps_per_beat, len(per_beat) - 1)] if per_beat else None for t in range(length)]


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


def normalize(counter: Mapping, keys: Sequence) -> list[float]:
    total = sum(counter.get(k, 0) for k in keys)
    if total == 0:
        return [0.0] * len(keys)
    return [counter.get(k, 0) / total for k in keys]


def js_divergence(p: Mapping, q: Mapping, *, smoothing: float = 0.5) -> float:
    """Jensen-Shannon divergence in bits between two count distributions.

    Symmetric, bounded in [0, 1], and finite even when the supports differ —
    which they always do on a few thousand samples of a sparse chord vocabulary.
    Add-0.5 smoothing over the union of observed events.
    """
    keys = sorted(set(p) | set(q), key=repr)
    if not keys:
        return 0.0
    p_total = sum(p.values()) + smoothing * len(keys)
    q_total = sum(q.values()) + smoothing * len(keys)
    divergence = 0.0
    for key in keys:
        pi = (p.get(key, 0) + smoothing) / p_total
        qi = (q.get(key, 0) + smoothing) / q_total
        mi = 0.5 * (pi + qi)
        if pi > 0:
            divergence += 0.5 * pi * math.log2(pi / mi)
        if qi > 0:
            divergence += 0.5 * qi * math.log2(qi / mi)
    return divergence


# ---------------------------------------------------------------------------
# Style statistics
# ---------------------------------------------------------------------------

CADENCE_TYPES = ("PAC", "IAC", "HC", "deceptive", "plagal", "phrygian", "other")


def chord_class(chord: ChordLabel) -> tuple[int, str]:
    """Coarse identity used for distributional comparison."""
    return (chord.relative_root, chord.quality)


def cadence_type(penult: ChordLabel | None, final: ChordLabel | None, soprano_pc: int | None, key: Key) -> str:
    if final is None:
        return "other"
    if penult is None:
        return "other"
    dominant = penult.relative_root == 7 and penult.quality in ("maj", "dom7")
    leading = penult.relative_root == 11 and penult.quality in ("dim", "dim7", "halfdim7")
    tonic = final.relative_root == 0 and final.quality in ("maj", "min")

    if final.relative_root == 7 and final.quality in ("maj", "dom7") and final.inversion == 0:
        return "HC"
    if (dominant or leading) and tonic:
        root_position = penult.inversion == 0 and final.inversion == 0
        tonic_in_soprano = soprano_pc is not None and key.to_relative(soprano_pc) == 0
        return "PAC" if (root_position and tonic_in_soprano and dominant) else "IAC"
    if dominant and final.relative_root in (9, 8) and final.quality in ("min", "maj"):
        return "deceptive"
    if penult.relative_root == 5 and tonic:
        return "plagal"
    if penult.relative_root == 5 and penult.inversion == 1 and final.relative_root == 7:
        return "phrygian"
    return "other"


@dataclass
class StyleCounts:
    """Raw event counts, so statistics can be pooled across pieces correctly."""

    chord_unigram: Counter = field(default_factory=Counter)
    chord_bigram: Counter = field(default_factory=Counter)
    root_motion: Counter = field(default_factory=Counter)
    inversion: Counter = field(default_factory=Counter)
    quality: Counter = field(default_factory=Counter)
    cadence: Counter = field(default_factory=Counter)
    melodic_interval: Counter = field(default_factory=Counter)
    outer_motion: Counter = field(default_factory=Counter)
    seventh_use: Counter = field(default_factory=Counter)
    applied_use: Counter = field(default_factory=Counter)
    harmonic_rhythm: Counter = field(default_factory=Counter)

    def merge(self, other: "StyleCounts") -> None:
        for name in self.__dataclass_fields__:
            getattr(self, name).update(getattr(other, name))


def collect_style(
    lines: Sequence[Sequence[int]],
    key: Key,
    *,
    phrase_ends: Sequence[bool] | None = None,
    steps_per_beat: int = STEPS_PER_QUARTER,
) -> StyleCounts:
    """Every style statistic for one harmonization."""
    counts = StyleCounts()
    chords = beat_chords(lines, key, steps_per_beat=steps_per_beat)

    previous: ChordLabel | None = None
    changes = 0
    for chord in chords:
        if chord is None:
            continue
        counts.chord_unigram[chord_class(chord)] += 1
        counts.inversion[chord.inversion] += 1
        counts.quality[chord.quality] += 1
        counts.seventh_use["seventh" if chord.is_seventh else "triad"] += 1
        counts.applied_use["applied" if chord.applied_to is not None else "native"] += 1
        if previous is not None:
            if chord_class(previous) != chord_class(chord):
                counts.chord_bigram[(chord_class(previous), chord_class(chord))] += 1
                counts.root_motion[(chord.relative_root - previous.relative_root) % 12] += 1
                changes += 1
        previous = chord
    if chords:
        counts.harmonic_rhythm[round(changes / max(1, len(chords)), 1)] += 1

    for voice in (ALTO, TENOR, BASS):
        if voice >= len(lines):
            continue
        line = [p for p in lines[voice]]
        previous_pitch = None
        for t in range(len(line)):
            pitch = line[t]
            if pitch == REST:
                previous_pitch = None
                continue
            if previous_pitch is not None and pitch != previous_pitch:
                counts.melodic_interval[max(-12, min(12, pitch - previous_pitch))] += 1
            previous_pitch = pitch

    if len(lines) >= 4:
        soprano, bass = lines[SOPRANO], lines[BASS]
        previous_pair = None
        for t in range(min(len(soprano), len(bass))):
            if soprano[t] == REST or bass[t] == REST:
                previous_pair = None
                continue
            pair = (soprano[t], bass[t])
            if previous_pair is not None and pair != previous_pair:
                counts.outer_motion[motion_type(previous_pair[1], pair[1], previous_pair[0], pair[0])] += 1
            previous_pair = pair

    if phrase_ends is not None:
        for index, chord in enumerate(chords):
            beat_start = index * steps_per_beat
            if beat_start >= len(phrase_ends) or not phrase_ends[beat_start]:
                continue
            is_last_of_phrase = (
                beat_start + steps_per_beat >= len(phrase_ends)
                or not phrase_ends[beat_start + steps_per_beat]
            )
            if not is_last_of_phrase:
                continue
            # The approach chord is the last *different* one. A fermata note
            # usually spans two beats, so chords[index - 1] is normally the same
            # chord as chords[index] and every cadence would classify as "other".
            penult = None
            for back in range(index - 1, max(-1, index - 9), -1):
                candidate = chords[back]
                if candidate is None:
                    continue
                if chord is None or chord_class(candidate) != chord_class(chord):
                    penult = candidate
                    break
            soprano_pitch = lines[SOPRANO][beat_start] if beat_start < len(lines[SOPRANO]) else REST
            counts.cadence[cadence_type(
                penult, chord, None if soprano_pitch == REST else soprano_pitch % 12, key
            )] += 1

    return counts


# ---------------------------------------------------------------------------
# Harmonic activity
# ---------------------------------------------------------------------------


@dataclass
class ActivityCounts:
    """How much harmonic work an engine actually does.

    The metric that exposes playing it safe. An engine can drive its defect rate
    to zero by moving less and choosing blander chords, and every voice-leading
    number will improve while the music gets worse. Bach is the calibration
    here as everywhere: the target is his rate of harmonic change and his
    variety, not the maximum or the minimum of either.
    """

    pieces: int = 0
    beats: int = 0
    sonority_changes: int = 0
    chord_changes: int = 0
    chord_beats: int = 0
    safe_chord_beats: int = 0
    distinct_classes: Counter = field(default_factory=Counter)
    classes_per_piece: list = field(default_factory=list)

    def merge(self, other: "ActivityCounts") -> None:
        self.pieces += other.pieces
        self.beats += other.beats
        self.sonority_changes += other.sonority_changes
        self.chord_changes += other.chord_changes
        self.chord_beats += other.chord_beats
        self.safe_chord_beats += other.safe_chord_beats
        self.distinct_classes.update(other.distinct_classes)
        self.classes_per_piece.extend(other.classes_per_piece)

    def chord_changes_per_100_beats(self) -> float:
        return 100.0 * self.chord_changes / self.beats if self.beats else 0.0

    def sonority_changes_per_100_beats(self) -> float:
        return 100.0 * self.sonority_changes / self.beats if self.beats else 0.0

    def mean_classes_per_piece(self) -> float:
        return sum(self.classes_per_piece) / len(self.classes_per_piece) if self.classes_per_piece else 0.0

    def safe_chord_share(self) -> float:
        """Share of beats sitting on the tonic or the dominant.

        The single clearest "played it safe" signal: an engine that never leaves
        I and V will look excellent on every voice-leading metric.
        """
        return self.safe_chord_beats / self.chord_beats if self.chord_beats else 0.0


def collect_activity(
    lines: Sequence[Sequence[int]],
    key: Key,
    *,
    steps_per_beat: int = STEPS_PER_QUARTER,
) -> ActivityCounts:
    counts = ActivityCounts(pieces=1)
    chords = beat_chords(lines, key, steps_per_beat=steps_per_beat)
    counts.beats = len(chords)

    texture = to_texture(lines)
    counts.sonority_changes = count_chord_changes(texture)

    seen: set = set()
    previous = None
    for chord in chords:
        if chord is None:
            continue
        counts.chord_beats += 1
        identity = chord_class(chord)
        seen.add(identity)
        counts.distinct_classes[identity] += 1
        if chord.relative_root in (0, 7) and chord.applied_to is None:
            counts.safe_chord_beats += 1
        if previous is not None and identity != previous:
            counts.chord_changes += 1
        previous = identity
    counts.classes_per_piece.append(len(seen))
    return counts


# ---------------------------------------------------------------------------
# Voice-leading defects
# ---------------------------------------------------------------------------


@dataclass
class DefectCounts:
    counts: Counter = field(default_factory=Counter)
    chord_changes: int = 0

    def merge(self, other: "DefectCounts") -> None:
        self.counts.update(other.counts)
        self.chord_changes += other.chord_changes

    def per_hundred(self, kind: str) -> float:
        if self.chord_changes == 0:
            return 0.0
        return 100.0 * self.counts.get(kind, 0) / self.chord_changes

    def hard_error_rate(self) -> float:
        return sum(self.per_hundred(kind) for kind in HARD_DEFECTS)


def collect_defects(lines: Sequence[Sequence[int]], key: Key, *, steps_per_beat: int = STEPS_PER_QUARTER) -> DefectCounts:
    texture = to_texture(lines)
    chords = step_chords(lines, key, steps_per_beat=steps_per_beat)
    defects = analyze_texture(texture, key, chords)
    counts = Counter(defect.kind for defect in defects)
    return DefectCounts(counts=counts, chord_changes=count_chord_changes(texture))


# ---------------------------------------------------------------------------
# Agreement with Bach (reported, deliberately NOT the headline)
# ---------------------------------------------------------------------------


@dataclass
class AgreementCounts:
    beats: int = 0
    exact: int = 0
    root_quality: int = 0
    root: int = 0
    voice_notes: int = 0
    voice_matches: int = 0
    bass_notes: int = 0
    bass_matches: int = 0

    def merge(self, other: "AgreementCounts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def as_dict(self) -> dict[str, float]:
        def ratio(a: int, b: int) -> float:
            return a / b if b else 0.0

        return {
            "chord_exact": ratio(self.exact, self.beats),
            "chord_root_quality": ratio(self.root_quality, self.beats),
            "chord_root": ratio(self.root, self.beats),
            "voice_note": ratio(self.voice_matches, self.voice_notes),
            "bass_note": ratio(self.bass_matches, self.bass_notes),
        }


def collect_agreement(
    predicted: Sequence[Sequence[int]],
    reference: Sequence[Sequence[int]],
    key: Key,
    *,
    steps_per_beat: int = STEPS_PER_QUARTER,
) -> AgreementCounts:
    """Per-beat chord agreement and per-step voice agreement against Bach.

    Included because it is the number v1 optimised, and because seeing it move
    independently of the quality metrics is itself the argument against it.
    """
    counts = AgreementCounts()
    pred_chords = beat_chords(predicted, key, steps_per_beat=steps_per_beat)
    ref_chords = beat_chords(reference, key, steps_per_beat=steps_per_beat)
    for pred, ref in zip(pred_chords, ref_chords):
        if pred is None or ref is None:
            continue
        counts.beats += 1
        if (pred.relative_root, pred.quality, pred.inversion) == (ref.relative_root, ref.quality, ref.inversion):
            counts.exact += 1
        if (pred.relative_root, pred.quality) == (ref.relative_root, ref.quality):
            counts.root_quality += 1
        if pred.relative_root == ref.relative_root:
            counts.root += 1

    for voice in (ALTO, TENOR, BASS):
        if voice >= len(predicted) or voice >= len(reference):
            continue
        for pred, ref in zip(predicted[voice], reference[voice]):
            if ref == REST:
                continue
            counts.voice_notes += 1
            if pred == ref:
                counts.voice_matches += 1
            if voice == BASS:
                counts.bass_notes += 1
                if pred == ref:
                    counts.bass_matches += 1
    return counts


def summarize_distribution(counter: Counter, keys: Sequence, *, top: int = 8) -> list[tuple[str, float]]:
    total = sum(counter.values()) or 1
    ordered = sorted(counter.items(), key=lambda item: -item[1])[:top]
    return [(str(k), v / total) for k, v in ordered]
