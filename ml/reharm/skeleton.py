"""The base progression: a clean diatonic-functional skeleton to reharmonize.

Reharmonization is taught, and practised, as a transformation of the *basic
changes*: you have to know what the plain harmony is before you can substitute
for it. The project already has a strong producer of plain functional harmony —
the rules engine, which does Viterbi chord search over a full functional
vocabulary — so the skeleton comes from there rather than from a second,
weaker implementation of the same idea.

The one non-obvious step is harmonic rhythm. The rules engine decides a chord
per beat, which is chorale harmonic rhythm; jazz changes move roughly every two
to four beats (the oracle measures 2.83 beats per chord on 1170 lead sheets and
4.18 on the Weimar changes as played). Reharmonizing a per-beat skeleton would
produce four substitutions a bar, which is not a reharmonization but a seizure.
So the skeleton is reduced onto bar and half-bar units first, and substitution
operates on those.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from contracts.schema import Chord, KeySignature, Melody

from .chords import JazzChord
from .data import ChordSpan, Progression
from .metrics import MelodyNote, note_weight

#: Qualities the rules engine can emit. All are already in our vocabulary, so
#: the skeleton needs no lossy translation step — the v1 post-mortem in
#: ml/theory/chords.py is about exactly that kind of projection.
_RULE_QUALITIES = frozenset(
    {"maj", "min", "dim", "aug", "dom7", "maj7", "min7", "halfdim7", "dim7", "minmaj7"}
)


@dataclass
class Unit:
    """One harmonic decision point for the reharmonizer."""

    start: float
    duration: float
    base: JazzChord
    base_roman: str
    #: Melody notes clipped to this unit, in (start, pitch, duration) form.
    melody: list[MelodyNote] = field(default_factory=list)
    #: 3 = bar downbeat, 2 = mid-bar, 1 = elsewhere.
    metric_level: int = 1
    is_last: bool = False

    @property
    def stop(self) -> float:
        return self.start + self.duration

    @property
    def weighted_pcs(self) -> list[tuple[int, float]]:
        """Melody pitch classes in this unit with their harmonic weight."""
        totals: dict[int, float] = {}
        for start, pitch, duration in self.melody:
            totals[pitch % 12] = totals.get(pitch % 12, 0.0) + note_weight(start, duration)
        return sorted(totals.items(), key=lambda item: (-item[1], item[0]))

    @property
    def melody_weight(self) -> float:
        return sum(weight for _, weight in self.weighted_pcs)

    def half(self) -> tuple[float, float]:
        """(start, duration) of each half of the unit, for inserted chords."""
        return self.start + self.duration / 2, self.duration / 2


def chord_to_jazz(chord: Chord) -> JazzChord | None:
    """`contracts.schema.Chord` -> `JazzChord`, ignoring provenance."""
    if chord.quality not in _RULE_QUALITIES and chord.quality not in ("maj6", "min6", "sus2", "sus4"):
        return None
    bass = None
    if chord.inversion:
        from .chords import QUALITY_TEMPLATES

        intervals = QUALITY_TEMPLATES[chord.quality]
        if chord.inversion < len(intervals):
            bass = (chord.root + intervals[chord.inversion]) % 12
    return JazzChord(root=chord.root, quality=chord.quality, extensions=tuple(chord.extensions), bass=bass)


def melody_notes(melody: Melody) -> list[MelodyNote]:
    """API melody -> (start, pitch, duration) in quarter-note beats from 0."""
    if not melody.notes:
        return []
    origin = min(note.start for note in melody.notes)
    return [
        (round(note.start - origin, 6), int(note.pitch), round(note.duration, 6))
        for note in sorted(melody.notes, key=lambda n: (n.start, n.pitch))
    ]


def clip_melody(melody: Sequence[MelodyNote], start: float, stop: float) -> list[MelodyNote]:
    """Melody notes overlapping [start, stop), clipped to it.

    Clipping matters: a whole note held across four substituted chords must be
    judged against each of them, and the part of it that sounds under a given
    chord is the part that constrains that chord.
    """
    out: list[MelodyNote] = []
    for note_start, pitch, duration in melody:
        overlap_start = max(note_start, start)
        overlap_stop = min(note_start + duration, stop)
        if overlap_stop - overlap_start > 1e-6:
            out.append((overlap_start, pitch, overlap_stop - overlap_start))
    return out


def build_units(
    spans: Sequence[ChordSpan],
    melody: Sequence[MelodyNote],
    *,
    meter: tuple[int, int] = (4, 4),
    min_unit: float = 2.0,
) -> list[Unit]:
    """Reduce a beat-level skeleton onto bar / half-bar harmonic units.

    The grid is decided by the METRE and the MELODY, never by which chord the
    rules engine happened to pick. That is a deliberate constraint and it was
    found by a test: deciding to split a bar because its two halves want
    different chords makes the unit count depend on chord choice, chord choice
    depends on register because SATB ranges are absolute, and so the same tune a
    fifth higher came out with a different number of chords — a transposition
    bug manufactured entirely inside this function.

    So: every bar splits at the half, and two halves merge back into one unit
    when the melody does not articulate the boundary. A held note across the
    middle of the bar is the tune saying the harmony can stay put, and it says
    the same thing in every key.
    """
    if not spans:
        return []
    numerator, denominator = meter
    beats_per_bar = numerator * (4.0 / denominator)
    start = spans[0].start
    end = spans[-1].stop
    splittable = beats_per_bar >= 2 * min_unit and numerator % 2 == 0
    onsets = sorted(round(note_start, 6) for note_start, _, _ in melody)

    def articulated(offset: float) -> bool:
        return any(abs(onset - offset) < 0.05 for onset in onsets)

    boundaries: list[tuple[float, float]] = []
    bar_start = start
    while bar_start < end - 1e-6:
        bar_stop = min(bar_start + beats_per_bar, end)
        middle = bar_start + beats_per_bar / 2
        if splittable and middle < bar_stop - 1e-6 and articulated(middle):
            boundaries.append((bar_start, middle))
            boundaries.append((middle, bar_stop))
        else:
            boundaries.append((bar_start, bar_stop))
        bar_start = bar_stop

    units: list[Unit] = []
    for unit_start, unit_stop in boundaries:
        if unit_stop - unit_start <= 1e-6:
            continue
        chord = _dominant_chord(spans, unit_start, unit_stop)
        if chord is None:
            continue
        position = (unit_start - start) % beats_per_bar
        units.append(Unit(
            start=unit_start,
            duration=unit_stop - unit_start,
            base=chord,
            base_roman="",
            melody=clip_melody(melody, unit_start, unit_stop),
            metric_level=3 if position < 1e-6 else 2,
        ))
    if units:
        units[-1].is_last = True
    return units


def _dominant_chord(spans: Sequence[ChordSpan], start: float, stop: float) -> JazzChord | None:
    """The chord holding the most weight in [start, stop).

    Weight is overlap duration times a metric bonus for starting on the unit's
    downbeat, so a chord struck on the beat outranks a passing chord of equal
    length.
    """
    totals: dict[tuple[int, str], tuple[float, JazzChord]] = {}
    for span in spans:
        overlap = min(stop, span.stop) - max(start, span.start)
        if overlap <= 1e-6:
            continue
        emphasis = 1.5 if span.start <= start + 1e-6 else 1.0
        key = (span.chord.root, span.chord.quality)
        weight = overlap * emphasis
        previous = totals.get(key)
        totals[key] = (weight + (previous[0] if previous else 0.0), previous[1] if previous else span.chord)
    best: tuple[float, JazzChord | None] = (0.0, None)
    for weight, chord in totals.values():
        if weight > best[0]:
            best = (weight, chord)
    return best[1]


@dataclass
class Skeleton:
    """The rules engine's harmony, reduced to jazz harmonic rhythm."""

    units: list[Unit]
    tonic: int
    mode: str
    meter: tuple[int, int]
    melody: list[MelodyNote]
    key: KeySignature

    def progression(self, title: str = "skeleton") -> Progression:
        return Progression(
            spans=[ChordSpan(unit.start, unit.duration, unit.base) for unit in self.units],
            tonic=self.tonic,
            mode=self.mode,
            meter=self.meter,
            title=title,
            source="rules",
        )


#: Where the rules engine needs the tune to sit. SATB ranges are absolute, so
#: its voicing search simply fails for a melody outside the soprano range and it
#: returns almost no chords: a tune two octaves low came back with ONE chord for
#: thirteen bars. That is not a bug in the rules engine, it is what a chorale
#: voicer should do — but harmony is pitch classes, so the fix is to ask it the
#: question in a register it can answer.
SKELETON_WINDOW = (62, 76)


def octave_shift_for(melody: Melody, window: tuple[int, int] = SKELETON_WINDOW) -> int:
    """Whole-octave shift that brings the median melody pitch into `window`."""
    if not melody.notes:
        return 0
    pitches = sorted(note.pitch for note in melody.notes)
    median = pitches[len(pitches) // 2]
    low, high = window
    shift = 0
    while median + shift < low and max(pitches) + shift + 12 <= 127:
        shift += 12
    while median + shift > high and min(pitches) + shift - 12 >= 0:
        shift -= 12
    return shift


def transpose(melody: Melody, semitones: int) -> Melody:
    if semitones == 0:
        return melody
    return melody.model_copy(update={
        "notes": [
            note.model_copy(update={"pitch": max(0, min(127, note.pitch + semitones))})
            for note in melody.notes
        ]
    })


def skeleton_from_rules(
    melody: Melody,
    *,
    engine=None,
    min_unit: float = 2.0,
) -> Skeleton:
    """Harmonize with the existing rules engine, then reduce to jazz units.

    The melody is octave-normalized before the rules engine sees it and the
    result is used against the original. That is sound rather than a trick:
    chords are pitch classes and an octave shift changes none of them, key
    detection is duration-weighted over pitch classes, and the voicing this
    package produces is its own. All that changes is whether the chorale voicer
    could answer the question at all.
    """
    if engine is None:
        from ..engines.rules import RuleHarmonyEngine

        engine = RuleHarmonyEngine()
    result = engine.harmonize(transpose(melody, octave_shift_for(melody)), voice_count=4)

    spans: list[ChordSpan] = []
    romans: list[str] = []
    for chord in result.chords:
        jazz = chord_to_jazz(chord)
        if jazz is None:
            continue
        spans.append(ChordSpan(chord.start, chord.duration, jazz))
        romans.append(chord.roman)

    line = melody_notes(melody)
    meter = (melody.timeSignature.numerator, melody.timeSignature.denominator)
    units = build_units(spans, line, meter=meter, min_unit=min_unit)
    _attach_romans(units, spans, romans)
    return Skeleton(
        units=units,
        tonic=result.key.tonic,
        mode=result.key.mode,
        meter=meter,
        melody=line,
        key=result.key,
    )


def _attach_romans(units: Sequence[Unit], spans: Sequence[ChordSpan], romans: Sequence[str]) -> None:
    """Carry the rules engine's roman numeral onto each unit.

    This is what `substitutionOf` reports, so it has to be the label the base
    engine actually produced — "this bII7 replaced your V7" is only true if the
    V7 is quoted verbatim from the engine that wrote it.
    """
    for unit in units:
        best_overlap = 0.0
        for span, roman in zip(spans, romans):
            overlap = min(unit.stop, span.stop) - max(unit.start, span.start)
            if overlap > best_overlap and span.chord.same_harmony(unit.base):
                best_overlap, unit.base_roman = overlap, roman
        if not unit.base_roman:
            for span, roman in zip(spans, romans):
                overlap = min(unit.stop, span.stop) - max(unit.start, span.start)
                if overlap > best_overlap:
                    best_overlap, unit.base_roman = overlap, roman
