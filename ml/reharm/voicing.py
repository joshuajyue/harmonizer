"""Jazz voicing. The chorale voicer would be actively wrong here.

`ml/theory/voicing.py` and the rules engine's SATB search encode chorale norms,
and in this idiom most of them invert:

  * parallel fifths are an idiom, not a defect — quartal voicings move in
    parallel by design and planing is a technique;
  * spacing is wide and irregular, and voices cross freely;
  * the root is routinely absent. A rootless left-hand voicing states 3-5-7-9
    or 7-9-3-5 and nothing else, and the bass is a separate instrument.

So this is written from scratch, and what it borrows from the rules engine is
technique rather than rules: enumerate the legal arrangements of a chord, then
choose the path through them by dynamic programming with voice leading as the
cost. That approach was right there and is right here; only the cost function
changes.

The one rule that survives intact, because it is acoustics rather than style:
nothing may sound a semitone below the melody. A chord tone a semitone under
the tune produces a minor ninth against it, which is the interval the ear reads
as a mistake. It is also, conveniently, the same constraint the reharmonizer
already enforces in `chords.classify_melody_note` — stated once as harmony and
once as spacing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from contracts.schema import Note, Voice, VoiceName

from .chords import EXTENSION_SEMITONES, JazzChord, classify_melody_note
from .data import ChordSpan
from .metrics import MelodyNote

#: Comping register for the inner voices. The floor is low (Bb2) on purpose:
#: when the tune sits at the bottom of its range there is no room for a voicing
#: between it and a fixed floor, and a real pianist simply plays the left hand
#: under the melody rather than refusing to play. The bass, being a different
#: instrument, is free to sit below that.
INNER_LOW = 48   # C3
INNER_HIGH = 81  # A5
BASS_LOW = 33    # A1
BASS_HIGH = 55   # G3

#: Keep the top of the accompaniment out of the melody's way.
MELODY_CLEARANCE = 2

MAX_INNER_SPREAD = 16


@dataclass(frozen=True)
class VoicingStyle:
    """Which arrangements of a chord the voicer is allowed to consider."""

    rootless: bool = True
    quartal: bool = True
    drop2: bool = True
    upper_structure: bool = True
    walking_bass: bool = True


DEFAULT_STYLE = VoicingStyle()


# ---------------------------------------------------------------------------
# Choosing which notes to play
# ---------------------------------------------------------------------------


def _tension_intervals(chord: JazzChord) -> list[int]:
    return [EXTENSION_SEMITONES[extension] for extension in chord.extensions]


def _colour_intervals(chord: JazzChord) -> list[int]:
    """Default colour tones for a quality, when the melody has not chosen any.

    A jazz chord symbol is a floor, not a ceiling: "Cm7" means at least those
    four notes, and a pianist supplies the 9 without being asked. Adding them
    here is what makes the result sound like the idiom rather than like a
    theory exercise.
    """
    if chord.quality in ("dom7", "sus4"):
        return [2, 9]
    if chord.quality in ("maj7", "maj6"):
        return [2]
    if chord.quality in ("min7", "min6", "minmaj7"):
        return [2]
    if chord.quality == "halfdim7":
        return [5]
    return []


def pitch_class_sets(chord: JazzChord, style: VoicingStyle, count: int) -> list[list[int]]:
    """Candidate note choices for a chord, as semitone intervals above the root.

    Ordered from most to least idiomatic; the voice-leading search picks among
    them, so a voicing that happens to lead well can win over one that is
    nominally more standard.
    """
    third = chord.third_interval
    seventh = chord.seventh_interval
    fifth = 7 if 7 in chord.core_intervals else next(
        (interval for interval in chord.core_intervals if interval in (6, 8)), None
    )
    tensions = _tension_intervals(chord)
    colours = [c for c in _colour_intervals(chord) if c not in tensions]

    guide = [interval for interval in (third, seventh) if interval is not None]
    options: list[list[int]] = []

    def add(intervals: Iterable[int | None]) -> None:
        cleaned = [interval for interval in intervals if interval is not None]
        deduped = list(dict.fromkeys(interval % 12 for interval in cleaned))
        if not deduped:
            return
        trimmed = deduped[:count] if len(deduped) > count else deduped
        if len(trimmed) == count and trimmed not in options:
            options.append(trimmed)

    if style.rootless:
        # Bill Evans A and B: 3-5-7-9 and 7-9-3-5. Same notes, different
        # inversion, and which one you play is decided by register.
        add([third, fifth, seventh, *tensions, *colours])
        add([seventh, *tensions, *colours, third, fifth])
        add([*guide, *tensions, *colours])
    if style.upper_structure and chord.is_dominant and tensions:
        # Guide tones underneath, an altered triad on top.
        add([third, seventh, *tensions])
    if style.quartal and chord.quality in ("min7", "sus4", "dom7", "min6"):
        base = 0 if chord.quality == "sus4" else (third or 0)
        add([base, (base + 5) % 12, (base + 10) % 12, (base + 15) % 12])
    add([*guide, *tensions, *colours, fifth, 0])
    add([*guide, fifth, 0])
    add([*guide, 0])
    add([0, third if third is not None else 0])
    add([0])
    if not options:
        options.append([0])
    return options


# ---------------------------------------------------------------------------
# Placing them
# ---------------------------------------------------------------------------


def _melody_pitch_at(melody: Sequence[MelodyNote], start: float, stop: float) -> int | None:
    """The melody pitch with the most weight inside a span."""
    best: tuple[float, int] | None = None
    for note_start, pitch, duration in melody:
        overlap = min(note_start + duration, stop) - max(note_start, start)
        if overlap <= 1e-6:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, pitch)
    return best[1] if best else None


def _melody_pitches_in(melody: Sequence[MelodyNote], start: float, stop: float) -> list[int]:
    return [
        pitch
        for note_start, pitch, duration in melody
        if min(note_start + duration, stop) - max(note_start, start) > 1e-6
    ]


def arrange(intervals: Sequence[int], root: int, ceiling: int, floor: int = INNER_LOW) -> list[list[int]]:
    """Every ascending placement of a set of pitch classes under `ceiling`."""
    if not intervals:
        return []
    pcs = [(root + interval) % 12 for interval in intervals]
    voicings: list[list[int]] = [[]]
    for pitch_class in pcs:
        grown: list[list[int]] = []
        for partial in voicings:
            low = max(floor, (partial[-1] + 1) if partial else floor)
            pitch = low + ((pitch_class - low) % 12)
            while pitch < ceiling:
                if not partial or pitch - partial[0] <= MAX_INNER_SPREAD:
                    grown.append([*partial, pitch])
                pitch += 12
        voicings = grown
        if not voicings:
            return []
    return voicings


def _clash_with_melody(voicing: Sequence[int], melody_pitches: Sequence[int]) -> bool:
    """Whether any voiced note fights a melody note sounding at the same time.

    Two things are forbidden and nothing else. A voiced note a semitone under a
    melody note makes a minor ninth against it, which is the interval the ear
    hears as wrong. A voiced note within a semitone of the melody in absolute
    pitch collides with it. Sitting ABOVE a passing low melody note is allowed:
    voice crossing is normal here, and forbidding it would drag the whole
    accompaniment into the bass every time the tune dips.
    """
    for melody_pitch in melody_pitches:
        for pitch in voicing:
            gap = melody_pitch - pitch
            if abs(gap) < MELODY_CLEARANCE:
                return True
            if gap > 0 and gap % 12 == 1 and gap <= 13:
                return True
    return False


def _voicing_cost(voicing: Sequence[int], previous: Sequence[int] | None, centre: float = 66.0) -> float:
    """Register preference plus movement from the previous chord.

    The register term is anchored under the melody rather than at a fixed
    pitch, because "where the left hand goes" is relative to where the tune is.
    """
    cost = 0.02 * sum(abs(pitch - centre) for pitch in voicing) / max(1, len(voicing))
    spread = voicing[-1] - voicing[0] if len(voicing) > 1 else 0
    cost += 0.05 * max(0, spread - 12)
    if previous:
        pairs = min(len(previous), len(voicing))
        movement = sum(abs(voicing[i] - previous[i]) for i in range(pairs))
        movement += 3.0 * abs(len(voicing) - len(previous))
        cost += 0.6 * movement
        common = len(set(pitch % 12 for pitch in voicing) & set(pitch % 12 for pitch in previous))
        cost -= 0.4 * common
    return cost


def voice_chords(
    spans: Sequence[ChordSpan],
    melody: Sequence[MelodyNote],
    *,
    inner_voices: int = 2,
    style: VoicingStyle = DEFAULT_STYLE,
) -> list[list[int]]:
    """Choose an inner voicing for every chord, with voice leading between them.

    Greedy from the previous voicing rather than a full Viterbi: the cost is
    dominated by the immediately preceding chord, jazz comping genuinely is
    decided chord to chord, and this keeps the engine's latency where a UI
    wants it.
    """
    out: list[list[int]] = []
    previous: list[int] | None = None
    for span in spans:
        melody_pitch = _melody_pitch_at(melody, span.start, span.stop)
        sounding = _melody_pitches_in(melody, span.start, span.stop)
        # The ceiling follows the melody note that actually holds the bar, not
        # the lowest note in it: one passing low note should not force the
        # accompaniment down an octave for the whole chord.
        ceiling = min(INNER_HIGH, melody_pitch if melody_pitch is not None else INNER_HIGH)
        centre = max(54.0, min(70.0, (melody_pitch if melody_pitch else 68) - 9.0))
        omit = _omitted_intervals(span.chord, sounding)
        best: tuple[float, list[int]] | None = None
        # Try the full number of parts first and give notes up only if the
        # register genuinely cannot hold them, which happens when the melody is
        # low. A three-note voicing that fits beats a four-note one that does
        # not exist.
        for size in range(inner_voices, 0, -1):
            for rank, intervals in enumerate(pitch_class_sets(span.chord, style, size)):
                kept = [interval for interval in intervals if interval % 12 not in omit] or intervals
                for voicing in arrange(kept, span.chord.root, ceiling):
                    if _clash_with_melody(voicing, sounding):
                        continue
                    cost = _voicing_cost(voicing, previous, centre) + 0.35 * rank
                    if best is None or cost < best[0]:
                        best = (cost, voicing)
            if best is not None:
                break
        if best is None:
            best = (0.0, _last_resort(span.chord, ceiling, sounding))
        out.append(best[1])
        previous = best[1] or previous
    return out


def _last_resort(chord: JazzChord, ceiling: int, melody_pitches: Sequence[int]) -> list[int]:
    """One note, or none at all, when the register cannot hold a voicing.

    The register runs out when the melody sits very low, and the honest answer
    is then to play less rather than to play something wrong: an accompaniment
    note that beats against the tune is worse than no accompaniment note, and
    the bass is still holding the harmony underneath. This path used to skip
    the clash check entirely, which is exactly how a minor ninth against the
    melody got into the output of a tune transposed down an octave.
    """
    intervals = [interval for interval in (chord.third_interval, chord.seventh_interval, 0, 7) if interval is not None]
    for interval in intervals:
        for pitch in reversed(arrange([interval], chord.root, ceiling) or []):
            if not _clash_with_melody(pitch, melody_pitches):
                return pitch
    return []


def _omitted_intervals(chord: JazzChord, melody_pitches: Sequence[int]) -> set[int]:
    """Chord tones the melody forces out of the voicing.

    This is `classify_melody_note`'s repair advice, applied. A melody on the
    b13 of a dominant means the natural fifth has to go; a melody on the 4th of
    a dominant means the third does. Following the advice is what turns a
    flagged clash into an intended sound — and it has to consider every note
    sounding over the chord, not just the longest one, because it is usually
    the passing note that does the clashing.
    """
    omit: set[int] = set()
    for pitch in melody_pitches:
        verdict = classify_melody_note(chord, pitch % 12)
        if verdict.omit is not None:
            omit.add(verdict.omit % 12)
    return omit


# ---------------------------------------------------------------------------
# Bass
# ---------------------------------------------------------------------------


def bass_line(
    spans: Sequence[ChordSpan],
    *,
    walking: bool = True,
    beats_per_bar: int = 4,
) -> list[Note]:
    """Roots, with a chromatic or fifth approach into the next chord.

    Not a full walking line — that would need a swing feel and a rhythm the
    contract has no way to express — but the approach note is what makes a
    chord change sound like it was going somewhere, and it costs one note.
    """
    notes: list[Note] = []
    previous_pitch: int | None = None
    for index, span in enumerate(spans):
        pitch = _nearest(span.chord.bass_pc, previous_pitch if previous_pitch is not None else 45, BASS_LOW, BASS_HIGH)
        following = spans[index + 1].chord.bass_pc if index + 1 < len(spans) else None
        approach = walking and span.duration >= 2.0 and following is not None
        if approach:
            target = _nearest(following, pitch, BASS_LOW, BASS_HIGH)
            step = 1 if target >= pitch else -1
            approach_pitch = target - step
            if abs(approach_pitch - pitch) > 7:
                approach_pitch = pitch + (7 if target > pitch else -7)
            approach_pitch = max(BASS_LOW, min(BASS_HIGH, approach_pitch))
            notes.append(Note(pitch=pitch, start=round(span.start, 6), duration=round(span.duration - 1.0, 6), velocity=82))
            notes.append(Note(pitch=approach_pitch, start=round(span.stop - 1.0, 6), duration=1.0, velocity=70))
        else:
            notes.append(Note(pitch=pitch, start=round(span.start, 6), duration=round(span.duration, 6), velocity=82))
        previous_pitch = pitch
    return notes


#: Where a bass line wants to sit when nothing else decides. Without a pull
#: toward it, following the previous note alone lets the line ratchet upward
#: into the voicing and the texture loses its floor.
BASS_CENTRE = 43


def _nearest(pitch_class: int, near: int, low: int, high: int) -> int:
    candidates = [pitch for pitch in range(low, high + 1) if pitch % 12 == pitch_class % 12]
    if not candidates:
        return low + (pitch_class % 12)
    return min(candidates, key=lambda pitch: (abs(pitch - near) + 0.4 * abs(pitch - BASS_CENTRE), pitch))


# ---------------------------------------------------------------------------
# Assembling voices
# ---------------------------------------------------------------------------

_INNER_NAMES: tuple[VoiceName, ...] = ("alto", "tenor")


def _inner_name(index: int) -> VoiceName:
    return _INNER_NAMES[0] if index == 0 else _INNER_NAMES[1]


def build_voices(
    spans: Sequence[ChordSpan],
    melody: Sequence[MelodyNote],
    *,
    voice_count: int = 4,
    style: VoicingStyle = DEFAULT_STYLE,
    beats_per_bar: int = 4,
    origin: float = 0.0,
) -> list[Voice]:
    """Melody on top, bass underneath, comped voices in between."""
    voice_count = max(2, min(8, voice_count))
    inner_count = max(0, voice_count - 2)
    voicings = voice_chords(spans, melody, inner_voices=inner_count, style=style) if inner_count else []

    soprano = Voice(
        name="soprano",
        notes=[
            Note(pitch=pitch, start=round(origin + start, 6), duration=round(duration, 6), velocity=92)
            for start, pitch, duration in melody
        ],
    )
    bass = Voice(
        name="bass",
        notes=[
            note.model_copy(update={"start": round(origin + note.start, 6)})
            for note in bass_line(spans, walking=style.walking_bass, beats_per_bar=beats_per_bar)
        ],
    )

    inner: list[list[Note]] = [[] for _ in range(inner_count)]
    for span, voicing in zip(spans, voicings):
        # Voicings are written top-down into the inner parts, so the top inner
        # voice keeps its identity across chords even when a voicing has fewer
        # notes than there are parts. That continuity is the guide-tone line.
        ordered = list(reversed(voicing))
        for index in range(inner_count):
            if index >= len(ordered):
                continue
            inner[index].append(Note(
                pitch=ordered[index],
                start=round(origin + span.start, 6),
                duration=round(span.duration, 6),
                velocity=68,
            ))

    voices = [soprano]
    for index, notes in enumerate(inner):
        voices.append(Voice(name=_inner_name(index), notes=_merge(notes)))
    voices.append(bass)
    return voices


def _merge(notes: Sequence[Note]) -> list[Note]:
    """Tie repeated pitches across adjacent chords into one sustained note."""
    merged: list[Note] = []
    for note in notes:
        if merged and merged[-1].pitch == note.pitch and abs(merged[-1].start + merged[-1].duration - note.start) < 1e-6:
            merged[-1] = merged[-1].model_copy(
                update={"duration": round(merged[-1].duration + note.duration, 6)}
            )
        else:
            merged.append(note)
    return merged
