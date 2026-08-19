"""SATB voicing constraints and voice-leading defect detection.

These are the primitives the whole project is measured with, so they are kept
small, explicit and unit-tested. A defect detector that is subtly wrong makes
every downstream number meaningless — which is exactly how v1 ended up unable
to tell whether its model was better than its rules.

Convention throughout: voices are ordered ``(soprano, alto, tenor, bass)`` =
index 0..3, highest to lowest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .chords import ChordLabel
from .pitch import Key, is_perfect_fifth, is_perfect_octave, motion_type

VOICE_NAMES = ("soprano", "alto", "tenor", "bass")
SOPRANO, ALTO, TENOR, BASS = 0, 1, 2, 3

#: Comfortable chorale ranges (inclusive MIDI). Calibrated against the music21
#: Bach corpus: each bound sits at roughly the 0.1/99.9 percentile of the notes
#: Bach actually writes for that part, so the Bach oracle scores near zero here
#: while genuinely out-of-tessitura writing is still caught.
VOICE_RANGES: dict[int, tuple[int, int]] = {
    SOPRANO: (60, 81),  # C4 - A5
    ALTO: (53, 74),     # F3 - D5
    TENOR: (47, 69),    # B2 - A4
    BASS: (38, 64),     # D2 - E4
}

#: Slightly tighter ranges the rule engine prefers to stay inside.
PREFERRED_RANGES: dict[int, tuple[int, int]] = {
    SOPRANO: (60, 79),
    ALTO: (55, 72),
    TENOR: (48, 67),
    BASS: (40, 62),
}

#: Max semitones between adjacent upper voices; bass-tenor may open to a 12th.
MAX_SPACING = {(SOPRANO, ALTO): 12, (ALTO, TENOR): 12, (TENOR, BASS): 19}

#: Melodic intervals a voice may leap, in semitones. Anything larger, plus every
#: augmented interval and the sevenths, counts as a defect.
MAX_COMFORTABLE_LEAP = 7
ALLOWED_LARGE_LEAPS = frozenset({8, 12})  # minor sixth and octave
FORBIDDEN_MELODIC = frozenset({6, 10, 11, 13, 14})  # tritone, 7ths, 9ths+


@dataclass(frozen=True)
class Defect:
    """One rule violation, located in time and attributed to specific voices."""

    kind: str
    severity: str  # "info" | "warning" | "error"
    offset: float
    voices: tuple[int, ...]
    message: str


Sonority = tuple[int | None, ...]


@dataclass
class VoicedTexture:
    """Four voices sampled on a fixed grid.

    `grid[t][v]` is the sounding MIDI pitch of voice v at step t, or None for a
    rest. `onsets[t][v]` marks a re-articulation (as opposed to a held note),
    which is what distinguishes a tie from a repeated note.
    """

    grid: list[Sonority]
    onsets: list[tuple[bool, ...]] = field(default_factory=list)
    step: float = 0.25
    start: float = 0.0

    def __post_init__(self) -> None:
        if not self.onsets:
            self.onsets = [tuple(True for _ in row) for row in self.grid]

    @property
    def n_voices(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def offset_of(self, index: int) -> float:
        return self.start + index * self.step

    def changes(self) -> list[tuple[int, Sonority]]:
        """Distinct successive verticals, with the grid index each begins at.

        Voice-leading rules apply between successive *sonorities*, not between
        successive grid steps: a held chord repeated for four sixteenths is one
        chord, and no voice moves, so no parallel can occur.
        """
        out: list[tuple[int, Sonority]] = []
        prev: Sonority | None = None
        for index, row in enumerate(self.grid):
            if prev is None or row != prev:
                out.append((index, row))
                prev = row
        return out

    def voice_line(self, voice: int) -> list[tuple[int, int]]:
        """(grid index, pitch) for each note the voice sounds, skipping holds."""
        out: list[tuple[int, int]] = []
        prev: int | None = None
        for index, row in enumerate(self.grid):
            pitch = row[voice]
            if pitch is None:
                prev = None
                continue
            if pitch != prev:
                out.append((index, pitch))
            prev = pitch
        return out


# ---------------------------------------------------------------------------
# Simultaneity checks
# ---------------------------------------------------------------------------


def find_voice_crossings(sonority: Sonority) -> list[tuple[int, int]]:
    """Voice pairs where a nominally lower voice sounds above a higher one."""
    out = []
    pitches = list(sonority)
    for upper in range(len(pitches) - 1):
        for lower in range(upper + 1, len(pitches)):
            a, b = pitches[upper], pitches[lower]
            if a is None or b is None:
                continue
            if b > a:
                out.append((upper, lower))
    return out


def find_spacing_errors(sonority: Sonority) -> list[tuple[int, int, int]]:
    """(upper, lower, gap) for adjacent voice pairs spaced too widely."""
    out = []
    for (upper, lower), limit in MAX_SPACING.items():
        if upper >= len(sonority) or lower >= len(sonority):
            continue
        a, b = sonority[upper], sonority[lower]
        if a is None or b is None:
            continue
        gap = a - b
        if gap > limit:
            out.append((upper, lower, gap))
    return out


def find_range_violations(sonority: Sonority, ranges: dict[int, tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    """(voice, pitch) for every note outside its part's range."""
    ranges = ranges or VOICE_RANGES
    out = []
    for voice, pitch in enumerate(sonority):
        if pitch is None or voice not in ranges:
            continue
        low, high = ranges[voice]
        if pitch < low or pitch > high:
            out.append((voice, pitch))
    return out


# ---------------------------------------------------------------------------
# Motion checks
# ---------------------------------------------------------------------------


def find_parallels(prev: Sonority, curr: Sonority) -> list[tuple[int, int, str]]:
    """Consecutive perfect fifths/octaves between two sonorities.

    Returns (upper_voice, lower_voice, kind) with kind in
    {"parallel_fifths", "parallel_octaves", "contrary_fifths", "contrary_octaves"}.

    Both voices must actually move: a sustained perfect fifth is not a parallel,
    and a compound fifth followed by a simple one still is.
    """
    out = []
    n = min(len(prev), len(curr))
    for high in range(n - 1):
        for low in range(high + 1, n):
            p_hi, p_lo = prev[high], prev[low]
            c_hi, c_lo = curr[high], curr[low]
            if None in (p_hi, p_lo, c_hi, c_lo):
                continue
            if p_hi == c_hi or p_lo == c_lo:
                continue  # oblique or static: no parallel possible

            p_upper, p_lower = max(p_hi, p_lo), min(p_hi, p_lo)
            c_upper, c_lower = max(c_hi, c_lo), min(c_hi, c_lo)

            was_fifth = is_perfect_fifth(p_lower, p_upper)
            now_fifth = is_perfect_fifth(c_lower, c_upper)
            was_octave = is_perfect_octave(p_lower, p_upper)
            now_octave = is_perfect_octave(c_lower, c_upper)

            motion = motion_type(p_lo, c_lo, p_hi, c_hi)
            if was_fifth and now_fifth:
                out.append((high, low, "parallel_fifths" if motion in ("parallel", "similar") else "contrary_fifths"))
            elif was_octave and now_octave:
                out.append((high, low, "parallel_octaves" if motion in ("parallel", "similar") else "contrary_octaves"))
    return out


def find_direct_perfects(prev: Sonority, curr: Sonority, *, outer_only: bool = True) -> list[tuple[int, int, str]]:
    """Direct (hidden/exposed) fifths and octaves.

    Similar motion into a perfect fifth or octave where the upper voice leaps.
    Restricted to the outer voices by default, which is where the effect is
    audible and where the textbook rule bites.
    """
    pairs = [(SOPRANO, BASS)] if outer_only else [
        (h, l) for h in range(len(prev) - 1) for l in range(h + 1, len(prev))
    ]
    out = []
    for high, low in pairs:
        if high >= len(prev) or low >= len(prev):
            continue
        p_hi, p_lo, c_hi, c_lo = prev[high], prev[low], curr[high], curr[low]
        if None in (p_hi, p_lo, c_hi, c_lo):
            continue
        if motion_type(p_lo, c_lo, p_hi, c_hi) != "similar":
            continue
        if abs(c_hi - p_hi) <= 2:
            continue  # upper voice moves by step: allowed
        c_upper, c_lower = max(c_hi, c_lo), min(c_hi, c_lo)
        if is_perfect_fifth(c_lower, c_upper):
            out.append((high, low, "direct_fifths"))
        elif is_perfect_octave(c_lower, c_upper):
            out.append((high, low, "direct_octaves"))
    return out


def find_overlaps(prev: Sonority, curr: Sonority) -> list[tuple[int, int]]:
    """Voice overlap: a voice moves past where an adjacent voice just was."""
    out = []
    n = min(len(prev), len(curr))
    for upper in range(n - 1):
        lower = upper + 1
        p_hi, p_lo, c_hi, c_lo = prev[upper], prev[lower], curr[upper], curr[lower]
        if None in (p_hi, p_lo, c_hi, c_lo):
            continue
        if c_lo > p_hi:
            out.append((lower, upper))
        if c_hi < p_lo:
            out.append((upper, lower))
    return out


def melodic_defect(prev_pitch: int, curr_pitch: int) -> str | None:
    """Classify an awkward melodic interval, or None if it is idiomatic."""
    leap = abs(curr_pitch - prev_pitch)
    if leap == 0 or leap <= 2:
        return None
    if leap in FORBIDDEN_MELODIC:
        return "awkward_melodic_interval"
    if leap <= MAX_COMFORTABLE_LEAP:
        return None
    if leap in ALLOWED_LARGE_LEAPS:
        return None
    return "large_leap"


# ---------------------------------------------------------------------------
# Tendency-tone resolution
# ---------------------------------------------------------------------------


def dominant_target(chord: ChordLabel) -> int | None:
    """The degree this chord resolves to if it has dominant function, else None.

    Covers V, V7, viio, viio7, viiø7 and every applied version of them.
    """
    base = 0 if chord.applied_to is None else chord.applied_to
    root_from_base = (chord.relative_root - base) % 12
    if root_from_base == 7 and chord.quality in ("maj", "dom7"):
        return base
    if root_from_base == 11 and chord.quality in ("dim", "dim7", "halfdim7"):
        return base
    return None


def leading_tone_pc_of(chord: ChordLabel) -> int | None:
    """Tonic-relative pitch class of the chord's leading tone, if it has one."""
    target = dominant_target(chord)
    if target is None:
        return None
    return (target + 11) % 12


def chordal_seventh_pc(chord: ChordLabel) -> int | None:
    return chord.seventh_relative_pc


def find_unresolved_leading_tones(
    prev_son: Sonority,
    curr_son: Sonority,
    prev_chord: ChordLabel,
    curr_chord: ChordLabel,
    key: Key,
) -> list[tuple[int, str]]:
    """Voices that abandon a leading tone at a dominant-to-tonic resolution.

    Only fires when the dominant actually resolves to its target, so deceptive
    and elided progressions are not penalised for the wrong reason. Inner voices
    that fall to the fifth of the target get the standard "frustrated leading
    tone" exemption, downgraded to info.
    """
    target = dominant_target(prev_chord)
    if target is None:
        return []
    resolves_to_target = curr_chord.relative_root % 12 == target % 12 and curr_chord.quality in (
        "maj", "min", "maj7", "min7", "dom7", "minmaj7"
    )
    if not resolves_to_target:
        return []

    lt_pc = (target + 11) % 12
    tonic_pc = target % 12
    fifth_pc = (target + 7) % 12
    out = []
    for voice, pitch in enumerate(prev_son):
        if pitch is None or curr_son[voice] is None:
            continue
        if key.to_relative(pitch % 12) != lt_pc:
            continue
        moved = curr_son[voice] - pitch
        dest = key.to_relative(curr_son[voice] % 12)
        if moved == 1 and dest == tonic_pc:
            continue
        if voice in (ALTO, TENOR) and dest == fifth_pc and abs(moved) <= 4:
            out.append((voice, "frustrated_leading_tone"))
            continue
        out.append((voice, "unresolved_leading_tone"))
    return out


def find_unresolved_sevenths(
    prev_son: Sonority,
    curr_son: Sonority,
    prev_chord: ChordLabel,
    curr_chord: ChordLabel,
    key: Key,
) -> list[int]:
    """Voices that fail to resolve a chordal seventh downward by step."""
    seventh = chordal_seventh_pc(prev_chord)
    if seventh is None:
        return []
    if prev_chord.key() == curr_chord.key():
        return []  # same chord still sounding: resolution not yet due
    curr_seventh = chordal_seventh_pc(curr_chord)
    out = []
    for voice, pitch in enumerate(prev_son):
        if pitch is None or curr_son[voice] is None:
            continue
        if key.to_relative(pitch % 12) != seventh:
            continue
        moved = curr_son[voice] - pitch
        if moved in (-1, -2):
            continue
        if moved == 0 and curr_seventh is not None and key.to_relative(curr_son[voice] % 12) == curr_seventh:
            continue  # seventh carried over into the next chord
        out.append(voice)
    return out


def find_doubled_leading_tone(sonority: Sonority, chord: ChordLabel, key: Key) -> list[int]:
    """Voices participating in a doubled leading tone of a dominant chord."""
    lt = leading_tone_pc_of(chord)
    if lt is None:
        return []
    holders = [v for v, p in enumerate(sonority) if p is not None and key.to_relative(p % 12) == lt]
    return holders if len(holders) > 1 else []


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

_SEVERITY = {
    "parallel_fifths": "error",
    "parallel_octaves": "error",
    "contrary_fifths": "warning",
    "contrary_octaves": "warning",
    "direct_fifths": "warning",
    "direct_octaves": "warning",
    "voice_crossing": "warning",
    "voice_overlap": "info",
    "spacing": "warning",
    "range": "error",
    "unresolved_leading_tone": "warning",
    "frustrated_leading_tone": "info",
    "unresolved_seventh": "warning",
    "doubled_leading_tone": "warning",
    "large_leap": "info",
    "awkward_melodic_interval": "warning",
}


def analyze_texture(
    texture: VoicedTexture,
    key: Key,
    chords: Sequence[ChordLabel | None] | None = None,
) -> list[Defect]:
    """Every voice-leading defect in a voiced texture.

    `chords` is optional per-grid-step harmonic context; the tendency-tone rules
    are skipped without it, everything else still applies.
    """
    defects: list[Defect] = []
    changes = texture.changes()

    for index, sonority in changes:
        offset = texture.offset_of(index)
        for upper, lower in find_voice_crossings(sonority):
            defects.append(Defect(
                "voice_crossing", _SEVERITY["voice_crossing"], offset, (upper, lower),
                f"{VOICE_NAMES[lower]} sounds above {VOICE_NAMES[upper]}",
            ))
        for upper, lower, gap in find_spacing_errors(sonority):
            defects.append(Defect(
                "spacing", _SEVERITY["spacing"], offset, (upper, lower),
                f"{gap} semitones between {VOICE_NAMES[upper]} and {VOICE_NAMES[lower]}",
            ))
        for voice, pitch in find_range_violations(sonority):
            low, high = VOICE_RANGES[voice]
            defects.append(Defect(
                "range", _SEVERITY["range"], offset, (voice,),
                f"{VOICE_NAMES[voice]} sings {pitch}, outside {low}-{high}",
            ))
        if chords is not None and index < len(chords) and chords[index] is not None:
            doubled = find_doubled_leading_tone(sonority, chords[index], key)
            if doubled:
                defects.append(Defect(
                    "doubled_leading_tone", _SEVERITY["doubled_leading_tone"], offset, tuple(doubled),
                    "leading tone doubled in a dominant-function chord",
                ))

    for (prev_index, prev_son), (curr_index, curr_son) in zip(changes, changes[1:]):
        offset = texture.offset_of(curr_index)
        for high, low, kind in find_parallels(prev_son, curr_son):
            defects.append(Defect(
                kind, _SEVERITY[kind], offset, (high, low),
                f"{kind.replace('_', ' ')} between {VOICE_NAMES[high]} and {VOICE_NAMES[low]}",
            ))
        for high, low, kind in find_direct_perfects(prev_son, curr_son):
            defects.append(Defect(
                kind, _SEVERITY[kind], offset, (high, low),
                f"{kind.replace('_', ' ')} in the outer voices",
            ))
        for moving, passed in find_overlaps(prev_son, curr_son):
            defects.append(Defect(
                "voice_overlap", _SEVERITY["voice_overlap"], offset, (moving, passed),
                f"{VOICE_NAMES[moving]} overlaps {VOICE_NAMES[passed]}",
            ))
        if chords is None:
            continue
        prev_chord = chords[prev_index] if prev_index < len(chords) else None
        curr_chord = chords[curr_index] if curr_index < len(chords) else None
        if prev_chord is None or curr_chord is None:
            continue
        for voice, kind in find_unresolved_leading_tones(prev_son, curr_son, prev_chord, curr_chord, key):
            defects.append(Defect(
                kind, _SEVERITY[kind], offset, (voice,),
                f"{VOICE_NAMES[voice]} leaves the leading tone",
            ))
        for voice in find_unresolved_sevenths(prev_son, curr_son, prev_chord, curr_chord, key):
            defects.append(Defect(
                "unresolved_seventh", _SEVERITY["unresolved_seventh"], offset, (voice,),
                f"{VOICE_NAMES[voice]} does not resolve the chordal seventh down by step",
            ))

    for voice in range(texture.n_voices):
        line = texture.voice_line(voice)
        for (_, prev_pitch), (index, pitch) in zip(line, line[1:]):
            kind = melodic_defect(prev_pitch, pitch)
            if kind:
                defects.append(Defect(
                    kind, _SEVERITY[kind], texture.offset_of(index), (voice,),
                    f"{VOICE_NAMES[voice]} moves by {abs(pitch - prev_pitch)} semitones",
                ))

    defects.sort(key=lambda d: (d.offset, d.kind, d.voices))
    return defects


def count_chord_changes(texture: VoicedTexture) -> int:
    """Number of sonority transitions, the denominator for per-100-chord rates."""
    return max(0, len(texture.changes()) - 1)


def texture_from_voices(lines: Sequence[Sequence[int | None]], step: float = 0.25, start: float = 0.0) -> VoicedTexture:
    """Build a VoicedTexture from parallel per-voice pitch grids."""
    length = max((len(line) for line in lines), default=0)
    grid: list[Sonority] = []
    for t in range(length):
        grid.append(tuple(line[t] if t < len(line) else None for line in lines))
    return VoicedTexture(grid=grid, step=step, start=start)


def iter_sonorities(texture: VoicedTexture) -> Iterable[Sonority]:
    return iter(texture.grid)
