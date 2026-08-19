"""Pitch, interval and key primitives.

Everything here is plain integer arithmetic on MIDI pitches and pitch classes so
it is fast, dependency-free and testable. music21 is used only at corpus-load
time, never on the inference path.

The central v1 correction lives in this module: `normalize_pitches` /
`Key.to_relative`. v1 fed the network ABSOLUTE pitch classes while training it on
TONIC-RELATIVE scale degrees, so the model had to induce the tonic across all 12
transpositions from ~400 chorales given only an `is_minor` bit. Here the tonic is
removed from the representation by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Pitch classes and intervals
# ---------------------------------------------------------------------------

PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

#: Preferred spellings when the tonic is a flat key; used only for display.
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
#: Natural minor. Raised 6/7 are handled as chromatic alterations, not a scale.
NATURAL_MINOR = (0, 2, 3, 5, 7, 8, 10)
HARMONIC_MINOR = (0, 2, 3, 5, 7, 8, 11)

PERFECT_FIFTH = 7
PERFECT_FOURTH = 5
OCTAVE = 12


def pc(pitch: int) -> int:
    """Pitch class of a MIDI pitch."""
    return pitch % 12


def interval_class(a: int, b: int) -> int:
    """Unordered interval class (0-6) between two pitches, octave-reduced."""
    d = abs(a - b) % 12
    return min(d, 12 - d)


def is_perfect_fifth(lower: int, upper: int) -> bool:
    """True if `upper` is a perfect fifth (or compound fifth) above `lower`.

    A perfect *twelfth* counts; a perfect fourth does not. Callers must pass the
    voices in sounding order.
    """
    if upper < lower:
        return False
    return (upper - lower) % 12 == PERFECT_FIFTH


def is_perfect_octave(lower: int, upper: int) -> bool:
    """True for a unison, octave or any compound octave between two pitches."""
    return (upper - lower) % 12 == 0


def signed_step(a: int, b: int) -> int:
    """Directed semitone motion from `a` to `b`."""
    return b - a


def motion_type(low_from: int, low_to: int, high_from: int, high_to: int) -> str:
    """Classify the relative motion of two voices between two chords.

    Returns one of "parallel", "similar", "contrary", "oblique", "static".
    "parallel" means same direction *and* the harmonic interval is preserved,
    which is the distinction that matters for parallel-fifth rules.
    """
    d_low = low_to - low_from
    d_high = high_to - high_from
    if d_low == 0 and d_high == 0:
        return "static"
    if d_low == 0 or d_high == 0:
        return "oblique"
    if (d_low > 0) != (d_high > 0):
        return "contrary"
    if d_low == d_high:
        return "parallel"
    return "similar"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Key:
    """A tonic pitch class plus a mode.

    `tonic` is 0-11 (C = 0). `mode` is "major" or "minor".
    """

    tonic: int
    mode: str

    def __post_init__(self) -> None:
        if not 0 <= self.tonic <= 11:
            raise ValueError(f"tonic must be a pitch class 0-11, got {self.tonic}")
        if self.mode not in ("major", "minor"):
            raise ValueError(f"mode must be 'major' or 'minor', got {self.mode!r}")

    @property
    def is_minor(self) -> bool:
        return self.mode == "minor"

    @property
    def scale(self) -> tuple[int, ...]:
        return NATURAL_MINOR if self.is_minor else MAJOR_SCALE

    @property
    def leading_tone_pc(self) -> int:
        """Absolute pitch class of the leading tone.

        In minor this is the RAISED seventh: chorale practice raises 7 in every
        dominant-function chord, and the subtonic (b7) belongs to the natural
        minor melodic descent rather than to the dominant.
        """
        return (self.tonic + 11) % 12

    def to_relative(self, pitch_class: int) -> int:
        """Absolute pitch class -> scale degree in semitones above the tonic."""
        return (pitch_class - self.tonic) % 12

    def to_absolute(self, relative_pc: int) -> int:
        """Semitones above the tonic -> absolute pitch class."""
        return (relative_pc + self.tonic) % 12

    def degree_of(self, pitch_class: int) -> int | None:
        """Diatonic degree index (0-6) of a pitch class, or None if chromatic."""
        rel = self.to_relative(pitch_class)
        scale = self.scale
        return scale.index(rel) if rel in scale else None

    def name(self) -> str:
        names = FLAT_NAMES if self.tonic in (1, 3, 5, 8, 10) else PITCH_CLASS_NAMES
        return f"{names[self.tonic]} {self.mode}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name()


#: The key every piece is normalized into for learning. C major / C minor keeps
#: the two modes in the same register, so a single mode flag is all the model
#: needs to disambiguate them.
NORMALIZED_TONIC = 0


def normalization_shift(key: Key, target_tonic: int = NORMALIZED_TONIC) -> int:
    """Semitone shift that moves `key.tonic` onto `target_tonic`.

    The shift is chosen in [-6, +5] so the piece stays as close as possible to
    its original register: transposing G major down a fifth and up a fourth give
    the same key but different tessituras, and SATB ranges are absolute.
    """
    raw = (target_tonic - key.tonic) % 12
    return raw - 12 if raw > 5 else raw


def normalize_pitches(pitches: Iterable[int], key: Key, target_tonic: int = NORMALIZED_TONIC) -> list[int]:
    """Transpose absolute MIDI pitches so that `key.tonic` lands on `target_tonic`."""
    shift = normalization_shift(key, target_tonic)
    return [p + shift for p in pitches]


def denormalize_pitches(pitches: Iterable[int], key: Key, target_tonic: int = NORMALIZED_TONIC) -> list[int]:
    """Inverse of `normalize_pitches`."""
    shift = normalization_shift(key, target_tonic)
    return [p - shift for p in pitches]


# ---------------------------------------------------------------------------
# Krumhansl-Schmuckler key finding
# ---------------------------------------------------------------------------

# Krumhansl-Kessler probe-tone profiles.
_KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
_KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)


def _correlate(profile: Sequence[float], weights: Sequence[float]) -> float:
    n = len(profile)
    mp = sum(profile) / n
    mw = sum(weights) / n
    num = sum((profile[i] - mp) * (weights[i] - mw) for i in range(n))
    dp = sum((profile[i] - mp) ** 2 for i in range(n)) ** 0.5
    dw = sum((weights[i] - mw) ** 2 for i in range(n)) ** 0.5
    if dp == 0 or dw == 0:
        return 0.0
    return num / (dp * dw)


def detect_key(
    pitch_durations: Sequence[tuple[int, float]],
    *,
    final_bonus_pitch: int | None = None,
) -> tuple[Key, float]:
    """Duration-weighted Krumhansl-Schmuckler key finding.

    `pitch_durations` is a sequence of (midi_pitch, duration_in_quarters).
    Returns (key, confidence) where confidence is the normalized margin between
    the best and second-best candidate, clipped to [0, 1].

    `final_bonus_pitch` nudges the result toward keys whose tonic triad contains
    the final melody note. Chorale phrases end on a chord tone of the tonic
    overwhelmingly often, and unaided KS confuses relative major/minor.
    """
    weights = [0.0] * 12
    for pitch, duration in pitch_durations:
        weights[pitch % 12] += float(duration)
    if sum(weights) == 0:
        return Key(0, "major"), 0.0

    scored: list[tuple[float, Key]] = []
    for tonic in range(12):
        rotated = [weights[(tonic + i) % 12] for i in range(12)]
        for mode, profile in (("major", _KK_MAJOR), ("minor", _KK_MINOR)):
            score = _correlate(profile, rotated)
            if final_bonus_pitch is not None:
                triad = (0, 4, 7) if mode == "major" else (0, 3, 7)
                if (final_bonus_pitch - tonic) % 12 in triad:
                    score += 0.08
                if (final_bonus_pitch - tonic) % 12 == 0:
                    score += 0.04
            scored.append((score, Key(tonic, mode)))

    scored.sort(key=lambda item: (-item[0], item[1].tonic, item[1].mode))
    best_score, best_key = scored[0]
    runner_up = scored[1][0]
    confidence = max(0.0, min(1.0, (best_score - runner_up) * 4.0))
    return best_key, confidence
