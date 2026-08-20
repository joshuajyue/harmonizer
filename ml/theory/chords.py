"""A chord vocabulary rich enough to actually represent a Bach chorale.

This module is the direct answer to v1 failure #3. v1's label space was seven
diatonic triads, and `extract_real_chord_labels` projected every seventh chord,
secondary dominant, borrowed chord and suspension onto the nearest triad with a
crude +1/-1 pitch-class vote. The network was then trained to fit those
corrupted labels — and most of what makes the harmony *Bach* lives precisely in
what the projection destroyed.

Chords here are represented tonic-relatively as
``(root_pc_above_tonic, quality, inversion)`` plus an optional applied target,
which covers triads, all five common seventh types, every inversion, secondary
dominants and leading-tone chords, and borrowed/chromatic roots.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

from .pitch import MAJOR_SCALE, NATURAL_MINOR, Key

# ---------------------------------------------------------------------------
# Qualities
# ---------------------------------------------------------------------------

#: Interval content of each quality, in semitones above the root.
QUALITY_TEMPLATES: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "halfdim7": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9),
    "minmaj7": (0, 3, 7, 11),
}

TRIAD_QUALITIES = frozenset({"maj", "min", "dim", "aug"})
SEVENTH_QUALITIES = frozenset({"dom7", "maj7", "min7", "halfdim7", "dim7", "minmaj7"})

#: Qualities whose roman numeral is written in lower case.
_MINOR_ISH = frozenset({"min", "dim", "min7", "halfdim7", "dim7", "minmaj7"})

#: Figured-bass suffix by (is_seventh, inversion).
_FIGURES = {
    (False, 0): "",
    (False, 1): "6",
    (False, 2): "64",
    (True, 0): "7",
    (True, 1): "65",
    (True, 2): "43",
    (True, 3): "42",
}

_QUALITY_SUFFIX = {
    "dim": "o",
    "aug": "+",
    "halfdim7": "\u00f8",
    "dim7": "o",
    "maj7": "M",
    "minmaj7": "M",
}

#: Roman numeral stems for each semitone above the tonic, per mode.
_NUMERAL_MAJOR = {
    0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
    6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII",
}
# In minor the natural-minor collection is the reference, so pc 10 is the plain
# subtonic VII and pc 11 is the raised leading-tone VII (disambiguated by case:
# "VII" vs "viio"). Chorales treat both as normal, not as chromatic alterations.
_NUMERAL_MINOR = {
    0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV",
    6: "#IV", 7: "V", 8: "VI", 9: "#VI", 10: "VII", 11: "VII",
}


def numeral_stem(relative_root: int, mode: str) -> str:
    table = _NUMERAL_MINOR if mode == "minor" else _NUMERAL_MAJOR
    return table[relative_root % 12]


def chord_size(quality: str) -> int:
    return len(QUALITY_TEMPLATES[quality])


@lru_cache(maxsize=4096)
def chord_pitch_classes(relative_root: int, quality: str) -> tuple[int, ...]:
    """Tonic-relative pitch classes of a chord, root first."""
    return tuple((relative_root + i) % 12 for i in QUALITY_TEMPLATES[quality])


# ---------------------------------------------------------------------------
# Chord labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChordLabel:
    """A chord expressed relative to the prevailing key.

    `relative_root` is semitones above the tonic, so the label is transposition
    invariant — the whole point of the tonic-relative representation.
    """

    relative_root: int
    quality: str
    inversion: int = 0
    applied_to: int | None = None

    def __post_init__(self) -> None:
        if self.quality not in QUALITY_TEMPLATES:
            raise ValueError(f"unknown quality {self.quality!r}")
        max_inv = chord_size(self.quality) - 1
        if not 0 <= self.inversion <= max_inv:
            raise ValueError(f"inversion {self.inversion} out of range for {self.quality}")

    @property
    def is_seventh(self) -> bool:
        return self.quality in SEVENTH_QUALITIES

    @property
    def pitch_classes(self) -> tuple[int, ...]:
        return chord_pitch_classes(self.relative_root, self.quality)

    @property
    def bass_relative_pc(self) -> int:
        return self.pitch_classes[self.inversion]

    @property
    def seventh_relative_pc(self) -> int | None:
        return self.pitch_classes[3] if self.is_seventh else None

    @property
    def third_relative_pc(self) -> int:
        return self.pitch_classes[1]

    def absolute_root(self, key: Key) -> int:
        return (self.relative_root + key.tonic) % 12

    def absolute_pitch_classes(self, key: Key) -> tuple[int, ...]:
        return tuple((p + key.tonic) % 12 for p in self.pitch_classes)

    def roman(self, mode: str) -> str:
        """Display string, e.g. "V65", "bII6", "V7/V", "viio7"."""
        if self.applied_to is not None:
            target_mode = _applied_target_mode(self.applied_to, mode)
            local_root = (self.relative_root - self.applied_to) % 12
            stem = numeral_stem(local_root, target_mode)
        else:
            stem = numeral_stem(self.relative_root, mode)

        # Case the numeral by quality, leaving any accidental prefix alone
        # (bII stays bII, never BII).
        accidental = stem[0] if stem and stem[0] in "b#" else ""
        body = stem[len(accidental):]
        body = body.lower() if self.quality in _MINOR_ISH else body.upper()
        stem = accidental + body

        text = stem + _QUALITY_SUFFIX.get(self.quality, "")
        text += _FIGURES[(self.is_seventh, self.inversion)]
        if self.applied_to is not None:
            target_stem = numeral_stem(self.applied_to, mode)
            target_quality = _diatonic_quality(self.applied_to, mode)
            if target_quality in _MINOR_ISH:
                target_stem = target_stem.lower()
            text += "/" + target_stem
        return text

    def contract_quality(self) -> str:
        """Quality string as used by `contracts.schema.Chord`."""
        return self.quality

    def key(self) -> tuple[int, str, int, int]:
        """Hashable identity used for style statistics."""
        return (self.relative_root, self.quality, self.inversion, -1 if self.applied_to is None else self.applied_to)


def _diatonic_quality(relative_root: int, mode: str) -> str:
    """Quality of the diatonic triad on a scale degree, for /x target naming."""
    table = _MINOR_TRIAD_QUALITY if mode == "minor" else _MAJOR_TRIAD_QUALITY
    return table.get(relative_root % 12, "maj")


def _applied_target_mode(target: int, mode: str) -> str:
    """Whether a tonicized degree behaves as a major or minor local tonic."""
    return "minor" if _diatonic_quality(target, mode) in _MINOR_ISH else "major"


_MAJOR_TRIAD_QUALITY = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
_MINOR_TRIAD_QUALITY = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "maj", 8: "maj", 10: "maj", 11: "dim"}


# ---------------------------------------------------------------------------
# Applied / secondary chords
# ---------------------------------------------------------------------------

#: Degrees that can be tonicized in each mode (never the tonic itself).
APPLIED_TARGETS = {
    "major": (2, 4, 5, 7, 9),
    "minor": (3, 5, 7, 8, 10),
}


def diatonic_collection(mode: str) -> frozenset[int]:
    """Pitch classes treated as in-key, tonic-relative.

    Minor includes the raised seventh: the dominant and leading-tone chords are
    ordinary in a minor-key chorale, not secondary function.
    """
    if mode == "minor":
        return frozenset(NATURAL_MINOR) | {11}
    return frozenset(MAJOR_SCALE)


def infer_applied_target(relative_root: int, quality: str, mode: str) -> int | None:
    """Return the degree this chord tonicizes, or None if it is a native chord.

    A chord is applied when (a) it has dominant or leading-tone quality, (b) its
    root sits a fifth above / semitone below a tonicizable degree, and (c) it
    introduces a pitch class foreign to the key. Condition (c) is what stops
    plain diatonic chords being relabelled as secondary function.
    """
    pcs = set(chord_pitch_classes(relative_root, quality))
    if pcs <= diatonic_collection(mode):
        return None

    candidates: list[int] = []
    for target in APPLIED_TARGETS[mode]:
        if quality in ("maj", "dom7") and relative_root == (target + 7) % 12:
            candidates.append(target)
        elif quality in ("dim", "dim7", "halfdim7") and relative_root == (target + 11) % 12:
            candidates.append(target)
    if not candidates:
        return None
    # Prefer tonicizing V, then the closest degree on the circle of fifths.
    candidates.sort(key=lambda t: (t != 7, min((t - 7) % 12, (7 - t) % 12)))
    return candidates[0]


def make_chord(relative_root: int, quality: str, inversion: int, mode: str) -> ChordLabel:
    """Build a ChordLabel with the applied target inferred automatically."""
    return ChordLabel(
        relative_root=relative_root % 12,
        quality=quality,
        inversion=inversion,
        applied_to=infer_applied_target(relative_root % 12, quality, mode),
    )


# ---------------------------------------------------------------------------
# Analysis: voiced pitches -> chord label
# ---------------------------------------------------------------------------

#: Qualities considered when analysing real music, cheapest-first.
_ANALYSIS_QUALITIES = ("maj", "min", "dim", "dom7", "min7", "halfdim7", "dim7", "maj7", "aug", "minmaj7")


def analyze_chord(
    pitches: Sequence[int],
    key: Key,
    *,
    bass: int | None = None,
) -> ChordLabel | None:
    """Best chord label for a set of sounding absolute MIDI pitches.

    Unlike v1's +1/-1 diatonic vote this searches the full vocabulary, tolerates
    non-chord tones with an explicit penalty, and reads the inversion off the
    bass. Returns None only when nothing sounds.
    """
    if not pitches:
        return None
    bass_pitch = min(pitches) if bass is None else bass
    bass_pc = bass_pitch % 12
    sounding = {p % 12 for p in pitches}
    diatonic = diatonic_collection(key.mode)

    best: tuple[float, ChordLabel] | None = None
    for root_abs in range(12):
        rel_root = key.to_relative(root_abs)
        for quality in _ANALYSIS_QUALITIES:
            template = set(chord_pitch_classes(rel_root, quality))
            template_abs = {(p + key.tonic) % 12 for p in template}
            matched = sounding & template_abs
            extra = sounding - template_abs
            missing = template_abs - sounding
            if not matched:
                continue
            score = 2.0 * len(matched) - 3.0 * len(extra) - 1.0 * len(missing)
            # A chord whose root or third is absent is usually the wrong root.
            if (rel_root + key.tonic) % 12 not in sounding:
                score -= 1.5
            if bass_pc in template_abs:
                score += 1.0
            if bass_pc == (rel_root + key.tonic) % 12:
                score += 0.6
            # Prefer explanations that stay inside the key.
            score -= 0.35 * len(template - diatonic)
            # Break ties toward the simpler (triadic) reading.
            score -= 0.05 * len(template)
            if best is None or score > best[0] + 1e-9:
                inversion = _inversion_for(rel_root, quality, bass_pc, key)
                best = (score, make_chord(rel_root, quality, inversion, key.mode))

    return best[1] if best else None


def _inversion_for(rel_root: int, quality: str, bass_pc: int, key: Key) -> int:
    members = [(p + key.tonic) % 12 for p in chord_pitch_classes(rel_root, quality)]
    return members.index(bass_pc) if bass_pc in members else 0


# ---------------------------------------------------------------------------
# The rule engine's search vocabulary
# ---------------------------------------------------------------------------


def build_vocabulary(mode: str) -> list[ChordLabel]:
    """All chords the rule engine may choose from, for one mode.

    Deliberately generous: every diatonic triad and seventh in every inversion,
    secondary dominants and applied leading-tone chords to every tonicizable
    degree, plus the Neapolitan and mode mixture.
    """
    out: list[ChordLabel] = []
    seen: set[tuple[int, str, int]] = set()

    def add(root: int, quality: str, inversions: Iterable[int]) -> None:
        for inv in inversions:
            if inv >= chord_size(quality):
                continue
            key3 = (root % 12, quality, inv)
            if key3 in seen:
                continue
            seen.add(key3)
            out.append(make_chord(root, quality, inv, mode))

    triad_table = _MINOR_TRIAD_QUALITY if mode == "minor" else _MAJOR_TRIAD_QUALITY
    for root, quality in triad_table.items():
        add(root, quality, (0, 1, 2))

    if mode == "major":
        add(7, "dom7", (0, 1, 2, 3))          # V7
        add(2, "min7", (0, 1, 2, 3))          # ii7
        add(11, "halfdim7", (0, 1, 2, 3))     # viiø7
        add(9, "min7", (0, 1, 2))             # vi7
        add(5, "maj7", (0, 1))                # IV7
        add(0, "maj7", (0,))                  # IM7
        add(11, "dim7", (0, 1, 2, 3))         # viio7 (borrowed)
        add(3, "maj", (0, 1))                 # bIII
        add(8, "maj", (0, 1))                 # bVI
        add(10, "maj", (0, 1))                # bVII
        add(5, "min", (0, 1))                 # iv (mixture)
        add(2, "dim", (0, 1))                 # iio (mixture)
        add(1, "maj", (1, 0))                 # Neapolitan, normally in first inversion
    else:
        add(7, "dom7", (0, 1, 2, 3))          # V7
        add(11, "dim7", (0, 1, 2, 3))         # viio7
        add(11, "dim", (0, 1))
        add(2, "halfdim7", (0, 1, 2, 3))      # iiø7
        add(10, "dom7", (0, 1, 2))            # VII7 -> III
        add(3, "maj7", (0, 1))                # IIIM7
        add(5, "min7", (0, 1, 2))             # iv7
        add(0, "maj", (0, 1))                 # picardy / major tonic
        add(5, "maj", (0, 1))                 # IV (raised 6, melodic minor)
        add(1, "maj", (1, 0))                 # Neapolitan

    for target in APPLIED_TARGETS[mode]:
        add((target + 7) % 12, "dom7", (0, 1, 2, 3))
        add((target + 7) % 12, "maj", (0, 1))
        add((target + 11) % 12, "dim7", (0, 1, 2))
        add((target + 11) % 12, "halfdim7", (0, 1))

    return out
