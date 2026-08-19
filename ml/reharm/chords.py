"""Jazz chord representation: core quality + explicit tensions, and the
melody-compatibility theory that makes reharmonization checkable.

The split between `quality` and `extensions` mirrors `contracts.schema.Chord`
exactly, and for the same reason: the cross product of core qualities and
altered tensions is unbounded, so 13b9#11 is a *dom7 with three extensions*, not
a 47th entry in a quality enum.

The other half of this module is the avoid-note model. Reharmonization fails, in
practice, by putting a chord under a melody note that the chord cannot support —
so that has to be a first-class, objectively computable thing rather than a
matter of taste. The model here is the standard one, stated as a single rule
with named exceptions:

    a melody note sounding a semitone ABOVE a chord tone voiced beneath it is a
    conflict.

That single rule derives the whole conventional avoid-note table: the 11 over
maj7 (semitone above the 3rd), the b9 over min7, the natural 5 over m7b5, the
1/4/7/10 over dim7. The exceptions are equally standard and are what make the
model musical rather than merely conservative:

  * On a dominant chord, b9 / #9 / #11 / b13 are *available altered tensions*,
    not conflicts — that is the entire sound of altered dominant harmony.
  * A conflict against the fifth is soft and self-repairing: drop the fifth from
    the voicing and the clash is gone. Pianists do this without thinking.
  * A conflict against the third of a dominant is repaired by suspending it —
    melody on the 4th is exactly why 7sus4 chords exist.
  * A conflict against a guide tone (3rd or 7th) of a non-dominant chord is
    real, unrepairable, and is what we actually want to count.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# Qualities
# ---------------------------------------------------------------------------

#: Core quality -> semitones above the root. Identical vocabulary to
#: contracts.schema.Chord.quality, so no translation layer is needed anywhere.
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
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}

#: Qualities that behave as dominants (V-function) for resolution and tension
#: purposes. sus4 is here because a 7sus4 is a dominant with a delayed third.
DOMINANT_QUALITIES = frozenset({"dom7", "sus4"})

#: Semitone offset of each extension name above the root.
EXTENSION_SEMITONES: dict[str, int] = {
    "b9": 1, "9": 2, "#9": 3,
    "11": 5, "#11": 6,
    "b13": 8, "13": 9,
    "b5": 6, "#5": 8,
    "6": 9, "7": 10, "maj7": 11,
}

#: Extensions that REPLACE the perfect fifth rather than adding to the chord.
FIFTH_REPLACING = frozenset({"b5", "#5", "b13"})

#: Altered tensions that are idiomatic on a dominant chord and therefore never
#: count as melody conflicts there.
DOMINANT_ALTERATIONS = frozenset({"b9", "#9", "#11", "b13"})

_ALTERATION_BY_SEMITONE = {1: "b9", 3: "#9", 6: "#11", 8: "b13"}

PITCH_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

#: Chord-symbol suffix per core quality, in the common lead-sheet spelling.
_SYMBOL_SUFFIX = {
    "maj": "", "min": "m", "dim": "dim", "aug": "+",
    "dom7": "7", "maj7": "maj7", "min7": "m7", "halfdim7": "m7b5",
    "dim7": "dim7", "minmaj7": "mMaj7", "maj6": "6", "min6": "m6",
    "sus2": "sus2", "sus4": "sus4",
}

#: Roman-numeral suffix per core quality. Case of the stem carries major/minor,
#: as in the rest of the project, so these only carry the extra information.
_ROMAN_SUFFIX = {
    "maj": "", "min": "", "dim": "o", "aug": "+",
    "dom7": "7", "maj7": "M7", "min7": "7", "halfdim7": "\u00f87",
    "dim7": "o7", "minmaj7": "M7", "maj6": "6", "min6": "6",
    "sus2": "sus2", "sus4": "sus4",
}

_LOWERCASE_QUALITIES = frozenset({"min", "dim", "min7", "halfdim7", "dim7", "minmaj7", "min6"})

_NUMERAL_MAJOR = {
    0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
    6: "#IV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII",
}
_NUMERAL_MINOR = {
    0: "I", 1: "bII", 2: "II", 3: "III", 4: "#III", 5: "IV",
    6: "#IV", 7: "V", 8: "VI", 9: "#VI", 10: "VII", 11: "#VII",
}


def numeral_stem(relative_root: int, mode: str) -> str:
    table = _NUMERAL_MINOR if mode == "minor" else _NUMERAL_MAJOR
    return table[relative_root % 12]


def pitch_name(pitch_class: int, *, flats: bool = True) -> str:
    names = PITCH_NAMES_FLAT if flats else PITCH_NAMES_SHARP
    return names[pitch_class % 12]


# ---------------------------------------------------------------------------
# The chord
# ---------------------------------------------------------------------------

#: Substitution provenance kinds, matching contracts.schema.Chord.substitutionKind.
SUBSTITUTION_KINDS = (
    "tritone",
    "backdoor",
    "modal_interchange",
    "relative",
    "passing_dim",
    "secondary_dominant",
    "chromatic_approach",
    "extension",
    "coltrane",
)


@dataclass(frozen=True)
class JazzChord:
    """An absolute jazz chord: root pitch class, core quality, tensions.

    `substitution_of` / `substitution_kind` carry reharmonization provenance
    straight through to the API so the UI can explain a substitution instead of
    just emitting it.
    """

    root: int
    quality: str
    extensions: tuple[str, ...] = ()
    bass: int | None = None
    substitution_of: str | None = None
    substitution_kind: str | None = None

    def __post_init__(self) -> None:
        if self.quality not in QUALITY_TEMPLATES:
            raise ValueError(f"unknown quality {self.quality!r}")
        for extension in self.extensions:
            if extension not in EXTENSION_SEMITONES:
                raise ValueError(f"unknown extension {extension!r}")
        object.__setattr__(self, "root", self.root % 12)
        if self.bass is not None:
            object.__setattr__(self, "bass", self.bass % 12)
        if self.substitution_kind is not None and self.substitution_kind not in SUBSTITUTION_KINDS:
            raise ValueError(f"unknown substitution kind {self.substitution_kind!r}")

    # -- pitch content -----------------------------------------------------

    @property
    def is_dominant(self) -> bool:
        return self.quality in DOMINANT_QUALITIES

    @property
    def is_seventh(self) -> bool:
        return len(QUALITY_TEMPLATES[self.quality]) >= 4

    @property
    def core_intervals(self) -> tuple[int, ...]:
        """Core chord tones as semitones above the root, alterations applied."""
        intervals = list(QUALITY_TEMPLATES[self.quality])
        if any(e in FIFTH_REPLACING for e in self.extensions) and 7 in intervals:
            intervals.remove(7)
            for extension in self.extensions:
                if extension in ("b5", "#5"):
                    intervals.append(EXTENSION_SEMITONES[extension])
        return tuple(sorted(set(intervals)))

    @property
    def core_pcs(self) -> tuple[int, ...]:
        return tuple((self.root + i) % 12 for i in self.core_intervals)

    @property
    def tension_pcs(self) -> tuple[int, ...]:
        return tuple((self.root + EXTENSION_SEMITONES[e]) % 12 for e in self.extensions)

    @property
    def all_pcs(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.core_pcs) | set(self.tension_pcs)))

    @property
    def third_interval(self) -> int | None:
        for candidate in (4, 3, 5, 2):  # maj3, min3, sus4, sus2
            if candidate in self.core_intervals:
                return candidate
        return None

    @property
    def seventh_interval(self) -> int | None:
        for candidate in (10, 11, 9):  # b7, maj7, 6th standing in for the 7th
            if candidate in self.core_intervals:
                return candidate
        return None

    @property
    def guide_tone_pcs(self) -> tuple[int, ...]:
        """Third and seventh — the tones that define the quality."""
        out = []
        for interval in (self.third_interval, self.seventh_interval):
            if interval is not None:
                out.append((self.root + interval) % 12)
        return tuple(out)

    @property
    def bass_pc(self) -> int:
        return self.root if self.bass is None else self.bass

    @property
    def inversion(self) -> int:
        pcs = self.core_pcs
        return pcs.index(self.bass_pc) if self.bass_pc in pcs else 0

    def with_extensions(self, extensions: Iterable[str]) -> "JazzChord":
        merged = list(self.extensions)
        for extension in extensions:
            if extension not in merged:
                merged.append(extension)
        return replace(self, extensions=tuple(_sorted_extensions(merged)))

    def without_extensions(self) -> "JazzChord":
        return replace(self, extensions=())

    def with_provenance(self, of: str | None, kind: str | None) -> "JazzChord":
        return replace(self, substitution_of=of, substitution_kind=kind)

    # -- naming ------------------------------------------------------------

    def symbol(self, *, flats: bool = True) -> str:
        """Lead-sheet chord symbol, e.g. "Db7b9" or "Cm7(11)"."""
        text = pitch_name(self.root, flats=flats) + _SYMBOL_SUFFIX[self.quality]
        text += _extension_text(self.extensions, base=text)
        if self.bass is not None and self.bass != self.root:
            text += "/" + pitch_name(self.bass, flats=flats)
        return text

    def roman(self, tonic: int, mode: str) -> str:
        """Roman numeral relative to a key, e.g. "bII7(#11)"."""
        stem = numeral_stem((self.root - tonic) % 12, mode)
        accidental = stem[0] if stem and stem[0] in "b#" else ""
        body = stem[len(accidental):]
        body = body.lower() if self.quality in _LOWERCASE_QUALITIES else body.upper()
        text = accidental + body + _ROMAN_SUFFIX[self.quality]
        text += _extension_text(self.extensions, base=text)
        if self.bass is not None and self.bass != self.root:
            bass_stem = numeral_stem((self.bass - tonic) % 12, mode)
            text += "/" + bass_stem
        return text

    def same_harmony(self, other: "JazzChord") -> bool:
        """Equal as a sounding chord, ignoring provenance metadata."""
        return (
            self.root == other.root
            and self.quality == other.quality
            and set(self.extensions) == set(other.extensions)
            and self.bass_pc == other.bass_pc
        )


_EXTENSION_ORDER = ["b9", "9", "#9", "11", "#11", "b5", "#5", "b13", "13", "6", "7", "maj7"]


def _sorted_extensions(extensions: Iterable[str]) -> list[str]:
    order = {name: i for i, name in enumerate(_EXTENSION_ORDER)}
    return sorted(set(extensions), key=lambda e: order.get(e, 99))


def _extension_text(extensions: Sequence[str], *, base: str = "") -> str:
    """Render extensions, parenthesising whenever digits would run together.

    "G7b9" reads correctly; "Cm7" + "11" glued together reads as "Cm711", which
    is a different and nonexistent chord.
    """
    if not extensions:
        return ""
    ordered = _sorted_extensions(extensions)
    ambiguous = base[-1:].isdigit() and ordered[0][0].isdigit()
    if len(ordered) == 1 and not ambiguous:
        return ordered[0]
    return "(" + ",".join(ordered) + ")"


# ---------------------------------------------------------------------------
# Chord-symbol parsing (Jazz Harmony Treebank / Weimar spellings)
# ---------------------------------------------------------------------------

_ROOT_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

#: Suffix -> (core quality, extensions). Covers the entire treebank vocabulary
#: (12 distinct suffixes) plus the richer Weimar spellings.
_SUFFIX_TABLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "": ("maj", ()),
    "^": ("maj", ()),
    "maj": ("maj", ()),
    "M": ("maj", ()),
    "6": ("maj6", ()),
    "69": ("maj6", ("9",)),
    "6/9": ("maj6", ("9",)),
    "^7": ("maj7", ()),
    "maj7": ("maj7", ()),
    "M7": ("maj7", ()),
    "^9": ("maj7", ("9",)),
    "^13": ("maj7", ("9", "13")),
    "^7#11": ("maj7", ("#11",)),
    "^#11": ("maj7", ("#11",)),
    "m": ("min", ()),
    "-": ("min", ()),
    "min": ("min", ()),
    "m6": ("min6", ()),
    "-6": ("min6", ()),
    "m7": ("min7", ()),
    "-7": ("min7", ()),
    "min7": ("min7", ()),
    "m9": ("min7", ("9",)),
    "m11": ("min7", ("9", "11")),
    "m^7": ("minmaj7", ()),
    "-^7": ("minmaj7", ()),
    "mM7": ("minmaj7", ()),
    "m7b5": ("halfdim7", ()),
    "%7": ("halfdim7", ()),
    "%": ("halfdim7", ()),
    "o": ("dim", ()),
    "dim": ("dim", ()),
    "o7": ("dim7", ()),
    "dim7": ("dim7", ()),
    "+": ("aug", ()),
    "aug": ("aug", ()),
    "+7": ("dom7", ("#5",)),
    "7#5": ("dom7", ("#5",)),
    "7": ("dom7", ()),
    "9": ("dom7", ("9",)),
    "13": ("dom7", ("9", "13")),
    "11": ("dom7", ("9", "11")),
    "7b9": ("dom7", ("b9",)),
    "7#9": ("dom7", ("#9",)),
    "7#11": ("dom7", ("#11",)),
    "7b13": ("dom7", ("b13",)),
    "7alt": ("dom7", ("b9", "#9", "b13")),
    "alt": ("dom7", ("b9", "#9", "b13")),
    "sus": ("sus4", ("7",)),
    "7sus": ("sus4", ("7",)),
    "sus4": ("sus4", ()),
    "7sus4": ("sus4", ("7",)),
    "sus2": ("sus2", ()),
}


def parse_symbol(text: str) -> JazzChord | None:
    """Parse a lead-sheet chord symbol. Returns None if it cannot be read.

    Handles the treebank's spellings ("C^7", "A%7", "Bo7", "Gsus", "Eb-7") and
    slash basses ("C/E"). Unknown suffixes return None rather than guessing —
    a silently mis-parsed chord corrupts the corpus statistics that everything
    downstream is calibrated against.
    """
    if not text:
        return None
    body = text.strip().rstrip("*")  # treebank marks some tree nodes with '*'
    if not body:
        return None

    bass_pc: int | None = None
    if "/" in body and not body.endswith("/"):
        head, _, tail = body.rpartition("/")
        parsed_bass = _parse_root(tail)
        if parsed_bass is not None and head:
            bass_pc, body = parsed_bass[0], head

    parsed = _parse_root(body)
    if parsed is None:
        return None
    root, suffix = parsed
    entry = _SUFFIX_TABLE.get(suffix)
    if entry is None:
        return None
    quality, extensions = entry
    return JazzChord(root=root, quality=quality, extensions=extensions, bass=bass_pc)


def _parse_root(text: str, *, dash_is_flat: bool = False) -> tuple[int, str] | None:
    """Split a symbol into (root pitch class, suffix).

    `dash_is_flat` is off by default and on only for key strings. In chord
    symbols "-" means MINOR ("F-7" is F minor 7), while in the treebank's key
    field it means FLAT ("E-" is E flat major). Reading one as the other is a
    silent semitone error across an entire corpus.
    """
    if not text or text[0] not in _ROOT_LETTERS:
        return None
    root = _ROOT_LETTERS[text[0]]
    accidentals = "#b-" if dash_is_flat else "#b"
    index = 1
    while index < len(text) and text[index] in accidentals:
        root += 1 if text[index] == "#" else -1
        index += 1
    return root % 12, text[index:]


def parse_key(text: str) -> tuple[int, str] | None:
    """Treebank key strings: "F" -> F major, "e-" -> Eb minor.

    Case carries the mode and "-" is a flat, so "b-" is Bb minor while "B" is
    B major — an easy and silent way to be a semitone wrong everywhere.
    """
    if not text:
        return None
    letter = text[0]
    mode = "major" if letter.isupper() else "minor"
    parsed = _parse_root(letter.upper() + text[1:], dash_is_flat=True)
    if parsed is None:
        return None
    return parsed[0], mode


# ---------------------------------------------------------------------------
# Melody compatibility
# ---------------------------------------------------------------------------

#: Verdicts for a melody note over a chord, best to worst.
CHORD_TONE = "chord_tone"
STATED_TENSION = "stated_tension"
AVAILABLE_TENSION = "available_tension"
SOFT_CONFLICT = "soft_conflict"
CONFLICT = "conflict"

#: Numeric penalty per verdict. Used as the weight in the aggregate
#: melody-conflict metric, so a soft clash costs a fraction of a hard one.
CONFLICT_WEIGHT = {
    CHORD_TONE: 0.0,
    STATED_TENSION: 0.0,
    AVAILABLE_TENSION: 0.0,
    SOFT_CONFLICT: 0.35,
    CONFLICT: 1.0,
}


@dataclass(frozen=True)
class MelodyVerdict:
    """How a melody pitch class sits over a chord, and how to fix it."""

    verdict: str
    interval: int
    #: Chord tone (semitones above root) the melody note clashes with, if any.
    against: int | None = None
    #: Extension name to declare so the note becomes a stated tension.
    tension: str | None = None
    #: Core interval the voicing must omit for the clash to disappear.
    omit: int | None = None

    @property
    def is_conflict(self) -> bool:
        return self.verdict in (CONFLICT, SOFT_CONFLICT)

    @property
    def weight(self) -> float:
        return CONFLICT_WEIGHT[self.verdict]


def classify_melody_note(chord: JazzChord, pitch_class: int) -> MelodyVerdict:
    """Classify one melody pitch class against one chord.

    The single rule is "a semitone above a voiced chord tone is a clash"; the
    exceptions are the altered dominant tensions and the omittable fifth.
    """
    interval = (pitch_class - chord.root) % 12
    core = set(chord.core_intervals)
    stated = {EXTENSION_SEMITONES[e] for e in chord.extensions}

    if interval in core:
        return MelodyVerdict(CHORD_TONE, interval)
    if interval in stated:
        return MelodyVerdict(STATED_TENSION, interval, tension=_ALTERATION_BY_SEMITONE.get(interval))

    # Altered tensions are the sound of a dominant chord, not a clash on one.
    if chord.is_dominant and interval in _ALTERATION_BY_SEMITONE:
        name = _ALTERATION_BY_SEMITONE[interval]
        if name in DOMINANT_ALTERATIONS:
            omit = 7 if name == "b13" and 7 in core else None
            return MelodyVerdict(AVAILABLE_TENSION, interval, tension=name, omit=omit)

    third, seventh = chord.third_interval, chord.seventh_interval

    # A melody note that flatly contradicts the chord quality is a conflict even
    # though it sits a semitone BELOW the chord tone rather than above it: b7
    # against a major seventh chord says "this is a dominant" while the chord
    # says otherwise.
    if interval == 10 and 11 in core:
        return MelodyVerdict(CONFLICT, interval, against=11)
    # The mirror case, a minor third against a major third, is the blue note —
    # idiomatic on a dominant chord and nowhere else. Allowing it everywhere
    # lets a reharmonizer put a major tonic under a minor-key melody and call
    # it colour, which is how "Greensleeves in A major" happens.
    if interval == 3 and 4 in core:
        if chord.is_dominant:
            return MelodyVerdict(AVAILABLE_TENSION, interval, tension="#9")
        return MelodyVerdict(CONFLICT, interval, against=4)

    # Melody on the third of a suspended chord: resolve the suspension rather
    # than sounding both. Cheap to repair, so soft.
    if chord.quality in ("sus4", "sus2") and interval in (3, 4):
        return MelodyVerdict(SOFT_CONFLICT, interval, against=third, omit=third)

    clashes = sorted(c for c in core if (interval - c) % 12 == 1)
    if not clashes:
        name = _tension_name(interval)
        return MelodyVerdict(AVAILABLE_TENSION, interval, tension=name)

    against = clashes[0]

    # Melody on the 4th over a dominant: suspend the third. That is what a
    # 7sus4 chord is for, and it converts a clash into an idiom.
    if chord.is_dominant and against == third and interval == 5:
        return MelodyVerdict(SOFT_CONFLICT, interval, against=against, omit=third)

    # The fifth is free to drop; the clash dies with it.
    if against == 7 and third is not None and seventh is not None:
        return MelodyVerdict(SOFT_CONFLICT, interval, against=against, tension=_tension_name(interval), omit=7)

    return MelodyVerdict(CONFLICT, interval, against=against)


def _tension_name(interval: int) -> str | None:
    for name, semitone in EXTENSION_SEMITONES.items():
        if semitone == interval and name not in ("b5", "#5", "7", "maj7"):
            return name
    return None


def supports_melody(chord: JazzChord, pitch_classes: Iterable[int]) -> bool:
    """True if no melody note hard-conflicts with the chord."""
    return all(classify_melody_note(chord, pc).verdict != CONFLICT for pc in pitch_classes)


#: Tensions each quality can state without turning into a different chord.
#: A min7 with a stated b13 is an aeolian colour nobody asked for; a dominant
#: with a stated b13 is the sound of the music.
ABSORBABLE: dict[str, frozenset[str]] = {
    "dom7": frozenset({"b9", "#9", "9", "#11", "b13", "13"}),
    "sus4": frozenset({"9", "13", "b9"}),
    "maj7": frozenset({"9", "#11", "13"}),
    "maj6": frozenset({"9", "#11"}),
    "maj": frozenset({"9", "13"}),
    "min7": frozenset({"9", "11", "13"}),
    "min6": frozenset({"9", "11"}),
    "min": frozenset({"9", "11"}),
    "minmaj7": frozenset({"9", "11"}),
    "halfdim7": frozenset({"9", "11"}),
    "dim7": frozenset(),
    "dim": frozenset(),
    "aug": frozenset({"9"}),
    "sus2": frozenset({"13"}),
}

#: A tension has to be worth stating. A passing eighth note that happens to
#: touch the 13th is not evidence that the chord is a 13 chord.
MIN_TENSION_WEIGHT = 0.75


def absorb_melody(chord: JazzChord, weighted_pcs: Sequence[tuple[int, float]]) -> JazzChord:
    """Declare the tensions the melody is actually sitting on.

    A melody note that lands on the b9 of a dominant should be *voiced* as a
    b9 rather than merely tolerated: the difference between a reharmonization
    that sounds intentional and one that sounds like a mistake is usually
    whether the accompaniment admits what the melody is doing.
    """
    allowed = ABSORBABLE.get(chord.quality, frozenset())
    tensions: list[str] = []
    for pitch_class, weight in weighted_pcs:
        if weight < MIN_TENSION_WEIGHT:
            continue
        verdict = classify_melody_note(chord, pitch_class)
        if verdict.verdict not in (AVAILABLE_TENSION, SOFT_CONFLICT):
            continue
        if verdict.tension in allowed:
            tensions.append(verdict.tension)
    return chord.with_extensions(tensions) if tensions else chord


# ---------------------------------------------------------------------------
# Functional relationships used by every substitution rule
# ---------------------------------------------------------------------------


def tritone_root(root: int) -> int:
    """Root a tritone away. The sub of V7 in C is Db7, never D7."""
    return (root + 6) % 12


def dominant_of(target_root: int) -> int:
    """Root of the V that resolves to `target_root`."""
    return (target_root + 7) % 12


def resolves_down_fifth(a: JazzChord, b: JazzChord) -> bool:
    return (a.root - b.root) % 12 == 7


def resolves_down_semitone(a: JazzChord, b: JazzChord) -> bool:
    return (a.root - b.root) % 12 == 1


def is_dominant_resolution(a: JazzChord, b: JazzChord) -> bool:
    """Whether a dominant chord `a` actually goes somewhere.

    Both classic resolutions count: down a fifth (V-I) and down a semitone,
    which is the same voice leading heard from the tritone substitute. A
    tritone sub that does not resolve down a semitone is just a wrong chord.
    """
    if not a.is_dominant:
        return False
    return resolves_down_fifth(a, b) or resolves_down_semitone(a, b)
