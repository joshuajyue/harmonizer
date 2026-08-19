"""The substitution vocabulary: every way one chord can stand in for another.

Each generator produces *candidates* for one harmonic unit, tagged with the
provenance that will reach the API — `substitutionOf` (the roman numeral of the
chord that was replaced) and `substitutionKind` (how it was derived). That is a
deliberate commitment: the project already explains why a harmonization is weak
through `violations[]`, and a reharmonization that cannot explain itself is just
a random chord generator with good PR.

Everything here is hand-written theory with no learned parameters. The learned
model in `model.py` scores this same candidate space; keeping the vocabulary
separate from the scoring is what makes the rule-based and stochastic engines
directly comparable, because they differ in exactly one thing — how a path
through these candidates is chosen.

The fundamentals under test in tests/test_substitutions.py, because they are
easy to get subtly and confidently wrong:

    the tritone substitute of V7 in C is Db7, not D7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .chords import (
    CONFLICT,
    SOFT_CONFLICT,
    JazzChord,
    classify_melody_note,
    dominant_of,
    tritone_root,
)
from .skeleton import Unit

#: How much melodic weight of hard conflict a candidate may carry before it is
#: refused outright. The oracle measures real jazz melodies sitting on avoid
#: notes 8% of the (weighted) time over the changes actually played, so a
#: zero-tolerance constraint would be stricter than the music it imitates —
#: but a chord that clashes with a THIRD of what sounds over it is simply the
#: wrong chord.
MAX_HARD_CONFLICT = 0.30

#: A unit has to be long enough for two chords to be two chords. Splitting a
#: two-beat unit gives one chord per beat, which is four changes a bar — the
#: oracle measures real jazz at 2.8 (lead sheets) to 4.2 (as played) beats per
#: chord, so that is not reharmonization, it is agitation.
MIN_SPLIT_DURATION = 3.0


@dataclass(frozen=True)
class Candidate:
    """One or two chords covering a unit, with provenance and a rule score."""

    chords: tuple[JazzChord, ...]
    starts: tuple[float, ...]
    durations: tuple[float, ...]
    kind: str | None
    #: Hand-written desirability, higher is better. Used by the rule engine;
    #: the stochastic engine replaces it with a learned log-probability.
    bonus: float = 0.0
    #: Weighted melody conflict inside the unit, 0 when the chord fits.
    melody_penalty: float = 0.0

    @property
    def first(self) -> JazzChord:
        return self.chords[0]

    @property
    def last(self) -> JazzChord:
        return self.chords[-1]

    @property
    def is_identity(self) -> bool:
        return self.kind is None

    def label(self) -> str:
        return " ".join(chord.symbol() for chord in self.chords)


@dataclass
class Context:
    """Everything a generator needs to know beyond the unit itself."""

    tonic: int
    mode: str
    previous: JazzChord | None
    following: JazzChord | None
    #: The base chord TWO units ahead. Needed for the two-bar ii-V-I, which is
    #: the most common shape in jazz and cannot be seen from one unit ahead:
    #: this unit becomes the ii of a dominant that the NEXT unit becomes.
    following2: JazzChord | None = None
    following_roman: str = ""
    is_last: bool = False
    allow_coltrane: bool = False

    def degree(self, root: int) -> int:
        return (root - self.tonic) % 12

    @property
    def tonic_pcs(self) -> tuple[int, ...]:
        third = 4 if self.mode == "major" else 3
        return (self.tonic, (self.tonic + third) % 12, (self.tonic + 7) % 12)


# ---------------------------------------------------------------------------
# Melody-driven filtering
# ---------------------------------------------------------------------------


def melody_penalty(chord: JazzChord, weighted_pcs: Sequence[tuple[int, float]]) -> tuple[float, float]:
    """(hard, soft) weighted conflict share of a chord against a unit's melody.

    Returned as shares of the unit's melodic weight so the number means the
    same thing for a bar of whole notes and a bar of eighth notes.
    """
    total = sum(weight for _, weight in weighted_pcs)
    if total <= 0:
        return 0.0, 0.0
    hard = soft = 0.0
    for pitch_class, weight in weighted_pcs:
        verdict = classify_melody_note(chord, pitch_class)
        if verdict.verdict == CONFLICT:
            hard += weight
        elif verdict.verdict == SOFT_CONFLICT:
            soft += weight
    return hard / total, soft / total


def _acceptable(chord: JazzChord, weighted_pcs: Sequence[tuple[int, float]]) -> tuple[bool, float]:
    hard, soft = melody_penalty(chord, weighted_pcs)
    return hard <= MAX_HARD_CONFLICT, hard + 0.35 * soft


# ---------------------------------------------------------------------------
# Quality upgrades — the floor of any jazz reharmonization
# ---------------------------------------------------------------------------

#: The rules engine is diatonic-functional by construction and emits triads;
#: jazz states sevenths (the oracle: 93% of treebank chords are sevenths or
#: sixths). Upgrading a triad is not really a substitution, it is a change of
#: dialect, so it is tagged "extension" rather than pretending to be more.
_UPGRADES: dict[str, tuple[str, ...]] = {
    "maj": ("maj7", "maj6"),
    "min": ("min7", "min6"),
    "dim": ("halfdim7", "dim7"),
    "aug": ("aug",),
    "dom7": ("dom7",),
    "maj7": ("maj7", "maj6"),
    "min7": ("min7", "min6"),
    "halfdim7": ("halfdim7",),
    "dim7": ("dim7",),
    "minmaj7": ("minmaj7",),
    "maj6": ("maj6", "maj7"),
    "min6": ("min6", "min7"),
    "sus4": ("sus4",),
    "sus2": ("sus2",),
}


def diatonic_pcs(context: Context) -> frozenset[int]:
    from .metrics import diatonic_pcs as _diatonic

    return _diatonic(context.tonic, context.mode)


def upgrade_qualities(chord: JazzChord, context: Context) -> list[str]:
    """Seventh-chord dialects of a triad, most in-key first.

    Which seventh a triad wants depends on the key, not on the triad. A major
    triad on the fourth degree of a minor key takes a dominant seventh (dorian
    IV7) while the same triad on the flat sixth takes a major seventh, and
    choosing by quality alone imports a chromatic note the tune never asked
    for.
    """
    options = list(_UPGRADES.get(chord.quality, (chord.quality,)))
    if chord.quality in ("maj", "maj7", "maj6"):
        scale = diatonic_pcs(context)
        seventh_flat = (chord.root + 10) % 12 in scale
        seventh_natural = (chord.root + 11) % 12 in scale
        if seventh_flat and not seventh_natural:
            # V in a major key: the diatonic seventh is flat, so the triad
            # becomes a dominant. Offering a major seventh here would import a
            # raised fourth and quietly modulate — Gmaj7 in C is a chord in G.
            options = [option for option in options if option != "maj7"]
            options.insert(0, "dom7")
        elif seventh_natural:
            options = [option for option in options if option != "dom7"] or options
    return list(dict.fromkeys(options))


def _tonic_function(context: Context, chord: JazzChord) -> bool:
    degree = context.degree(chord.root)
    if context.mode == "major":
        return degree in (0, 9)
    return degree in (0, 3)


def _is_dominant_of(chord: JazzChord, target: JazzChord | None) -> bool:
    """Whether `chord` behaves as the dominant of `target`."""
    if target is None:
        return False
    return (chord.root - target.root) % 12 == 7


def identity_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """The base chord, in jazz dialect. Always available, never refused."""
    out: list[Candidate] = []
    weighted = unit.weighted_pcs
    base = unit.base
    degree = context.degree(base.root)

    qualities = upgrade_qualities(base, context)
    # A diatonic triad on V, or any chord that resolves down a fifth into the
    # next one, wants to be a dominant seventh: that is the single most
    # characteristic move from common-practice into jazz.
    if base.quality in ("maj", "dom7") and (
        degree == 7 or _is_dominant_of(base, context.following)
    ):
        qualities.insert(0, "dom7")
    qualities = list(dict.fromkeys(qualities))
    preferred = qualities[0]
    # Inversions are dropped: they are chorale voice-leading artefacts, and a
    # jazz bass plays roots unless the harmony itself asks otherwise.
    for quality in dict.fromkeys(qualities):
        chord = JazzChord(root=base.root, quality=quality)
        ok, penalty = _acceptable(chord, weighted)
        kind = None if quality == base.quality else "extension"
        bonus = 0.0 if quality == base.quality else 0.25
        if quality == preferred:
            bonus += 0.15
        if not ok and quality != base.quality:
            continue
        out.append(_candidate(unit, [chord], kind, bonus - penalty * 2.0, penalty))
    if not out:
        chord = JazzChord(root=base.root, quality=base.quality)
        _, penalty = _acceptable(chord, weighted)
        out.append(_candidate(unit, [chord], None, -penalty * 2.0, penalty))
    return out


# ---------------------------------------------------------------------------
# Substitutions
# ---------------------------------------------------------------------------


def tritone_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """bII7 for V7: same tritone, bass moving by semitone instead of a fifth.

    Offered when the base chord is functioning as a dominant — either because
    it is the key's V, or because it resolves down a fifth into whatever comes
    next. The substitute is a dominant seventh a tritone above the original
    root, which for V7 in C is Db7.
    """
    base = unit.base
    degree = context.degree(base.root)
    functioning = (
        base.quality in ("dom7", "maj", "sus4")
        and (degree == 7 or _is_dominant_of(base, context.following))
    )
    if not functioning:
        return []
    substitute = JazzChord(root=tritone_root(base.root), quality="dom7")
    ok, penalty = _acceptable(substitute, unit.weighted_pcs)
    out: list[Candidate] = []
    if ok:
        out.append(_candidate(unit, [substitute], "tritone", 0.9 - penalty * 2.0, penalty))
    out.extend(approach_split(unit, context, substitute, "tritone", 0.85))
    return out


def secondary_dominant_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """Turn this unit into the dominant of whatever follows, or its ii-V.

    Any chord can be preceded by its own dominant, which is why this is the
    most productive single device in the vocabulary. The split form inserts the
    related ii in the first half, which is the same idea stated in two chords.
    """
    following = context.following
    if following is None:
        return []
    target = following.root
    if (unit.base.root - target) % 12 == 0:
        return []
    out: list[Candidate] = []
    weighted = unit.weighted_pcs

    dominant = JazzChord(root=dominant_of(target), quality="dom7")
    ok, penalty = _acceptable(dominant, weighted)
    if ok and dominant.root != unit.base.root:
        out.append(_candidate(unit, [dominant], "secondary_dominant", 0.7 - penalty * 2.0, penalty))
    substitute_whole = JazzChord(root=tritone_root(dominant.root), quality="dom7")
    ok_sub, penalty_sub = _acceptable(substitute_whole, weighted)
    if ok_sub and substitute_whole.root != unit.base.root:
        out.append(_candidate(unit, [substitute_whole], "tritone", 0.75 - penalty_sub * 2.0, penalty_sub))
    out.extend(approach_split(unit, context, dominant, "secondary_dominant", 0.95))
    out.extend(approach_split(
        unit, context, JazzChord(root=tritone_root(dominant.root), quality="dom7"), "tritone", 0.9,
    ))

    if unit.duration >= MIN_SPLIT_DURATION:
        related = JazzChord(root=(dominant_of(target) + 5) % 12, quality="min7")
        first_half, second_half = _halves(unit)
        ok_a, penalty_a = _acceptable(related, _clip_weights(unit, *first_half))
        ok_b, penalty_b = _acceptable(dominant, _clip_weights(unit, *second_half))
        if ok_a and ok_b:
            out.append(_candidate(
                unit,
                [related, dominant],
                "secondary_dominant",
                0.85 - (penalty_a + penalty_b),
                (penalty_a + penalty_b) / 2,
                split=True,
            ))
        # ...and its tritone substitute, which is the same ii-V with a
        # chromatic bass: Dm7 Db7 | Cmaj7.
        substitute = JazzChord(root=tritone_root(dominant.root), quality="dom7")
        ok_c, penalty_c = _acceptable(substitute, _clip_weights(unit, *second_half))
        if ok_a and ok_c:
            out.append(_candidate(
                unit,
                [related, substitute],
                "tritone",
                0.8 - (penalty_a + penalty_c),
                (penalty_a + penalty_c) / 2,
                split=True,
            ))
    return out


def related_ii_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """Become the ii of a dominant that the next unit can become.

    The oracle measures 4.65 ii-V pairs per 16 bars in the changes real bands
    play, against 0.08 in the diatonic skeleton this engine starts from. Almost
    all of that gap is two-bar ii-Vs, which no amount of per-unit substitution
    can produce: the ii and the V live in different units, so the candidate for
    THIS unit has to be chosen with the unit after next in view.
    """
    target = context.following2
    if target is None:
        return []
    root = (target.root + 2) % 12
    if root == unit.base.root and unit.base.quality in ("min7", "min", "halfdim7"):
        return []
    target_is_minor = target.quality in ("min", "min7", "min6", "minmaj7", "dim", "dim7", "halfdim7")
    qualities = ("halfdim7", "min7") if target_is_minor else ("min7", "halfdim7")
    out: list[Candidate] = []
    for quality in qualities:
        chord = JazzChord(root=root, quality=quality)
        ok, penalty = _acceptable(chord, unit.weighted_pcs)
        if not ok:
            continue
        out.append(_candidate(unit, [chord], "secondary_dominant", 0.65 - penalty * 2.0, penalty))
        break
    return out


def backdoor_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """bVII7 (and iv7-bVII7) resolving to the tonic — the backdoor cadence.

    The classic soul/standards alternative to V-I: bVII7 shares its guide tones
    with iiø7 of the parallel minor and resolves up a whole tone. Only offered
    into a tonic-function chord, because that is the only place it means
    anything.
    """
    following = context.following
    if following is None or not _tonic_function(context, following):
        return []
    if context.degree(unit.base.root) == 0:
        return []
    out: list[Candidate] = []
    weighted = unit.weighted_pcs
    backdoor = JazzChord(root=(following.root + 10) % 12, quality="dom7")
    ok, penalty = _acceptable(backdoor, weighted)
    if ok:
        out.append(_candidate(unit, [backdoor], "backdoor", 0.75 - penalty * 2.0, penalty))
    out.extend(approach_split(unit, context, backdoor, "backdoor", 0.8))
    if unit.duration >= MIN_SPLIT_DURATION:
        minor_iv = JazzChord(root=(following.root + 5) % 12, quality="min7")
        first_half, second_half = _halves(unit)
        ok_a, penalty_a = _acceptable(minor_iv, _clip_weights(unit, *first_half))
        ok_b, penalty_b = _acceptable(backdoor, _clip_weights(unit, *second_half))
        if ok_a and ok_b:
            out.append(_candidate(
                unit, [minor_iv, backdoor], "backdoor",
                0.9 - (penalty_a + penalty_b), (penalty_a + penalty_b) / 2, split=True,
            ))
    return out


#: Borrowed chords by scale degree of the chord being replaced, in major.
_MIXTURE_MAJOR: dict[int, tuple[tuple[int, str], ...]] = {
    0: ((8, "maj7"), (3, "maj7")),           # I -> bVI, bIII
    2: ((2, "halfdim7"), (10, "dom7")),      # ii -> iiø7, bVII7
    4: ((3, "maj7"), (8, "maj7")),           # iii -> bIII, bVI
    5: ((5, "min7"), (5, "min6"), (10, "dom7")),  # IV -> iv, bVII7
    7: ((10, "dom7"), (1, "maj7")),          # V -> bVII7, bII
    9: ((8, "maj7"), (9, "halfdim7")),       # vi -> bVI, viø7
    11: ((11, "halfdim7"),),
}

#: In minor the borrowing runs the other way, toward the parallel major.
_MIXTURE_MINOR: dict[int, tuple[tuple[int, str], ...]] = {
    0: ((0, "minmaj7"), (0, "min6"), (0, "maj7")),   # i -> iM7, i6, picardy I
    5: ((5, "min6"), (5, "dom7")),                   # iv -> iv6, IV7 (dorian)
    7: ((7, "dom7"), (10, "dom7")),                  # V7, bVII7
    3: ((3, "maj7"),),
    8: ((8, "maj7"), (8, "dom7")),
    10: ((10, "dom7"),),
}


def modal_interchange_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """Chords borrowed from the parallel mode.

    Modal interchange is the device that colours a progression without moving
    it: the root motion survives, the mode does not. iv in a major key is the
    single most recognisable example.
    """
    table = _MIXTURE_MINOR if context.mode == "minor" else _MIXTURE_MAJOR
    options = table.get(context.degree(unit.base.root), ())
    out: list[Candidate] = []
    weighted = unit.weighted_pcs
    for degree, quality in options:
        chord = JazzChord(root=(context.tonic + degree) % 12, quality=quality)
        if chord.same_harmony(unit.base):
            continue
        ok, penalty = _acceptable(chord, weighted)
        if not ok:
            continue
        out.append(_candidate(unit, [chord], "modal_interchange", 0.6 - penalty * 2.0, penalty))
    return out


def relative_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """Swap a chord for its relative: I/vi/iii, IV/ii, V/viiø.

    Two chords sharing two notes can stand in for each other with barely any
    melodic consequence, which is exactly why this is the safest device in the
    book and the first one taught.
    """
    degree = context.degree(unit.base.root)
    if context.mode == "major":
        table = {
            0: ((9, "min7"), (4, "min7")),
            5: ((2, "min7"),),
            2: ((5, "maj7"),),
            9: ((0, "maj7"),),
            4: ((0, "maj7"),),
            7: ((11, "halfdim7"),),
        }
    else:
        table = {
            0: ((3, "maj7"), (8, "maj7")),
            5: ((10, "dom7"),),
            3: ((0, "min7"),),
            7: ((2, "halfdim7"),),
        }
    out: list[Candidate] = []
    weighted = unit.weighted_pcs
    for target_degree, quality in table.get(degree, ()):
        chord = JazzChord(root=(context.tonic + target_degree) % 12, quality=quality)
        if chord.same_harmony(unit.base):
            continue
        ok, penalty = _acceptable(chord, weighted)
        if not ok:
            continue
        out.append(_candidate(unit, [chord], "relative", 0.45 - penalty * 2.0, penalty))
    return out


def passing_dim_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """A diminished seventh in the second half, leading by semitone into the next chord.

    #Io7 between I and ii is the archetype. The diminished chord is a passing
    event, so it takes the weak half of the unit and never the strong one.
    """
    following = context.following
    if following is None or unit.duration < MIN_SPLIT_DURATION:
        return []
    approach = JazzChord(root=(following.root - 1) % 12, quality="dim7")
    return approach_split(unit, context, approach, "passing_dim", 0.55)


def chromatic_approach_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """Approach the next chord from a semitone away, in its own quality.

    Planing a chord chromatically into the next one is a texture device rather
    than a functional one, which is why it is offered on the weak half only.
    """
    following = context.following
    if following is None or unit.duration < MIN_SPLIT_DURATION:
        return []
    out: list[Candidate] = []
    for offset in (1, -1):
        approach = JazzChord(
            root=(following.root + offset) % 12,
            quality=upgrade_qualities(following, context)[0],
        )
        if approach.same_harmony(unit.base):
            continue
        out.extend(approach_split(unit, context, approach, "chromatic_approach", 0.4))
    return out


def coltrane_candidates(unit: Unit, context: Context) -> list[Candidate]:
    """Major-third cycle into the following chord: Giant Steps in one bar.

    The cycle divides the octave in three, so the unit becomes two dominants a
    major third apart, each resolving down a fifth, landing on the chord that
    was coming anyway. It is the most disruptive device here and is gated
    behind an explicit flag and a full melody check.
    """
    following = context.following
    if not context.allow_coltrane or following is None or unit.duration < MIN_SPLIT_DURATION:
        return []
    target = following.root
    first_root = (target + 8) % 12   # bVI7 of the target: down a major third
    second_root = dominant_of(target)
    first = JazzChord(root=first_root, quality="dom7")
    second = JazzChord(root=second_root, quality="dom7")
    first_half, second_half = _halves(unit)
    ok_a, penalty_a = _acceptable(first, _clip_weights(unit, *first_half))
    ok_b, penalty_b = _acceptable(second, _clip_weights(unit, *second_half))
    if not (ok_a and ok_b):
        return []
    return [_candidate(
        unit, [first, second], "coltrane",
        0.5 - (penalty_a + penalty_b), (penalty_a + penalty_b) / 2, split=True,
    )]


GENERATORS = (
    identity_candidates,
    tritone_candidates,
    secondary_dominant_candidates,
    related_ii_candidates,
    backdoor_candidates,
    modal_interchange_candidates,
    relative_candidates,
    passing_dim_candidates,
    chromatic_approach_candidates,
    coltrane_candidates,
)


def generate(unit: Unit, context: Context) -> list[Candidate]:
    """Every candidate for one unit, deduplicated, identity first.

    Hard melody conflicts are refused inside the generators rather than
    filtered afterwards, so a constrained sampler downstream never has to
    reject-sample: the lattice it walks contains only chords that can actually
    support the melody.
    """
    out: list[Candidate] = []
    seen: set[tuple] = set()
    for generator in GENERATORS:
        for candidate in generator(unit, context):
            signature = tuple(
                (chord.root, chord.quality, chord.bass_pc) for chord in candidate.chords
            )
            if signature in seen:
                continue
            seen.add(signature)
            out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def approach_split(
    unit: Unit,
    context: Context,
    approach: JazzChord,
    kind: str,
    bonus: float,
) -> list[Candidate]:
    """[base chord | approach chord] — the tune's harmony, then the move.

    This is the shape most reharmonization actually takes. Replacing a whole
    bar re-functions the tune; putting the substitute on the weak half leaves
    the tune's harmony where the listener expects it and lands the surprise on
    the way out of the bar.
    """
    if unit.duration < MIN_SPLIT_DURATION:
        return []
    if approach.root == unit.base.root:
        return []
    base_chord = JazzChord(root=unit.base.root, quality=upgrade_qualities(unit.base, context)[0])
    first_half, second_half = _halves(unit)
    ok_a, penalty_a = _acceptable(base_chord, _clip_weights(unit, *first_half))
    ok_b, penalty_b = _acceptable(approach, _clip_weights(unit, *second_half))
    if not (ok_a and ok_b):
        return []
    return [_candidate(
        unit, [base_chord, approach], kind,
        bonus - (penalty_a + penalty_b), (penalty_a + penalty_b) / 2, split=True,
    )]


def _halves(unit: Unit) -> tuple[tuple[float, float], tuple[float, float]]:
    half = unit.duration / 2
    return (unit.start, half), (unit.start + half, half)


def _clip_weights(unit: Unit, start: float, duration: float) -> list[tuple[int, float]]:
    from .metrics import note_weight

    totals: dict[int, float] = {}
    stop = start + duration
    for note_start, pitch, note_duration in unit.melody:
        overlap = min(note_start + note_duration, stop) - max(note_start, start)
        if overlap <= 1e-6:
            continue
        totals[pitch % 12] = totals.get(pitch % 12, 0.0) + note_weight(max(note_start, start), overlap)
    return sorted(totals.items())


def _candidate(
    unit: Unit,
    chords: Sequence[JazzChord],
    kind: str | None,
    bonus: float,
    penalty: float,
    *,
    split: bool = False,
) -> Candidate:
    if split and len(chords) == 2:
        half = unit.duration / 2
        starts = (unit.start, unit.start + half)
        durations = (half, half)
    else:
        starts = tuple(unit.start for _ in chords)
        durations = tuple(unit.duration / len(chords) for _ in chords)
        if len(chords) > 1:
            starts = tuple(unit.start + i * unit.duration / len(chords) for i in range(len(chords)))
    tagged = tuple(
        chord.with_provenance(unit.base_roman or None, kind) if kind else chord
        for chord in chords
    )
    return Candidate(
        chords=tagged,
        starts=starts,
        durations=durations,
        kind=kind,
        bonus=bonus,
        melody_penalty=penalty,
    )
