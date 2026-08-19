"""The fundamentals of the chord vocabulary and the avoid-note model.

Substitution theory is exactly the kind of thing that is subtly and
confidently wrong, so the arithmetic is pinned here rather than trusted. The
headline case is the first test in the file: the tritone substitute of V7 in C
is Db7, and anyone who writes D7 has swapped a tritone for a whole tone and
will not hear about it from any type checker.
"""

import pytest

from ml.reharm.chords import (
    AVAILABLE_TENSION,
    CHORD_TONE,
    CONFLICT,
    SOFT_CONFLICT,
    STATED_TENSION,
    JazzChord,
    absorb_melody,
    classify_melody_note,
    dominant_of,
    is_dominant_resolution,
    parse_key,
    parse_symbol,
    resolves_down_semitone,
    supports_melody,
    tritone_root,
)

C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)


# ---------------------------------------------------------------------------
# The one that matters most
# ---------------------------------------------------------------------------


def test_tritone_substitute_of_v7_in_c_is_db7_not_d7():
    v7 = JazzChord(root=G, quality="dom7")
    substitute = JazzChord(root=tritone_root(v7.root), quality="dom7")
    assert substitute.root == Db
    assert substitute.root != D
    assert substitute.symbol() == "Db7"


def test_tritone_substitute_shares_the_guide_tones():
    """The substitution works because B/F is both G7's and Db7's tritone."""
    g7 = JazzChord(root=G, quality="dom7")
    db7 = JazzChord(root=Db, quality="dom7")
    assert set(g7.guide_tone_pcs) == set(db7.guide_tone_pcs)


def test_tritone_substitute_resolves_down_a_semitone():
    db7 = JazzChord(root=Db, quality="dom7")
    c = JazzChord(root=C, quality="maj7")
    assert resolves_down_semitone(db7, c)
    assert is_dominant_resolution(db7, c)


def test_tritone_root_is_an_involution():
    for root in range(12):
        assert tritone_root(tritone_root(root)) == root


def test_dominant_of_resolves_down_a_fifth():
    assert dominant_of(C) == G
    assert dominant_of(D) == A
    assert is_dominant_resolution(JazzChord(root=G, quality="dom7"), JazzChord(root=C, quality="maj7"))


# ---------------------------------------------------------------------------
# Pitch content
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quality,expected",
    [
        ("maj7", (C, E, G, B)),
        ("dom7", (C, E, G, Bb)),
        ("min7", (C, Eb, G, Bb)),
        ("halfdim7", (C, Eb, Gb, Bb)),
        ("dim7", (C, Eb, Gb, A)),
        ("maj6", (C, E, G, A)),
        ("min6", (C, Eb, G, A)),
        ("minmaj7", (C, Eb, G, B)),
        ("sus4", (C, F, G)),
    ],
)
def test_quality_pitch_classes(quality, expected):
    assert set(JazzChord(root=C, quality=quality).core_pcs) == set(expected)


def test_altered_fifth_replaces_rather_than_adds():
    chord = JazzChord(root=C, quality="dom7", extensions=("#5",))
    assert G not in chord.core_pcs
    assert Ab in chord.core_pcs


def test_guide_tones_are_third_and_seventh():
    assert set(JazzChord(root=C, quality="dom7").guide_tone_pcs) == {E, Bb}
    assert set(JazzChord(root=D, quality="min7").guide_tone_pcs) == {F, C}


def test_extensions_reach_the_pitch_content():
    chord = JazzChord(root=G, quality="dom7", extensions=("b9",))
    assert Ab in chord.all_pcs


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def test_symbols_and_romans():
    assert JazzChord(root=Db, quality="dom7").roman(C, "major") == "bII7"
    assert JazzChord(root=G, quality="dom7").roman(C, "major") == "V7"
    assert JazzChord(root=D, quality="min7").roman(C, "major") == "ii7"
    assert JazzChord(root=C, quality="maj7").roman(C, "major") == "IM7"
    assert JazzChord(root=B, quality="halfdim7").roman(C, "major") == "vii\u00f87"


def test_extension_rendering_never_glues_digits():
    assert JazzChord(root=G, quality="dom7", extensions=("b9",)).symbol() == "G7b9"
    assert JazzChord(root=C, quality="min7", extensions=("11",)).symbol() == "Cm7(11)"
    assert JazzChord(root=G, quality="dom7", extensions=("b9", "b13")).symbol() == "G7(b9,b13)"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,root,quality",
    [
        ("C^7", C, "maj7"),
        ("Eb7", Eb, "dom7"),
        ("F#m7", Gb, "min7"),
        ("A%7", A, "halfdim7"),
        ("Bo7", B, "dim7"),
        ("Gsus", G, "sus4"),
        ("Bb6", Bb, "maj6"),
        ("Dm^7", D, "minmaj7"),
        ("Ab^", Ab, "maj"),
        ("G7alt", G, "dom7"),
    ],
)
def test_parse_treebank_symbols(text, root, quality):
    chord = parse_symbol(text)
    assert chord is not None
    assert (chord.root, chord.quality) == (root, quality)


def test_parse_slash_chord():
    chord = parse_symbol("C/E")
    assert chord is not None and chord.root == C and chord.bass == E and chord.inversion == 1


def test_unknown_symbol_is_refused_not_guessed():
    assert parse_symbol("Cwhat9") is None
    assert parse_symbol("H7") is None


def test_dash_is_flat_in_keys_and_minor_in_chords():
    """"E-" is E flat major, "b-" is B flat minor, and "F-7" is F minor 7."""
    assert parse_key("E-") == (Eb, "major")
    assert parse_key("b-") == (Bb, "minor")
    assert parse_key("B") == (B, "major")
    chord = parse_symbol("F-7")
    assert chord is not None and (chord.root, chord.quality) == (F, "min7")


# ---------------------------------------------------------------------------
# The avoid-note model
# ---------------------------------------------------------------------------


def test_eleventh_over_major_seventh_is_the_classic_avoid_note():
    verdict = classify_melody_note(JazzChord(root=C, quality="maj7"), F)
    assert verdict.verdict == CONFLICT


def test_flat_nine_is_a_conflict_on_a_minor_chord_and_a_colour_on_a_dominant():
    assert classify_melody_note(JazzChord(root=D, quality="min7"), Eb).verdict == CONFLICT
    dominant = classify_melody_note(JazzChord(root=G, quality="dom7"), Ab)
    assert dominant.verdict == AVAILABLE_TENSION and dominant.tension == "b9"


def test_altered_tensions_are_available_on_dominants():
    g7 = JazzChord(root=G, quality="dom7")
    assert classify_melody_note(g7, Bb).tension == "#9"
    assert classify_melody_note(g7, Db).tension == "#11"
    assert classify_melody_note(g7, Eb).tension == "b13"


def test_melody_on_the_fourth_of_a_dominant_suspends_the_third():
    verdict = classify_melody_note(JazzChord(root=G, quality="dom7"), C)
    assert verdict.verdict == SOFT_CONFLICT
    assert verdict.omit == 4  # the major third steps aside


def test_conflict_against_the_fifth_is_soft_because_the_fifth_can_go():
    verdict = classify_melody_note(JazzChord(root=C, quality="maj7"), Ab)
    assert verdict.verdict == SOFT_CONFLICT
    assert verdict.omit == 7


def test_flat_seven_contradicts_a_major_seventh_chord():
    assert classify_melody_note(JazzChord(root=C, quality="maj7"), Bb).verdict == CONFLICT


def test_blue_note_is_allowed_on_a_dominant_and_refused_on_a_major_seventh():
    assert classify_melody_note(JazzChord(root=C, quality="dom7"), Eb).verdict == AVAILABLE_TENSION
    assert classify_melody_note(JazzChord(root=C, quality="maj7"), Eb).verdict == CONFLICT


def test_chord_tones_and_stated_tensions_are_never_conflicts():
    chord = JazzChord(root=C, quality="dom7", extensions=("b9",))
    assert classify_melody_note(chord, E).verdict == CHORD_TONE
    assert classify_melody_note(chord, Db).verdict == STATED_TENSION


def test_diminished_chord_avoid_notes_follow_the_diminished_scale():
    dim = JazzChord(root=C, quality="dim7")
    for available in (D, F, Ab, B):
        assert classify_melody_note(dim, available).verdict != CONFLICT
    for avoid in (Db, E, G, Bb):
        assert classify_melody_note(dim, avoid).verdict == CONFLICT


def test_supports_melody_is_the_hard_constraint():
    assert supports_melody(JazzChord(root=C, quality="dom7"), [C, E, G, Bb, D, A])
    assert not supports_melody(JazzChord(root=C, quality="maj7"), [F])


# ---------------------------------------------------------------------------
# Stating what the melody is doing
# ---------------------------------------------------------------------------


def test_absorb_melody_states_the_tension_the_melody_sits_on():
    chord = absorb_melody(JazzChord(root=G, quality="dom7"), [(Ab, 2.0)])
    assert "b9" in chord.extensions


def test_absorb_melody_ignores_passing_notes():
    chord = absorb_melody(JazzChord(root=G, quality="dom7"), [(Ab, 0.3)])
    assert chord.extensions == ()


def test_absorb_melody_refuses_tensions_the_quality_cannot_carry():
    """A min7 with a stated b13 is a colour nobody asked for."""
    chord = absorb_melody(JazzChord(root=D, quality="min7"), [(Bb, 3.0)])
    assert "b13" not in chord.extensions
