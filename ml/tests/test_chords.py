"""Tests for the chord vocabulary and analyser.

v1's label space was seven diatonic triads and its analyser was a +1/-1
pitch-class vote, so a V7 became V and a V/V became ii. These tests pin down
that the replacement actually represents what it claims to, and cross-check the
roman numerals against music21's independent analyser.
"""

import pytest

from ml.theory.chords import (
    QUALITY_TEMPLATES,
    ChordLabel,
    analyze_chord,
    build_vocabulary,
    chord_pitch_classes,
    diatonic_collection,
    infer_applied_target,
    make_chord,
)
from ml.theory.pitch import Key

C_MAJOR = Key(0, "major")
A_MINOR = Key(9, "minor")


class TestChordLabel:
    def test_rejects_impossible_inversion(self):
        with pytest.raises(ValueError):
            ChordLabel(0, "maj", 3)
        with pytest.raises(ValueError):
            ChordLabel(0, "nonsense", 0)

    def test_pitch_classes_are_tonic_relative(self):
        assert make_chord(7, "dom7", 0, "major").pitch_classes == (7, 11, 2, 5)

    def test_bass_follows_inversion(self):
        v7 = ChordLabel(7, "dom7", 0)
        assert [ChordLabel(7, "dom7", i).bass_relative_pc for i in range(4)] == [7, 11, 2, 5]
        assert v7.seventh_relative_pc == 5

    def test_triads_have_no_seventh(self):
        assert make_chord(0, "maj", 0, "major").seventh_relative_pc is None

    @pytest.mark.parametrize(
        "root,quality,inversion,mode,expected",
        [
            (7, "dom7", 0, "major", "V7"),
            (7, "dom7", 1, "major", "V65"),
            (7, "dom7", 2, "major", "V43"),
            (7, "dom7", 3, "major", "V42"),
            (0, "maj", 2, "major", "I64"),
            (2, "min", 1, "major", "ii6"),
            (11, "dim", 1, "major", "viio6"),
            (11, "halfdim7", 0, "major", "vii\u00f87"),
            (11, "dim7", 0, "minor", "viio7"),
            (1, "maj", 1, "major", "bII6"),          # Neapolitan, not "BII6"
            (8, "maj", 0, "major", "bVI"),           # borrowed
            (0, "min", 0, "minor", "i"),
            (10, "maj", 0, "minor", "VII"),          # subtonic, natural minor
            (5, "min", 0, "minor", "iv"),
        ],
    )
    def test_roman_numerals(self, root, quality, inversion, mode, expected):
        assert make_chord(root, quality, inversion, mode).roman(mode) == expected

    def test_secondary_dominants_render_with_a_slash(self):
        assert make_chord(2, "dom7", 0, "major").roman("major") == "V7/V"
        assert make_chord(4, "dom7", 0, "major").roman("major") == "V7/vi"
        assert make_chord(9, "maj", 0, "major").roman("major") == "V/ii"
        assert make_chord(6, "dim7", 0, "major").roman("major") == "viio7/V"

    def test_transposition_invariance(self):
        # The same label means the same thing in every key: that invariance is
        # what makes the tonic-relative representation learnable.
        chord = make_chord(7, "dom7", 1, "major")
        for tonic in range(12):
            key = Key(tonic, "major")
            assert set(chord.absolute_pitch_classes(key)) == {(tonic + i) % 12 for i in (7, 11, 2, 5)}


class TestAppliedChords:
    def test_diatonic_chords_are_never_applied(self):
        for root, quality in ((0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min"), (11, "dim")):
            assert infer_applied_target(root, quality, "major") is None
        # V and viio in minor use the raised 7 and are still native, not secondary.
        assert infer_applied_target(7, "maj", "minor") is None
        assert infer_applied_target(7, "dom7", "minor") is None
        assert infer_applied_target(11, "dim7", "minor") is None

    def test_secondary_dominants_are_detected(self):
        assert infer_applied_target(2, "dom7", "major") == 7    # V7/V
        assert infer_applied_target(4, "dom7", "major") == 9    # V7/vi
        assert infer_applied_target(9, "dom7", "major") == 2    # V7/ii
        assert infer_applied_target(6, "dim7", "major") == 7    # viio7/V

    def test_diatonic_collection_includes_the_raised_seventh_in_minor(self):
        assert 11 in diatonic_collection("minor")
        assert 10 in diatonic_collection("minor")
        assert 1 not in diatonic_collection("major")


class TestAnalyzer:
    def test_plain_triad(self):
        chord = analyze_chord([48, 64, 67, 72], C_MAJOR)
        assert (chord.relative_root, chord.quality, chord.inversion) == (0, "maj", 0)

    def test_first_inversion_read_from_the_bass(self):
        chord = analyze_chord([52, 60, 67, 76], C_MAJOR)
        assert (chord.relative_root, chord.quality, chord.inversion) == (0, "maj", 1)

    def test_dominant_seventh_is_not_flattened_to_a_triad(self):
        # This is exactly the case v1's +1/-1 vote destroyed.
        chord = analyze_chord([55, 65, 71, 74], C_MAJOR)
        assert (chord.relative_root, chord.quality) == (7, "dom7")

    def test_secondary_dominant_is_not_flattened_to_a_diatonic_chord(self):
        chord = analyze_chord([50, 66, 72, 81], C_MAJOR)   # D F# C A = V7/V
        assert chord.quality == "dom7"
        assert chord.relative_root == 2
        assert chord.applied_to == 7

    def test_half_diminished_and_fully_diminished_are_distinguished(self):
        assert analyze_chord([59, 62, 65, 69], C_MAJOR).quality == "halfdim7"
        assert analyze_chord([59, 62, 65, 68], C_MAJOR).quality == "dim7"

    def test_minor_key_dominant_uses_the_raised_seventh(self):
        chord = analyze_chord([52, 68, 71, 76], A_MINOR)   # E G# B E in A minor
        assert (chord.relative_root, chord.quality) == (7, "maj")
        assert chord.applied_to is None

    def test_doubling_does_not_change_the_root(self):
        assert analyze_chord([48, 60, 64, 67], C_MAJOR).relative_root == 0
        assert analyze_chord([48, 60, 72, 64], C_MAJOR).relative_root == 0

    def test_empty_input_returns_none(self):
        assert analyze_chord([], C_MAJOR) is None

    def test_analysis_is_transposition_equivariant(self):
        base = analyze_chord([55, 65, 71, 74], C_MAJOR)
        for tonic in range(1, 12):
            moved = analyze_chord([p + tonic for p in [55, 65, 71, 74]], Key(tonic, "major"))
            assert (moved.relative_root, moved.quality, moved.inversion) == (
                base.relative_root, base.quality, base.inversion
            )

    def test_a_passing_tone_does_not_derail_the_root(self):
        # C major triad with a D sounding as a passing tone in an inner voice.
        chord = analyze_chord([48, 62, 64, 72], C_MAJOR)
        assert chord.relative_root == 0 and chord.quality == "maj"


class TestVocabulary:
    def test_covers_the_things_v1_could_not_represent(self):
        vocab = build_vocabulary("major")
        keys = {(c.relative_root, c.quality, c.inversion) for c in vocab}
        assert (7, "dom7", 0) in keys           # V7
        assert (7, "dom7", 2) in keys           # V43
        assert (2, "dom7", 0) in keys           # V7/V
        assert (11, "dim7", 0) in keys          # viio7
        assert (1, "maj", 1) in keys            # Neapolitan
        assert (8, "maj", 0) in keys            # bVI
        assert (0, "maj", 2) in keys            # cadential 6-4
        assert len(vocab) > 90

    def test_no_duplicates(self):
        for mode in ("major", "minor"):
            vocab = build_vocabulary(mode)
            keys = [(c.relative_root, c.quality, c.inversion) for c in vocab]
            assert len(keys) == len(set(keys))

    def test_every_entry_is_constructible(self):
        for mode in ("major", "minor"):
            for chord in build_vocabulary(mode):
                assert len(chord.pitch_classes) == len(QUALITY_TEMPLATES[chord.quality])
                assert chord.roman(mode)


class TestAgainstMusic21:
    """Cross-check the analyser against music21's independent implementation.

    Only unambiguous textbook cases are compared: the two systems make different
    (equally defensible) choices about added-sixth and quartal sonorities, and
    pinning those would test music21 rather than this code.
    """

    CASES = [
        ("C", "major", [48, 60, 64, 67], 0, "maj", 0),
        ("C", "major", [52, 60, 67, 76], 0, "maj", 1),
        ("C", "major", [55, 62, 71, 77], 7, "dom7", 0),
        ("C", "major", [50, 57, 65, 74], 2, "min", 0),
        ("C", "major", [59, 62, 65, 71], 11, "dim", 0),
        ("G", "major", [55, 62, 71, 79], 0, "maj", 0),
        ("Eb", "major", [51, 58, 67, 75], 0, "maj", 0),
        ("A", "minor", [45, 52, 60, 69], 0, "min", 0),
        ("A", "minor", [52, 56, 59, 68], 7, "maj", 0),
    ]

    @pytest.mark.parametrize("tonic_name,mode,pitches,rel_root,quality,inversion", CASES)
    def test_matches_music21(self, tonic_name, mode, pitches, rel_root, quality, inversion):
        music21 = pytest.importorskip("music21")
        m21_chord = music21.chord.Chord(sorted(pitches))
        key = Key(music21.pitch.Pitch(tonic_name).pitchClass, mode)

        ours = analyze_chord(pitches, key)
        assert (ours.relative_root, ours.quality, ours.inversion) == (rel_root, quality, inversion)

        # music21 agrees on root and inversion, reached by a different algorithm.
        assert m21_chord.root().pitchClass == (rel_root + key.tonic) % 12
        assert m21_chord.inversion() == inversion
