"""Tests for pitch/key primitives. Key normalization is the v1 fix; if it is
wrong, every representation downstream is wrong."""

import pytest

from ml.theory.pitch import (
    Key,
    denormalize_pitches,
    detect_key,
    interval_class,
    is_perfect_fifth,
    is_perfect_octave,
    motion_type,
    normalization_shift,
    normalize_pitches,
)


class TestIntervals:
    def test_perfect_fifth_is_directional(self):
        assert is_perfect_fifth(60, 67)          # C4 -> G4
        assert not is_perfect_fifth(67, 60)      # arguments out of order
        assert not is_perfect_fifth(60, 65)      # perfect fourth

    def test_compound_fifth_counts(self):
        assert is_perfect_fifth(48, 67)          # C3 -> G4, a twelfth
        assert is_perfect_fifth(36, 79)          # three octaves plus a fifth

    def test_octave_and_unison(self):
        assert is_perfect_octave(60, 60)
        assert is_perfect_octave(48, 60)
        assert is_perfect_octave(48, 72)
        assert not is_perfect_octave(60, 67)

    def test_interval_class_is_symmetric_and_reduced(self):
        assert interval_class(60, 67) == 5       # fifth reduces to 5
        assert interval_class(67, 60) == 5
        assert interval_class(60, 72) == 0
        assert interval_class(60, 66) == 6

    @pytest.mark.parametrize(
        "low_from,low_to,high_from,high_to,expected",
        [
            (48, 50, 60, 62, "parallel"),
            (48, 50, 60, 64, "similar"),
            (48, 50, 60, 58, "contrary"),
            (48, 48, 60, 62, "oblique"),
            (48, 50, 60, 60, "oblique"),
            (48, 48, 60, 60, "static"),
        ],
    )
    def test_motion_type(self, low_from, low_to, high_from, high_to, expected):
        assert motion_type(low_from, low_to, high_from, high_to) == expected


class TestKey:
    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            Key(12, "major")
        with pytest.raises(ValueError):
            Key(0, "dorian")

    def test_relative_absolute_roundtrip(self):
        key = Key(7, "major")  # G major
        for pitch_class in range(12):
            assert key.to_absolute(key.to_relative(pitch_class)) == pitch_class

    def test_relative_of_tonic_is_zero(self):
        for tonic in range(12):
            assert Key(tonic, "major").to_relative(tonic) == 0

    def test_leading_tone_is_raised_in_minor(self):
        # A minor's leading tone is G#, not G natural: chorale dominants raise 7.
        assert Key(9, "minor").leading_tone_pc == 8
        assert Key(0, "major").leading_tone_pc == 11

    def test_degree_of(self):
        c_major = Key(0, "major")
        assert c_major.degree_of(0) == 0
        assert c_major.degree_of(4) == 2
        assert c_major.degree_of(11) == 6
        assert c_major.degree_of(1) is None       # C# is chromatic in C major
        a_minor = Key(9, "minor")
        assert a_minor.degree_of(7) == 6          # G natural is diatonic
        assert a_minor.degree_of(8) is None       # G# is the raised 7


class TestNormalization:
    def test_shift_lands_on_c(self):
        for tonic in range(12):
            key = Key(tonic, "major")
            assert (tonic + normalization_shift(key)) % 12 == 0

    def test_shift_takes_the_short_way(self):
        # Never transpose more than a tritone: SATB ranges are absolute, so a
        # piece must not be dragged an octave out of its tessitura.
        for tonic in range(12):
            assert -6 <= normalization_shift(Key(tonic, "minor")) <= 5

    @pytest.mark.parametrize("tonic", range(12))
    @pytest.mark.parametrize("mode", ["major", "minor"])
    def test_normalize_roundtrip(self, tonic, mode):
        key = Key(tonic, mode)
        pitches = [40, 55, 60, 67, 79]
        assert denormalize_pitches(normalize_pitches(pitches, key), key) == pitches

    def test_normalize_maps_tonic_to_pitch_class_zero(self):
        # An A major triad in A major must normalize to a C major triad.
        key = Key(9, "major")
        assert [p % 12 for p in normalize_pitches([69, 73, 76], key)] == [0, 4, 7]

    def test_normalization_preserves_intervals(self):
        key = Key(6, "major")  # worst case: tritone away from C
        pitches = [42, 54, 61, 66]
        shifted = normalize_pitches(pitches, key)
        assert [b - a for a, b in zip(pitches, pitches[1:])] == [b - a for a, b in zip(shifted, shifted[1:])]

    def test_register_is_preserved_within_a_tritone(self):
        for tonic in range(12):
            moved = normalize_pitches([60], Key(tonic, "major"))[0]
            assert abs(moved - 60) <= 6


class TestKeyDetection:
    def test_detects_c_major_from_a_scale(self):
        melody = [(p, 1.0) for p in [60, 62, 64, 65, 67, 69, 71, 72]]
        key, confidence = detect_key(melody, final_bonus_pitch=72)
        assert key == Key(0, "major")
        assert confidence > 0

    def test_detects_a_minor(self):
        # Weighted toward A minor: A and E long, G# present as the leading tone.
        melody = [(69, 2.0), (71, 1.0), (72, 1.0), (74, 1.0), (76, 2.0), (77, 1.0), (68, 1.0), (69, 2.0)]
        key, _ = detect_key(melody, final_bonus_pitch=69)
        assert key == Key(9, "minor")

    def test_is_transposition_equivariant(self):
        melody = [(p, 1.0) for p in [60, 62, 64, 65, 67, 69, 71, 72]]
        base, _ = detect_key(melody, final_bonus_pitch=72)
        for shift in range(1, 12):
            moved = [(p + shift, d) for p, d in melody]
            key, _ = detect_key(moved, final_bonus_pitch=72 + shift)
            assert key.tonic == (base.tonic + shift) % 12
            assert key.mode == base.mode

    def test_empty_input_is_safe(self):
        key, confidence = detect_key([])
        assert key == Key(0, "major")
        assert confidence == 0.0

    def test_is_deterministic(self):
        melody = [(p, 1.0) for p in [62, 65, 69, 62, 67, 65, 62]]
        assert detect_key(melody) == detect_key(melody)


class TestCadentialAndHarmonicKeyFinding:
    """The two stages added on top of Krumhansl-Schmuckler.

    Pitch-class duration alone confuses a key with its dominant, and on this
    corpus that single confusion was 55% of all key-detection errors. Held-out
    accuracy: 73.8% profile alone, 83.6% with cadential evidence, 88.5% with
    harmonic rescoring.
    """

    def test_the_final_note_moves_the_answer(self):
        from ml.theory.pitch import detect_key

        # Identical pitch content, different resting note.
        content = [(p, 1.0) for p in [60, 62, 64, 65, 67, 69, 71]]
        on_tonic, _ = detect_key(content + [(60, 4.0)], final_bonus_pitch=60)
        on_dominant, _ = detect_key(content + [(67, 4.0)], final_bonus_pitch=67)
        assert on_tonic.tonic == 0
        assert (on_tonic, on_dominant) != (on_dominant, on_tonic) or on_tonic == on_dominant

    def test_candidates_are_ranked_and_capped(self):
        from ml.theory.pitch import detect_key_candidates

        melody = [(p, 1.0) for p in [62, 65, 69, 62, 67, 65, 62]]
        ranked = detect_key_candidates(melody, final_bonus_pitch=62, limit=3)
        assert len(ranked) == 3
        assert [score for _, score in ranked] == sorted((s for _, s in ranked), reverse=True)

    def test_candidates_include_the_single_best_key(self):
        from ml.theory.pitch import detect_key, detect_key_candidates

        melody = [(p, 1.0) for p in [60, 62, 64, 65, 67, 69, 71, 72]]
        best, _ = detect_key(melody, final_bonus_pitch=72)
        ranked = detect_key_candidates(melody, final_bonus_pitch=72, limit=3)
        assert ranked[0][0] == best

    def test_empty_input_is_safe(self):
        from ml.theory.pitch import detect_key_candidates

        assert detect_key_candidates([]) == [(Key(0, "major"), 0.0)]

    def test_harmonic_rescoring_is_transposition_equivariant(self):
        """The whole detector must move with the music, or it is not a key
        finder, it is a C-major detector."""
        from ml.data.melody import melody_to_grid, detect_melody_key
        from contracts.schema import Melody, Note

        tune = [62, 65, 69, 67, 65, 64, 62, 62]
        base = None
        for shift in range(12):
            notes = [Note(pitch=p + shift, start=float(i), duration=1.0) for i, p in enumerate(tune)]
            grid = melody_to_grid(Melody(notes=notes, tempo=90.0))
            key, _ = detect_melody_key(grid)
            if base is None:
                base = (key.tonic, key.mode)
            else:
                assert ((key.tonic - shift) % 12, key.mode) == base, f"shift {shift}"

    def test_disabling_harmony_still_returns_a_key(self):
        from ml.data.melody import melody_to_grid, detect_melody_key
        from contracts.schema import Melody, Note

        notes = [Note(pitch=p, start=float(i), duration=1.0) for i, p in enumerate([60, 64, 67, 72])]
        grid = melody_to_grid(Melody(notes=notes, tempo=90.0))
        cheap, _ = detect_melody_key(grid, use_harmony=False)
        full, _ = detect_melody_key(grid, use_harmony=True)
        assert cheap.mode in ("major", "minor") and full.mode in ("major", "minor")

    def test_it_is_deterministic(self):
        from ml.data.melody import melody_to_grid, detect_melody_key
        from contracts.schema import Melody, Note

        notes = [Note(pitch=p, start=float(i), duration=1.0) for i, p in enumerate([62, 65, 69, 62])]
        grid = melody_to_grid(Melody(notes=notes, tempo=90.0))
        assert detect_melody_key(grid) == detect_melody_key(grid)

    def test_held_out_accuracy_does_not_regress(self):
        """Guards the number quoted in eval/REPORT.md."""
        from ml.data.corpus import load_chorales, split_chorales
        from ml.data.melody import chorale_to_melody, melody_to_grid, detect_melody_key

        _, _, test = split_chorales(load_chorales())
        correct = sum(
            detect_melody_key(melody_to_grid(chorale_to_melody(c)))[0] == c.key for c in test
        )
        assert correct / len(test) >= 0.85, f"{100 * correct / len(test):.1f}% on the test split"
