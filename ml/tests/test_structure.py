"""Tests for structural defect detection.

The category that did not exist: every detector in `voicing.py` looks at two
sonorities at a time, so **a harmonization that ends on the dominant scored zero
defects**. It breaks no voice-leading rule, and it is audibly wrong in a way no
parallel fifth ever is.

The calibration discipline matters more here than anywhere else, because
structural rules written from theory are confidently wrong. "A piece must end on
the tonic" flags 15% of Bach: he closes ~8% of chorales on a root-position V, and
checking the pitch content shows only 1 of those 22 has a mistaken key label.
Those are Phrygian half cadences. So the oracle test below is not a formality —
it is the thing that stops this module inventing errors.
"""

import pytest

from ml.theory.chords import make_chord
from ml.theory.pitch import Key
from ml.theory.structure import (
    DESCRIPTIVE_KINDS,
    STRUCTURAL_KINDS,
    find_structural_defects,
    phrase_end_beats,
)

C_MAJOR = Key(0, "major")
A_MINOR = Key(9, "minor")


def progression(specs, mode="major"):
    return [make_chord(root, quality, inversion, mode) for root, quality, inversion in specs]


def kinds(defects):
    return sorted(d.kind for d in defects)


I = (0, "maj", 0)
V = (7, "maj", 0)
IV = (5, "maj", 0)
ii = (2, "min", 0)
vi = (9, "min", 0)
viio = (11, "dim", 0)
#: The Neapolitan. Bach uses it, but never to close a phrase in a major key -
#: unlike viio, which the measured table DOES accept as a phrase ending. That
#: surprise is exactly why these tables are calibrated rather than written.
neapolitan = (1, "maj", 0)


class TestTonalClosure:
    def test_a_normal_ending_is_clean(self):
        chords = progression([I, IV, V, I])
        assert find_structural_defects(chords, C_MAJOR) == []

    def test_ending_on_a_supertonic_has_no_closure(self):
        """The gap this module exists to close: nothing in `voicing.py` sees it."""
        chords = progression([I, IV, V, ii])
        assert "no_tonal_closure" in kinds(find_structural_defects(chords, C_MAJOR))

    def test_ending_on_a_leading_tone_chord_has_no_closure(self):
        chords = progression([I, IV, V, viio])
        assert "no_tonal_closure" in kinds(find_structural_defects(chords, C_MAJOR))

    def test_ending_on_an_inverted_tonic_is_tolerated(self):
        # Rare in Bach but not structurally broken; the tier is for "wrong",
        # not "unusual".
        chords = progression([I, IV, V, (0, "maj", 1)])
        assert "no_tonal_closure" not in kinds(find_structural_defects(chords, C_MAJOR))

    def test_ending_on_the_dominant_is_reported_but_not_a_defect(self):
        """Bach ends 9.2% of chorales this way, so it is not an error — but an
        engine doing it constantly is broken, so it must still be visible."""
        chords = progression([I, IV, ii, V])
        found = find_structural_defects(chords, C_MAJOR)
        assert kinds(found) == ["half_cadence_ending"]
        assert found[0].severity == "info"
        assert found[0].kind not in STRUCTURAL_KINDS
        assert found[0].kind in DESCRIPTIVE_KINDS

    def test_a_picardy_third_closes_a_minor_piece(self):
        chords = progression([(0, "min", 0), (5, "min", 0), V, I], mode="minor")
        assert "no_tonal_closure" not in kinds(find_structural_defects(chords, A_MINOR))

    def test_minor_piece_ending_on_the_minor_tonic_is_clean(self):
        chords = progression([(0, "min", 0), (5, "min", 0), V, (0, "min", 0)], mode="minor")
        assert find_structural_defects(chords, A_MINOR) == []

    def test_trailing_silence_does_not_hide_the_ending(self):
        chords = progression([I, IV, V, ii]) + [None, None]
        assert "no_tonal_closure" in kinds(find_structural_defects(chords, C_MAJOR))

    def test_empty_input_is_safe(self):
        assert find_structural_defects([], C_MAJOR) == []
        assert find_structural_defects([None, None], C_MAJOR) == []


class TestKeyEstablishment:
    def test_an_opening_tonic_establishes_the_key(self):
        assert "key_not_established" not in kinds(find_structural_defects(progression([I, V, I]), C_MAJOR))

    def test_a_late_tonic_still_counts(self):
        chords = progression([V, IV, ii, V, I])
        assert "key_not_established" not in kinds(find_structural_defects(chords, C_MAJOR))

    def test_never_reaching_a_tonic_early_is_flagged(self):
        chords = progression([V] * 14 + [I])
        assert "key_not_established" in kinds(find_structural_defects(chords, C_MAJOR))

    def test_the_flag_is_an_error(self):
        chords = progression([V] * 14 + [I])
        found = [d for d in find_structural_defects(chords, C_MAJOR) if d.kind == "key_not_established"]
        assert found and found[0].severity == "error"


class TestPhraseCadences:
    def test_a_plausible_phrase_ending_is_clean(self):
        chords = progression([I, V, I, IV, V, I])
        phrases = [False, True, False, False, True, False]
        found = find_structural_defects(chords, C_MAJOR, phrase_ends=phrases)
        assert "implausible_phrase_cadence" not in kinds(found)

    def test_a_phrase_ending_bach_never_writes_is_flagged(self):
        chords = progression([I, neapolitan, I, IV, V, I])
        phrases = [False, True, False, False, False, False]
        found = find_structural_defects(chords, C_MAJOR, phrase_ends=phrases)
        assert "implausible_phrase_cadence" in kinds(found)

    def test_a_leading_tone_chord_closing_a_phrase_is_accepted(self):
        """Measured, not assumed: Bach does close phrases on viio often enough
        that flagging it would be inventing an error."""
        chords = progression([I, viio, I, IV, V, I])
        phrases = [False, True, False, False, False, False]
        found = find_structural_defects(chords, C_MAJOR, phrase_ends=phrases)
        assert "implausible_phrase_cadence" not in kinds(found)

    def test_the_final_cadence_is_not_double_counted(self):
        """The last phrase is judged by `no_tonal_closure`, not twice."""
        chords = progression([I, IV, V, ii])
        phrases = [False, False, False, True]
        found = kinds(find_structural_defects(chords, C_MAJOR, phrase_ends=phrases))
        assert found.count("no_tonal_closure") == 1
        assert "implausible_phrase_cadence" not in found

    def test_only_the_last_beat_of_a_phrase_is_judged(self):
        # A phrase-final note spanning two beats must be judged once, at its end.
        chords = progression([I, neapolitan, neapolitan, I])
        phrases = [False, True, True, False]
        found = find_structural_defects(chords, C_MAJOR, phrase_ends=phrases)
        assert kinds(found).count("implausible_phrase_cadence") == 1

    def test_no_phrase_information_skips_the_check(self):
        chords = progression([I, viio, I, IV, V, I])
        found = find_structural_defects(chords, C_MAJOR, phrase_ends=None)
        assert "implausible_phrase_cadence" not in kinds(found)


class TestPhraseEndBeats:
    def test_collapses_a_step_mask_onto_beats(self):
        mask = [False] * 4 + [True] * 4 + [False] * 4
        assert phrase_end_beats(mask, 4) == [False, True, False]

    def test_a_partial_beat_still_counts(self):
        mask = [False, False, True, False]
        assert phrase_end_beats(mask, 4) == [True]

    def test_handles_a_ragged_tail(self):
        assert phrase_end_beats([False] * 6, 4) == [False, False]


class TestOracleCalibration:
    """Bach must score ~0 structurally. If he does not, the detector is wrong.

    This is the check that would have caught "a piece must end on the tonic",
    which flags 15% of the corpus.
    """

    @pytest.fixture(scope="class")
    def bach(self):
        from ml.data.corpus import load_chorales
        from ml.eval.metrics import DefectCounts, collect_defects

        chorales = load_chorales()
        counts = DefectCounts()
        for chorale in chorales:
            counts.merge(collect_defects(chorale.voices, chorale.key, phrase_ends=chorale.fermatas))
        return counts

    def test_bach_is_near_zero_on_the_structural_tier(self, bach):
        assert bach.structural_rate() < 0.05, (
            f"structural detectors fire on Bach at {bach.structural_rate():.4f}/piece; "
            "the detector is wrong, not Bach"
        )

    def test_no_single_structural_rule_fires_often_on_bach(self, bach):
        for kind in STRUCTURAL_KINDS:
            rate = bach.per_piece(kind)
            assert rate < 0.03, f"{kind} fires on {100 * rate:.1f}% of Bach's chorales"

    def test_bach_does_end_on_the_dominant_sometimes(self, bach):
        """Guards the other direction: if this ever reads zero, the descriptive
        flag has silently stopped working and the calibration above is vacuous."""
        assert bach.per_piece("half_cadence_ending") > 0.02

    def test_the_tiers_are_disjoint_and_cover_every_kind(self):
        from ml.eval.metrics import (
            DEFECT_KINDS,
            DESCRIPTIVE_KINDS as METRIC_DESCRIPTIVE,
            HARD_DEFECTS,
            SOFT_DEFECTS,
            STRUCTURAL_DEFECTS,
        )

        tiers = [set(STRUCTURAL_DEFECTS), set(HARD_DEFECTS), set(SOFT_DEFECTS), set(METRIC_DESCRIPTIVE)]
        for i, first in enumerate(tiers):
            for second in tiers[i + 1:]:
                assert not (first & second), f"overlapping tiers: {first & second}"
        assert set(DEFECT_KINDS) == set().union(*tiers)

    def test_voice_crossing_is_soft_not_hard(self):
        """It dominated the old hard figure for Bach (3.33 of 3.73) while being
        the least audible thing in it."""
        from ml.eval.metrics import HARD_DEFECTS, SOFT_DEFECTS

        assert "voice_crossing" in SOFT_DEFECTS
        assert "voice_crossing" not in HARD_DEFECTS
