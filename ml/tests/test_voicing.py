"""Tests for voice-leading detection.

Every headline number in ml/eval rests on these functions. A parallel-fifth
detector that is subtly wrong makes the entire rules-vs-neural comparison
meaningless, which is precisely the failure mode v1 fell into.
"""

import pytest

from ml.theory.chords import ChordLabel, make_chord
from ml.theory.pitch import Key
from ml.theory.voicing import (
    ALTO,
    BASS,
    SOPRANO,
    TENOR,
    VOICE_RANGES,
    VoicedTexture,
    analyze_texture,
    count_chord_changes,
    find_direct_perfects,
    find_doubled_leading_tone,
    find_overlaps,
    find_parallels,
    find_range_violations,
    find_spacing_errors,
    find_unresolved_leading_tones,
    find_unresolved_sevenths,
    find_voice_crossings,
    melodic_defect,
    texture_from_voices,
)

C_MAJOR = Key(0, "major")


def kinds(results):
    return sorted(kind for _, _, kind in results)


class TestParallels:
    def test_textbook_parallel_fifths(self):
        # C-G moving up to D-A in the outer voices.
        prev = (None, None, 67, 60)
        curr = (None, None, 69, 62)
        assert kinds(find_parallels(prev, curr)) == ["parallel_fifths"]

    def test_textbook_parallel_octaves(self):
        prev = (72, None, None, 60)
        curr = (74, None, None, 62)
        assert kinds(find_parallels(prev, curr)) == ["parallel_octaves"]

    def test_parallel_unisons_count_as_octaves(self):
        prev = (None, 60, 60, None)
        curr = (None, 62, 62, None)
        assert kinds(find_parallels(prev, curr)) == ["parallel_octaves"]

    def test_static_fifth_is_not_a_parallel(self):
        # Nothing moves, so nothing can be parallel.
        son = (None, None, 67, 60)
        assert find_parallels(son, son) == []

    def test_oblique_motion_from_a_fifth_is_clean(self):
        prev = (None, None, 67, 60)
        curr = (None, None, 67, 64)   # only the lower voice moves
        assert find_parallels(prev, curr) == []

    def test_fifth_to_compound_fifth_is_still_parallel(self):
        # A perfect fifth expanding to a perfect twelfth in similar motion is
        # still consecutive perfect fifths; octave-blind detectors miss this.
        prev = (None, None, 67, 60)   # P5
        curr = (None, None, 81, 62)   # P12
        assert kinds(find_parallels(prev, curr)) == ["parallel_fifths"]

    def test_contrary_fifths_are_separated_from_parallel(self):
        prev = (None, None, 67, 48)   # P12
        curr = (None, None, 62, 55)   # P5 reached by contrary motion
        assert kinds(find_parallels(prev, curr)) == ["contrary_fifths"]

    def test_fifth_to_octave_is_not_flagged(self):
        prev = (None, None, 67, 60)
        curr = (None, None, 69, 57)
        assert find_parallels(prev, curr) == []

    def test_fifth_to_sixth_is_clean(self):
        prev = (None, None, 67, 60)
        curr = (None, None, 69, 60)
        assert find_parallels(prev, curr) == []

    def test_all_voice_pairs_are_checked_not_just_adjacent(self):
        # Parallel fifths between soprano and tenor, two voices apart.
        prev = (72, 69, 65, 53)
        curr = (74, 71, 67, 55)
        found = find_parallels(prev, curr)
        assert (SOPRANO, TENOR, "parallel_fifths") in found

    def test_rests_are_ignored(self):
        assert find_parallels((None, None, None, 60), (None, None, 69, 62)) == []


class TestParallelVersusContraryClassification:
    """Consecutive perfect intervals are graded, not lumped together.

    Two voices arriving at a second perfect fifth by CONTRARY motion is a real
    but much milder fault than both sliding there in the same direction. A
    detector that reported one kind for both would be easier to write and would
    make every engine look equally bad at two different things.

    The lead's fixture work surfaced this directly: a repair attempt left a
    contrary-fifths pair, which this code classified as the milder kind rather
    than as a true parallel, and asked for the behaviour to be preserved. These
    tests pin it in all four combinations plus the severity and scoring
    consequences, because the distinction is only useful if it survives.
    """

    @staticmethod
    def classify(prev, curr):
        return {kind for _, _, kind in find_parallels(prev, curr)}

    def test_fifths_in_the_same_direction_are_parallel(self):
        assert self.classify((None, None, 67, 60), (None, None, 69, 62)) == {"parallel_fifths"}

    def test_fifths_in_opposite_directions_are_contrary(self):
        assert self.classify((None, None, 67, 48), (None, None, 62, 55)) == {"contrary_fifths"}

    def test_octaves_in_the_same_direction_are_parallel(self):
        assert self.classify((None, None, 72, 60), (None, None, 74, 62)) == {"parallel_octaves"}

    def test_octaves_in_opposite_directions_are_contrary(self):
        assert self.classify((None, None, 72, 60), (None, None, 62, 74)) == {"contrary_octaves"}

    def test_similar_motion_counts_as_parallel_not_contrary(self):
        # Same direction, different distances, landing on a compound fifth.
        # Still consecutive perfect fifths; only the direction matters here.
        assert self.classify((None, None, 67, 60), (None, None, 81, 62)) == {"parallel_fifths"}

    def test_the_two_kinds_carry_different_severities(self):
        texture = texture_from_voices([[None, None], [None, None], [67, 69], [60, 62]])
        parallel = [d for d in analyze_texture(texture, C_MAJOR) if d.kind == "parallel_fifths"]
        texture = texture_from_voices([[None, None], [None, None], [67, 62], [48, 55]])
        contrary = [d for d in analyze_texture(texture, C_MAJOR) if d.kind == "contrary_fifths"]
        assert parallel and parallel[0].severity == "error"
        assert contrary and contrary[0].severity == "warning"

    def test_contrary_perfects_do_not_count_as_hard_errors(self):
        """The scoring consequence, and the reason this must not drift.

        HARD TOTAL in the report is the headline correctness number. Folding
        contrary fifths into it would silently change every row, including
        Bach's own, and make the engines look worse at something they are not
        doing.
        """
        from ml.eval.metrics import HARD_DEFECTS

        assert "parallel_fifths" in HARD_DEFECTS
        assert "parallel_octaves" in HARD_DEFECTS
        assert "contrary_fifths" not in HARD_DEFECTS
        assert "contrary_octaves" not in HARD_DEFECTS

    def test_both_kinds_are_still_reported(self):
        """Milder is not the same as hidden: the table shows both."""
        from ml.eval.metrics import DEFECT_KINDS

        for kind in ("parallel_fifths", "parallel_octaves", "contrary_fifths", "contrary_octaves"):
            assert kind in DEFECT_KINDS


class TestDirectPerfects:
    def test_direct_octave_with_soprano_leap(self):
        prev = (69, None, None, 50)
        curr = (72, None, None, 60)   # similar motion into an octave, soprano leaps
        assert kinds(find_direct_perfects(prev, curr)) == ["direct_octaves"]

    def test_soprano_stepwise_arrival_is_allowed(self):
        prev = (71, None, None, 50)
        curr = (72, None, None, 60)   # soprano moves by step: idiomatic
        assert find_direct_perfects(prev, curr) == []

    def test_contrary_motion_into_an_octave_is_allowed(self):
        prev = (65, None, None, 62)
        curr = (72, None, None, 60)
        assert find_direct_perfects(prev, curr) == []


class TestVerticalChecks:
    def test_voice_crossing_detected(self):
        assert find_voice_crossings((72, 74, 60, 48)) == [(SOPRANO, ALTO)]

    def test_no_crossing_when_properly_ordered(self):
        assert find_voice_crossings((72, 67, 60, 48)) == []

    def test_equal_pitches_are_not_crossing(self):
        assert find_voice_crossings((67, 67, 60, 48)) == []

    def test_spacing_error_between_upper_voices(self):
        errors = find_spacing_errors((79, 60, 55, 43))   # 19 semitones S-A
        assert (SOPRANO, ALTO, 19) in errors

    def test_octave_gap_is_allowed_between_upper_voices(self):
        assert find_spacing_errors((72, 60, 55, 48)) == []

    def test_bass_tenor_may_open_to_a_twelfth(self):
        assert find_spacing_errors((72, 65, 60, 41)) == []      # 19 semitones T-B
        assert find_spacing_errors((72, 65, 60, 40)) != []      # 20 is too far

    def test_range_violations(self):
        low, _ = VOICE_RANGES[BASS]
        assert find_range_violations((72, 67, 60, low - 1)) == [(BASS, low - 1)]
        assert find_range_violations((72, 67, 60, low)) == []


class TestOverlap:
    def test_lower_voice_rising_past_upper(self):
        prev = (72, 67, 60, 48)
        curr = (74, 69, 68, 48)   # tenor lands above where the alto just was
        assert (TENOR, ALTO) in find_overlaps(prev, curr)

    def test_normal_motion_has_no_overlap(self):
        assert find_overlaps((72, 67, 60, 48), (71, 65, 59, 50)) == []


class TestMelodicIntervals:
    @pytest.mark.parametrize("interval", [0, 1, 2, 3, 4, 5, 7, 8, 12])
    def test_idiomatic_intervals_are_clean(self, interval):
        assert melodic_defect(60, 60 + interval) is None
        assert melodic_defect(60, 60 - interval) is None

    @pytest.mark.parametrize("interval", [6, 10, 11, 13, 14])
    def test_awkward_intervals_are_flagged(self, interval):
        assert melodic_defect(60, 60 + interval) == "awkward_melodic_interval"

    def test_large_leap_beyond_an_octave(self):
        assert melodic_defect(48, 48 + 15) == "large_leap"


class TestTendencyTones:
    def test_leading_tone_must_rise_in_an_authentic_cadence(self):
        v = make_chord(7, "dom7", 0, "major")
        i = make_chord(0, "maj", 0, "major")
        # Soprano holds B and refuses to move to C.
        prev = (71, 67, 62, 55)
        curr = (71, 65, 60, 48)
        found = find_unresolved_leading_tones(prev, curr, v, i, C_MAJOR)
        assert (SOPRANO, "unresolved_leading_tone") in found

    def test_correct_resolution_is_clean(self):
        v = make_chord(7, "dom7", 0, "major")
        i = make_chord(0, "maj", 0, "major")
        prev = (71, 67, 62, 55)
        curr = (72, 67, 60, 48)
        assert find_unresolved_leading_tones(prev, curr, v, i, C_MAJOR) == []

    def test_frustrated_leading_tone_in_an_inner_voice_is_only_info(self):
        v = make_chord(7, "dom7", 0, "major")
        i = make_chord(0, "maj", 0, "major")
        prev = (74, 71, 65, 55)   # alto has the leading tone
        curr = (72, 67, 64, 48)   # alto drops to the fifth of I
        assert find_unresolved_leading_tones(prev, curr, v, i, C_MAJOR) == [(ALTO, "frustrated_leading_tone")]

    def test_deceptive_resolution_is_not_judged_by_this_rule(self):
        v = make_chord(7, "dom7", 0, "major")
        vi = make_chord(9, "min", 0, "major")
        prev = (71, 67, 62, 55)
        curr = (72, 69, 60, 57)
        assert find_unresolved_leading_tones(prev, curr, v, vi, C_MAJOR) == []

    def test_secondary_dominant_leading_tone(self):
        # V7/V -> V in C major: the F# must rise to G.
        v_of_v = make_chord(2, "dom7", 0, "major")
        v = make_chord(7, "maj", 0, "major")
        assert v_of_v.applied_to == 7
        prev = (66, 62, 60, 50)   # soprano F#
        curr = (65, 62, 59, 55)   # falls to F natural instead
        assert (SOPRANO, "unresolved_leading_tone") in find_unresolved_leading_tones(prev, curr, v_of_v, v, C_MAJOR)

    def test_chordal_seventh_must_fall(self):
        v7 = make_chord(7, "dom7", 0, "major")
        i = make_chord(0, "maj", 0, "major")
        prev = (77, 67, 62, 55)   # soprano F, the seventh of G7
        curr = (79, 67, 60, 48)   # leaps up instead of resolving down
        assert find_unresolved_sevenths(prev, curr, v7, i, C_MAJOR) == [SOPRANO]

    def test_chordal_seventh_resolving_down_is_clean(self):
        v7 = make_chord(7, "dom7", 0, "major")
        i = make_chord(0, "maj", 0, "major")
        prev = (77, 67, 62, 55)
        curr = (76, 67, 60, 48)
        assert find_unresolved_sevenths(prev, curr, v7, i, C_MAJOR) == []

    def test_doubled_leading_tone(self):
        v = make_chord(7, "maj", 0, "major")
        assert sorted(find_doubled_leading_tone((71, 67, 59, 55), v, C_MAJOR)) == [SOPRANO, TENOR]
        assert find_doubled_leading_tone((71, 67, 62, 55), v, C_MAJOR) == []


class TestTexture:
    def test_changes_collapses_held_chords(self):
        texture = texture_from_voices([[72, 72, 74], [67, 67, 65], [64, 64, 62], [48, 48, 55]])
        assert [index for index, _ in texture.changes()] == [0, 2]
        assert count_chord_changes(texture) == 1

    def test_held_chord_produces_no_parallels(self):
        # Four sixteenths of the same chord must not be scored as three
        # transitions; that would inflate every per-chord rate.
        texture = texture_from_voices([[72] * 4, [67] * 4, [64] * 4, [48] * 4])
        assert analyze_texture(texture, C_MAJOR) == []

    def test_voice_line_skips_holds(self):
        texture = texture_from_voices([[72, 72, 74, 74], [67] * 4, [64] * 4, [48] * 4])
        assert texture.voice_line(0) == [(0, 72), (2, 74)]

    def test_repeated_note_after_a_change_is_a_new_line_entry(self):
        texture = texture_from_voices([[72, 74, 72], [67] * 3, [64] * 3, [48] * 3])
        assert texture.voice_line(0) == [(0, 72), (1, 74), (2, 72)]

    def test_analyze_texture_finds_parallel_fifths_with_offsets(self):
        texture = texture_from_voices([[72, 74], [67, 69], [64, 66], [60, 62]], step=0.5)
        defects = analyze_texture(texture, C_MAJOR)
        parallels = [d for d in defects if d.kind == "parallel_fifths"]
        assert parallels and all(d.offset == 0.5 for d in parallels)

    def test_analyze_texture_is_deterministic(self):
        texture = texture_from_voices([[72, 74, 71], [67, 69, 67], [64, 66, 62], [48, 50, 55]])
        assert analyze_texture(texture, C_MAJOR) == analyze_texture(texture, C_MAJOR)

    def test_clean_progression_is_reported_clean(self):
        # I - V6 - I in C major, correctly voiced: the bass leading tone rises,
        # nothing doubles it, and the soprano pedals on G.
        texture = texture_from_voices([[67, 67, 67], [64, 62, 60], [60, 55, 52], [48, 47, 48]])
        chords = [make_chord(0, "maj", 0, "major"), make_chord(7, "maj", 1, "major"), make_chord(0, "maj", 0, "major")]
        defects = [d for d in analyze_texture(texture, C_MAJOR, chords) if d.severity != "info"]
        assert defects == []
