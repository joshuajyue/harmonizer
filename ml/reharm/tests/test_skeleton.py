"""The skeleton: from the rules engine's chorale harmony to jazz units.

Two things here have already been wrong once and are pinned accordingly. The
unit grid must depend only on the metre and the melody, never on which chord
the rules engine picked, or the same tune in a different key comes out with a
different number of chords. And the melody must be octave-normalized before the
rules engine sees it, because that engine voices real SATB parts and returns
almost nothing for a tune outside soprano range.
"""

import pytest

from contracts.schema import KeySignature, Melody, Note, TimeSignature
from ml.reharm.chords import JazzChord
from ml.reharm.data import ChordSpan
from ml.reharm.melodies import TRADITIONAL
from ml.reharm.skeleton import (
    SKELETON_WINDOW,
    build_units,
    chord_to_jazz,
    clip_melody,
    melody_notes,
    octave_shift_for,
    skeleton_from_rules,
    transpose,
)

C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)


def melody(pitches, *, start: float = 0.0, duration: float = 1.0) -> Melody:
    return Melody(
        notes=[
            Note(pitch=pitch, start=start + index * duration, duration=duration)
            for index, pitch in enumerate(pitches)
        ],
        tempo=120.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=0, mode="major"),
    )


def spans(*roots_and_durations, quality: str = "maj") -> list[ChordSpan]:
    out = []
    offset = 0.0
    for root, duration in roots_and_durations:
        out.append(ChordSpan(offset, duration, JazzChord(root=root, quality=quality)))
        offset += duration
    return out


# ---------------------------------------------------------------------------
# The unit grid
# ---------------------------------------------------------------------------


def test_a_bar_the_melody_does_not_articulate_stays_one_unit():
    line = [(0.0, 60, 4.0)]  # one whole note
    units = build_units(spans((C, 2.0), (G, 2.0)), line)
    assert len(units) == 1
    assert units[0].duration == pytest.approx(4.0)


def test_a_bar_the_melody_articulates_at_the_half_splits():
    line = [(0.0, 60, 2.0), (2.0, 62, 2.0)]
    units = build_units(spans((C, 2.0), (G, 2.0)), line)
    assert len(units) == 2
    assert [unit.duration for unit in units] == [2.0, 2.0]
    assert [unit.base.root for unit in units] == [C, G]


def test_the_grid_ignores_which_chords_the_skeleton_picked():
    """Same melody, different harmony underneath, identical segmentation.

    This is the invariant that stops a transposition bug: chord choice depends
    on register because SATB ranges are absolute, so a grid that depends on
    chord choice makes the chord COUNT depend on register.
    """
    line = [(0.0, 60, 2.0), (2.0, 62, 2.0), (4.0, 64, 4.0)]
    one = build_units(spans((C, 2.0), (G, 2.0), (F, 4.0)), line)
    two = build_units(spans((A, 2.0), (A, 2.0), (D, 4.0)), line)
    assert [(unit.start, unit.duration) for unit in one] == [(unit.start, unit.duration) for unit in two]


def test_three_four_bars_are_never_split():
    line = [(0.0, 60, 1.0), (1.0, 62, 1.0), (2.0, 64, 1.0)]
    units = build_units(spans((C, 1.0), (G, 1.0), (C, 1.0)), line, meter=(3, 4))
    assert len(units) == 1 and units[0].duration == pytest.approx(3.0)


def test_the_last_unit_is_marked():
    line = [(0.0, 60, 2.0), (2.0, 62, 2.0)]
    units = build_units(spans((C, 2.0), (G, 2.0)), line)
    assert units[-1].is_last and not any(unit.is_last for unit in units[:-1])


def test_the_dominant_chord_of_a_unit_wins_on_weight():
    line = [(0.0, 60, 4.0)]
    units = build_units(spans((C, 3.0), (G, 1.0)), line)
    assert units[0].base.root == C


def test_units_carry_the_melody_that_sounds_over_them():
    line = [(0.0, 60, 2.0), (2.0, 67, 2.0)]
    units = build_units(spans((C, 2.0), (G, 2.0)), line)
    assert [note[1] for note in units[0].melody] == [60]
    assert [note[1] for note in units[1].melody] == [67]


def test_weighted_pitch_classes_rank_by_harmonic_weight():
    line = [(0.0, 60, 3.5), (3.5, 61, 0.5)]
    units = build_units(spans((C, 4.0)), line)
    ranked = units[0].weighted_pcs
    assert ranked[0][0] == 0 and ranked[0][1] > ranked[1][1]


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------


def test_clip_melody_trims_to_the_window():
    line = [(0.0, 60, 8.0)]
    clipped = clip_melody(line, 2.0, 4.0)
    assert clipped == [(2.0, 60, 2.0)]


def test_clip_melody_drops_notes_that_do_not_sound():
    assert clip_melody([(0.0, 60, 1.0)], 2.0, 4.0) == []


def test_melody_notes_stays_in_absolute_time():
    """It must NOT rebase to zero.

    This test asserted the opposite until a melody starting at bar 5 came back
    at bar 9. The rules engine reports its chords in absolute time; rebasing
    the melody and not the harmony put them in different frames, so the units
    were handed the wrong notes. One frame, and it is the absolute one.
    """
    line = melody_notes(melody([60, 62], start=16.0))
    assert line[0][0] == pytest.approx(16.0)
    assert line[1][0] == pytest.approx(17.0)


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


def test_octave_shift_brings_a_low_melody_into_range():
    low = melody([36, 38, 40, 41])
    shift = octave_shift_for(low)
    assert shift % 12 == 0 and shift > 0
    median = sorted(note.pitch + shift for note in low.notes)[len(low.notes) // 2]
    assert SKELETON_WINDOW[0] <= median <= SKELETON_WINDOW[1]


def test_octave_shift_brings_a_high_melody_into_range():
    high = melody([96, 98, 100, 101])
    shift = octave_shift_for(high)
    assert shift % 12 == 0 and shift < 0


def test_a_melody_already_in_range_is_left_alone():
    assert octave_shift_for(melody([60, 64, 67, 72])) == 0


def test_transpose_preserves_rhythm_and_clamps_to_midi():
    moved = transpose(melody([60, 62]), 12)
    assert [note.pitch for note in moved.notes] == [72, 74]
    assert [note.start for note in moved.notes] == [0.0, 1.0]
    assert all(0 <= note.pitch <= 127 for note in transpose(melody([120, 125]), 24).notes)


def test_skeleton_survives_a_melody_the_chorale_voicer_cannot_reach():
    """Two octaves down used to return one chord for the whole tune."""
    tune = TRADITIONAL["shenandoah"]
    low = transpose(tune, -24)
    assert len(skeleton_from_rules(low).units) == len(skeleton_from_rules(tune).units)


def test_skeleton_harmony_is_octave_invariant():
    tune = TRADITIONAL["twinkle"]
    reference = [(unit.base.root, unit.base.quality) for unit in skeleton_from_rules(tune).units]
    for octaves in (-2, -1, 1, 2):
        moved = skeleton_from_rules(transpose(tune, 12 * octaves))
        assert [(unit.base.root, unit.base.quality) for unit in moved.units] == reference


# ---------------------------------------------------------------------------
# Translation from the contract
# ---------------------------------------------------------------------------


def test_chord_to_jazz_reads_the_inversion_as_a_bass_note():
    from contracts.schema import Chord

    chord = Chord(start=0.0, duration=4.0, roman="I6", root=C, quality="maj", inversion=1)
    jazz = chord_to_jazz(chord)
    assert jazz is not None and jazz.root == C and jazz.bass == E


def test_skeleton_quotes_the_rules_engine_roman_numerals():
    """`substitutionOf` is only honest if it is the base engine's own label."""
    skeleton = skeleton_from_rules(TRADITIONAL["twinkle"])
    assert all(unit.base_roman for unit in skeleton.units)
    assert any(unit.base_roman.startswith("I") for unit in skeleton.units)
