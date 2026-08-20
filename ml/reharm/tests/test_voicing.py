"""Jazz voicing, and the chorale rules that deliberately do not apply.

The point of this module is that most of `ml/theory/voicing.py` is wrong here:
parallel fifths are an idiom, spacing is irregular, voices cross, and the root
is routinely absent because the bass is a different instrument. So these tests
assert the *jazz* behaviour, including the things a chorale test would fail —
a rootless voicing, a voicing that crosses above a passing low melody note.

The one chorale rule that survives is acoustics rather than style: nothing may
sound a semitone under the melody, because that is a minor ninth against the
tune and the ear reads it as a mistake.
"""

import pytest

from ml.reharm.chords import JazzChord
from ml.reharm.data import ChordSpan
from ml.reharm.voicing import (
    BASS_HIGH,
    BASS_LOW,
    DEFAULT_STYLE,
    INNER_HIGH,
    INNER_LOW,
    VoicingStyle,
    _clash_with_melody,
    _last_resort,
    _omitted_intervals,
    arrange,
    bass_line,
    build_voices,
    pitch_class_sets,
    voice_chords,
)

C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)

CMAJ7 = JazzChord(root=C, quality="maj7")
G7 = JazzChord(root=G, quality="dom7")
DM7 = JazzChord(root=D, quality="min7")


def spans(*chords, duration: float = 4.0) -> list[ChordSpan]:
    return [ChordSpan(index * duration, duration, chord) for index, chord in enumerate(chords)]


# ---------------------------------------------------------------------------
# Which notes to play
# ---------------------------------------------------------------------------


def test_a_four_note_voicing_is_rootless():
    """3-5-7-9 and 7-9-3-5: the bass has the root, so the left hand does not."""
    options = pitch_class_sets(CMAJ7, DEFAULT_STYLE, 4)
    assert options
    assert 0 not in options[0], "the first choice should be rootless"
    assert {4, 11} <= set(options[0]), "third and seventh always present"


def test_a_two_note_voicing_is_the_guide_tones():
    options = pitch_class_sets(G7, DEFAULT_STYLE, 2)
    assert set(options[0]) == {4, 10}


def test_altered_tensions_reach_the_voicing():
    altered = JazzChord(root=G, quality="dom7", extensions=("b9", "b13"))
    assert any({1, 8} <= set(option) for option in pitch_class_sets(altered, DEFAULT_STYLE, 4))


def test_quartal_voicings_are_offered_on_the_chords_that_take_them():
    quartal_only = VoicingStyle(rootless=False, quartal=True, drop2=False, upper_structure=False)
    stacked = [
        option for option in pitch_class_sets(DM7, quartal_only, 4)
        if all((option[i + 1] - option[i]) % 12 == 5 for i in range(len(option) - 1))
    ]
    assert stacked, "So What voicings should be reachable on a min7"


def test_quartal_stacks_stay_inside_the_chord():
    """D-G-C-F is D dorian. F-Bb-Eb-Ab is four fourths and two wrong notes."""
    quartal_only = VoicingStyle(rootless=False, quartal=True, drop2=False, upper_structure=False)
    allowed = set(DM7.all_pcs) | {(DM7.root + interval) % 12 for interval in (2, 5, 9)}
    for count in (2, 3, 4):
        for option in pitch_class_sets(DM7, quartal_only, count):
            for interval in option:
                assert (DM7.root + interval) % 12 in allowed


@pytest.mark.parametrize(
    "chord",
    [CMAJ7, G7, DM7, JazzChord(root=A, quality="min6"), JazzChord(root=B, quality="halfdim7"),
     JazzChord(root=C, quality="dim7"), JazzChord(root=G, quality="sus4", extensions=("7",)),
     JazzChord(root=G, quality="dom7", extensions=("b9", "b13"))],
    ids=lambda chord: chord.symbol(),
)
def test_no_option_ever_contains_a_note_outside_the_chord(chord):
    """The property that would have caught the quartal bug immediately."""
    allowed = set(chord.all_pcs) | {(chord.root + interval) % 12 for interval in (2, 5, 9)}
    for count in (1, 2, 3, 4):
        for option in pitch_class_sets(chord, DEFAULT_STYLE, count):
            for interval in option:
                assert (chord.root + interval) % 12 in allowed, f"{chord.symbol()} -> {option}"


def test_colour_tones_are_added_without_being_asked():
    """A chord symbol is a floor: a pianist supplies the 9 unprompted."""
    assert any(2 in option for option in pitch_class_sets(DM7, DEFAULT_STYLE, 4))


def test_every_option_has_the_requested_number_of_notes():
    for count in (1, 2, 3, 4):
        for option in pitch_class_sets(CMAJ7, DEFAULT_STYLE, count):
            assert len(option) == count


# ---------------------------------------------------------------------------
# Placing them
# ---------------------------------------------------------------------------


def test_arrange_is_ascending_and_under_the_ceiling():
    for voicing in arrange([4, 10], G, ceiling=76):
        assert voicing == sorted(voicing)
        assert all(INNER_LOW <= pitch < 76 for pitch in voicing)
        assert len(set(pitch % 12 for pitch in voicing)) == 2


def test_arrange_returns_nothing_when_the_register_cannot_hold_it():
    assert arrange([0, 4, 7, 11], C, ceiling=INNER_LOW + 2) == []


def test_a_minor_ninth_under_the_melody_is_refused():
    assert _clash_with_melody([64], [77])       # E under F an octave up
    assert not _clash_with_melody([64], [76])   # E under E is an octave, fine


def test_a_unison_or_semitone_against_the_melody_is_refused():
    assert _clash_with_melody([72], [72])
    assert _clash_with_melody([71], [72])
    assert not _clash_with_melody([70], [72])


def test_crossing_above_a_passing_low_melody_note_is_allowed():
    """Deliberate, and a chorale test would fail it.

    Voice crossing is normal in this idiom; forbidding it drags the whole
    accompaniment into the bass every time the tune dips for one note.
    """
    assert not _clash_with_melody([67, 71], [55])


def test_melody_on_the_fourth_of_a_dominant_drops_the_third():
    assert _omitted_intervals(G7, [72]) == {4}


def test_melody_on_the_flat_thirteenth_drops_the_fifth():
    assert _omitted_intervals(G7, [63]) == {7}


def test_no_omission_when_the_melody_is_a_chord_tone():
    assert _omitted_intervals(G7, [71]) == set()


def test_last_resort_plays_nothing_rather_than_something_wrong():
    """The register is impossible here; silence beats a beating note."""
    assert _last_resort(CMAJ7, ceiling=INNER_LOW, melody_pitches=[INNER_LOW]) == []


def test_last_resort_still_avoids_the_melody():
    for pitch in _last_resort(G7, ceiling=72, melody_pitches=[72]):
        assert not _clash_with_melody([pitch], [72])


# ---------------------------------------------------------------------------
# Voice leading between chords
# ---------------------------------------------------------------------------


def test_voice_leading_moves_as_little_as_it_can():
    """ii-V-I is the test case: the guide tones move by a semitone each."""
    melody = [(0.0, 74, 4.0), (4.0, 71, 4.0), (8.0, 72, 4.0)]
    voicings = voice_chords(spans(DM7, G7, CMAJ7), melody, inner_voices=2)
    assert len(voicings) == 3
    for previous, current in zip(voicings, voicings[1:]):
        movement = sum(abs(a - b) for a, b in zip(previous, current))
        assert movement <= 4, f"{previous} -> {current} is not voice leading"


def test_the_voicing_never_reaches_the_melody():
    melody = [(0.0, 72, 4.0), (4.0, 67, 4.0)]
    for voicing, span in zip(voice_chords(spans(CMAJ7, G7), melody, inner_voices=3), spans(CMAJ7, G7)):
        sounding = [pitch for start, pitch, _ in melody if span.start <= start < span.stop]
        assert not _clash_with_melody(voicing, sounding)


def test_guide_tones_survive_a_low_melody():
    """The register shrinks, the voicing thins, the third and seventh stay."""
    melody = [(0.0, 55, 4.0)]
    voicing = voice_chords(spans(G7), melody, inner_voices=3)[0]
    assert voicing, "some accompaniment should still be playable"
    assert set(pitch % 12 for pitch in voicing) <= set(G7.all_pcs)


# ---------------------------------------------------------------------------
# Bass
# ---------------------------------------------------------------------------


def test_the_bass_plays_roots_in_the_bass_register():
    notes = bass_line(spans(CMAJ7, G7), walking=False)
    assert [note.pitch % 12 for note in notes] == [C, G]
    assert all(BASS_LOW <= note.pitch <= BASS_HIGH for note in notes)


def test_walking_adds_an_approach_note_into_the_next_chord():
    plain = bass_line(spans(CMAJ7, G7), walking=False)
    walking = bass_line(spans(CMAJ7, G7), walking=True)
    assert len(walking) > len(plain)
    approach = walking[1]
    target = next(note for note in walking if note.pitch % 12 == G)
    assert abs(approach.pitch - target.pitch) <= 7


def test_a_slash_bass_is_honoured():
    slash = JazzChord(root=C, quality="maj7", bass=E)
    assert bass_line(spans(slash), walking=False)[0].pitch % 12 == E


def test_the_bass_does_not_climb_into_the_voicing():
    long_line = spans(*[JazzChord(root=root, quality="dom7") for root in (C, F, Bb, Eb, Ab, Db)])
    notes = bass_line(long_line, walking=False)
    assert max(note.pitch for note in notes) <= BASS_HIGH


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_voice_count_shapes_the_texture():
    melody = [(0.0, 72, 4.0), (4.0, 71, 4.0)]
    for count in (2, 3, 4, 5, 6):
        voices = build_voices(spans(CMAJ7, G7), melody, voice_count=count)
        assert len(voices) == count
        assert voices[0].name == "soprano" and voices[-1].name == "bass"


def test_two_voices_are_melody_and_bass_only():
    voices = build_voices(spans(CMAJ7), [(0.0, 72, 4.0)], voice_count=2)
    assert [voice.name for voice in voices] == ["soprano", "bass"]


def test_the_melody_is_reproduced_exactly():
    melody = [(0.0, 72, 2.5), (2.5, 71, 1.5)]
    soprano = build_voices(spans(CMAJ7), melody, voice_count=4)[0]
    assert [(note.pitch, note.start, note.duration) for note in soprano.notes] == [
        (72, 0.0, 2.5), (71, 2.5, 1.5)
    ]


def test_a_held_inner_note_is_tied_rather_than_repeated():
    """C and Am7 share E and G; the parts should sustain, not restrike."""
    melody = [(0.0, 72, 8.0)]
    voices = build_voices(
        spans(JazzChord(root=C, quality="maj6"), JazzChord(root=A, quality="min7")),
        melody,
        voice_count=4,
    )
    inner = [note for voice in voices[1:-1] for note in voice.notes]
    assert any(note.duration > 4.0 for note in inner), "a common tone should be tied"


def test_an_offset_origin_shifts_everything_together():
    melody = [(0.0, 72, 4.0)]
    voices = build_voices(spans(CMAJ7), melody, voice_count=4, origin=16.0)
    assert all(note.start >= 16.0 for voice in voices for note in voice.notes)


def test_the_voicing_says_what_the_chord_is():
    """Two inner voices, seven tunes: how often is the quality actually stated?

    A regression guard for the whole point of the texture. It was much worse
    before rotations and colour-substitutes: the third could not sit below the
    seventh, so a blocked guide tone fell back to root-and-third, which states
    nothing. 89% of chords now sound at least one guide tone and none is voiced
    with the root alone.
    """
    from ml.reharm.engine import JAZZ_REHARM
    from ml.reharm.melodies import TRADITIONAL

    total = with_guide = root_only = 0
    for melody in TRADITIONAL.values():
        result = JAZZ_REHARM.harmonize(melody, voice_count=4, temperature=0.0)
        for chord in result.chords:
            plain = JazzChord(root=chord.root, quality=chord.quality)
            voiced = {
                note.pitch % 12
                for voice in result.voices[1:-1]
                for note in voice.notes
                if note.start <= chord.start < note.start + note.duration
            }
            if not voiced:
                continue
            total += 1
            with_guide += bool(set(plain.guide_tone_pcs) & voiced)
            root_only += voiced == {plain.root}
    assert total > 50
    assert with_guide / total > 0.8
    assert root_only == 0
