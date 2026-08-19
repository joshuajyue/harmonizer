"""Absolute-time alignment: the harmony must land where the melody is.

The bug this pins: `melody_to_grid` builds a compact grid starting at the first
sounding note, and the offset was never restored on the way out. A melody
beginning in bar 2 came back harmonized from beat 0 — every generated voice a
full bar early against the melody it accompanies. It affected all five engines,
because the fault was in the shared conversion layer rather than in any one of
them.

It also silently broke the contract invariant that the melody is retained as the
soprano. `contracts/test_fixtures.py` asserts that, and passed throughout, because
the canonical fixture happens to start at beat 0. An invariant that only holds for
the fixture is not an invariant, so these tests use offsets, pickups, fractional
starts and internal rests instead.
"""

import pytest

from contracts.schema import KeySignature, Melody, Note, TimeSignature
from ml.data.melody import melody_to_grid, voices_to_grid

import ml.engines.baselines  # noqa: F401
import ml.engines.neural  # noqa: F401
import ml.engines.rules  # noqa: F401
from ml.engines.base import all_engines

ENGINES = [e for e in all_engines() if e.is_available()]
ENGINE_IDS = [e.id for e in ENGINES]

TUNE = (72, 74, 76, 77, 76, 74, 72, 72)


def melody_at(offset: float, pitches=TUNE, duration: float = 1.0) -> Melody:
    return Melody(
        notes=[Note(pitch=p, start=offset + i * duration, duration=duration)
               for i, p in enumerate(pitches)],
        tempo=90.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=0, mode="major"),
    )


def fingerprint(notes):
    return [(n.pitch, n.start, n.duration) for n in notes]


class TestGridRoundTrip:
    """The conversion layer itself, where the fault was."""

    def test_grid_records_the_melody_origin(self):
        assert melody_to_grid(melody_at(0.0)).origin == 0.0
        assert melody_to_grid(melody_at(4.0)).origin == 4.0
        assert melody_to_grid(melody_at(2.5)).origin == 2.5

    def test_grid_length_covers_only_the_sounding_span(self):
        """Leading silence is carried in `origin`, not padded into the grid: a
        melody starting at bar 200 must not allocate 800 steps of rest."""
        near = melody_to_grid(melody_at(0.0))
        far = melody_to_grid(melody_at(200.0))
        assert far.length == near.length

    def test_voices_to_grid_inverts_with_the_same_origin(self):
        grid = melody_to_grid(melody_at(4.0))
        from ml.data.melody import grid_to_voices

        voices = grid_to_voices([grid.pitches], origin=grid.origin)
        recovered = voices_to_grid(voices, length=grid.length, origin=grid.origin)
        assert recovered[0] == grid.pitches

    def test_internal_rests_survive_the_round_trip(self):
        notes = [
            Note(pitch=72, start=4.0, duration=1.0),
            Note(pitch=76, start=7.0, duration=1.0),
        ]
        grid = melody_to_grid(Melody(notes=notes, tempo=90.0))
        assert grid.origin == 4.0
        assert -1 in grid.pitches, "the two-beat gap should be a rest, not closed up"
        assert grid.pitches[0] == 72 and grid.pitches[-1] == 76


@pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
class TestEngineAlignment:
    @pytest.mark.parametrize("offset", [0.0, 1.0, 4.0, 7.0, 16.0, 2.5, 0.25])
    def test_soprano_is_the_input_melody_exactly(self, engine, offset):
        """The contract invariant, checked where it actually bites.

        Includes fractional and off-grid offsets, because the soprano is
        returned verbatim rather than rebuilt from the sixteenth grid — a
        rebuilt one would quantize the user's own notes.
        """
        melody = melody_at(offset)
        result = engine.harmonize(melody, voice_count=4, seed=0)
        assert fingerprint(result.voices[0].notes) == fingerprint(melody.notes)

    @pytest.mark.parametrize("offset", [0.0, 4.0, 7.0, 16.0])
    def test_accompanying_voices_start_with_the_melody(self, engine, offset):
        result = engine.harmonize(melody_at(offset), voice_count=4, seed=0)
        for voice in result.voices[1:]:
            assert voice.notes, f"{voice.name} is empty"
            assert voice.notes[0].start == pytest.approx(offset), (
                f"{engine.id}: {voice.name} starts at {voice.notes[0].start}, melody at {offset}"
            )

    @pytest.mark.parametrize("offset", [0.0, 4.0, 7.0, 16.0])
    def test_chords_start_with_the_melody(self, engine, offset):
        result = engine.harmonize(melody_at(offset), voice_count=4, seed=0)
        assert result.chords
        assert result.chords[0].start == pytest.approx(offset)

    @pytest.mark.parametrize("offset", [0.0, 4.0, 16.0])
    def test_nothing_sounds_before_the_melody(self, engine, offset):
        """No invented introduction in the leading silence."""
        result = engine.harmonize(melody_at(offset), voice_count=4, seed=0)
        for voice in result.voices:
            for note in voice.notes:
                assert note.start >= offset - 1e-6, f"{voice.name} sounds at {note.start}"
        for chord in result.chords:
            assert chord.start >= offset - 1e-6

    @pytest.mark.parametrize("offset", [0.0, 4.0, 16.0])
    def test_harmony_does_not_outlast_the_melody(self, engine, offset):
        melody = melody_at(offset)
        end = max(n.start + n.duration for n in melody.notes)
        result = engine.harmonize(melody, voice_count=4, seed=0)
        for voice in result.voices:
            for note in voice.notes:
                assert note.start + note.duration <= end + 1e-6

    def test_offsetting_a_melody_only_translates_the_answer(self, engine):
        """The same tune in bar 1 and bar 3 must get the same harmony, moved.

        This is the property the bug destroyed: the offset was dropped, so the
        two answers came back identical in absolute time instead of translated.
        """
        base = engine.harmonize(melody_at(0.0), voice_count=4, seed=0)
        moved = engine.harmonize(melody_at(8.0), voice_count=4, seed=0)
        assert [c.roman for c in moved.chords] == [c.roman for c in base.chords]
        for original, shifted in zip(base.chords, moved.chords):
            assert shifted.start == pytest.approx(original.start + 8.0)
        for base_voice, moved_voice in zip(base.voices[1:], moved.voices[1:]):
            assert fingerprint(moved_voice.notes) == [
                (p, round(s + 8.0, 6), d) for p, s, d in fingerprint(base_voice.notes)
            ]

    def test_a_melody_with_an_internal_rest_keeps_its_timing(self, engine):
        notes = [
            Note(pitch=72, start=4.0, duration=1.0),
            Note(pitch=74, start=5.0, duration=1.0),
            Note(pitch=76, start=8.0, duration=1.0),
            Note(pitch=72, start=9.0, duration=1.0),
        ]
        melody = Melody(notes=notes, tempo=90.0, key=KeySignature(tonic=0, mode="major"))
        result = engine.harmonize(melody, voice_count=4, seed=0)
        assert fingerprint(result.voices[0].notes) == fingerprint(notes)
        for voice in result.voices[1:]:
            for note in voice.notes:
                assert 4.0 - 1e-6 <= note.start <= 10.0 + 1e-6
