"""The engines, end to end, against the contract they have to satisfy.

The API-facing promises are the ones worth pinning: the melody survives as the
soprano, the output validates as a `HarmonizeResponse`, the same seed gives the
same music, and every substitution says what it replaced and how. That last one
is a product requirement rather than a technical one — a reharmonization the
user cannot interrogate is indistinguishable from a random chord generator.
"""

import pytest

from contracts.schema import (
    HarmonizeResponse,
    KeySignature,
    Melody,
    Note,
    TimeSignature,
)
from ml.engines.base import all_engines, get_engine
from ml.reharm.chords import EXTENSION_SEMITONES, SUBSTITUTION_KINDS
from ml.reharm.engine import JAZZ_REHARM, JAZZ_REHARM_RULES
from ml.reharm.melodies import TRADITIONAL

ENGINES = (JAZZ_REHARM, JAZZ_REHARM_RULES)


def response(harmonization, engine_id: str) -> HarmonizeResponse:
    return HarmonizeResponse(
        key=harmonization.key,
        chords=harmonization.chords,
        voices=harmonization.voices,
        violations=harmonization.violations,
        engine=engine_id,
        latencyMs=0.0,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_engines_are_registered_under_stable_ids():
    assert get_engine("jazz_reharm") is JAZZ_REHARM
    assert get_engine("jazz_reharm_rules") is JAZZ_REHARM_RULES
    assert {"jazz_reharm", "jazz_reharm_rules"} <= {engine.id for engine in all_engines()}


def test_learned_flag_distinguishes_the_two():
    assert JAZZ_REHARM.learned is True
    assert JAZZ_REHARM_RULES.learned is False


def test_availability():
    assert JAZZ_REHARM_RULES.is_available()
    assert JAZZ_REHARM.is_available(), "the shipped chord model should be present"


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.id)
def test_output_validates_against_the_contract(engine):
    result = engine.harmonize(TRADITIONAL["twinkle"], voice_count=4, temperature=1.0, seed=3)
    payload = response(result, engine.id)
    assert payload.chords and payload.voices
    for chord in payload.chords:
        assert 0 <= chord.root <= 11
        assert chord.duration > 0
        assert all(extension in EXTENSION_SEMITONES for extension in chord.extensions)
        if chord.substitutionKind is not None:
            assert chord.substitutionKind in SUBSTITUTION_KINDS
            assert chord.substitutionOf, "a substitution must say what it replaced"


@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.id)
def test_melody_survives_as_the_soprano(engine):
    melody = TRADITIONAL["shenandoah"]
    result = engine.harmonize(melody, voice_count=4, temperature=1.0, seed=1)
    soprano = next(voice for voice in result.voices if voice.name == "soprano")
    assert [(note.pitch, note.start) for note in soprano.notes] == [
        (note.pitch, note.start) for note in melody.notes
    ]


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6])
def test_voice_count_is_respected(count):
    result = JAZZ_REHARM.harmonize(TRADITIONAL["twinkle"], voice_count=count, temperature=1.0, seed=2)
    assert len(result.voices) == count
    assert result.voices[0].name == "soprano"
    assert result.voices[-1].name == "bass"


def test_empty_melody_does_not_explode():
    empty = Melody(notes=[], tempo=120.0, timeSignature=TimeSignature(numerator=4, denominator=4))
    for engine in ENGINES:
        result = engine.harmonize(empty)
        assert result.voices == [] or all(voice.notes == [] for voice in result.voices)


def _offset_melody(offset: float) -> Melody:
    return Melody(
        notes=[
            Note(pitch=pitch, start=offset + index, duration=1.0)
            for index, pitch in enumerate((65, 69, 72, 74, 72, 69, 65, 65))
        ],
        tempo=120.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=5, mode="major"),
    )


@pytest.mark.parametrize("offset", [0.0, 4.0, 7.0, 16.0])
@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.id)
def test_offset_melody_keeps_its_offset_in_every_voice(engine, offset):
    """A melody that starts at bar 5 must come back at bar 5 — in ALL voices.

    The previous version of this test checked the soprano and the bass start
    time only. The soprano is passed through from the input melody rather than
    converted out of the grid, so it is structurally immune to an origin fault
    and looked right while the inner voices were doubled. Checking every voice
    at several offsets is what makes the test able to fail.
    """
    result = engine.harmonize(_offset_melody(offset), voice_count=4, temperature=0.0)
    assert result.chords[0].start == pytest.approx(offset)
    for voice in result.voices:
        assert voice.notes, f"{voice.name} is empty"
        assert voice.notes[0].start == pytest.approx(offset), f"{voice.name} starts in the wrong bar"


@pytest.mark.parametrize("offset", [0.0, 4.0, 7.0, 16.0])
def test_offset_does_not_change_the_harmony(offset):
    """Moving a tune later in the bar line cannot change what it is.

    The real damage from the origin fault was not the output timing: the chord
    spans were absolute while the melody had been rebased to zero, so the units
    were handed the WRONG melody notes and the melody-compatibility constraint
    — the thing this engine exists to enforce — was checked against them. The
    timing was the symptom that happened to be visible.
    """
    reference = [
        (chord.root, chord.quality)
        for chord in JAZZ_REHARM.harmonize(_offset_melody(0.0), temperature=0.0).chords
    ]
    moved = JAZZ_REHARM.harmonize(_offset_melody(offset), temperature=0.0)
    assert [(chord.root, chord.quality) for chord in moved.chords] == reference


@pytest.mark.parametrize("offset", [0.0, 4.0, 7.0, 16.0])
def test_every_melody_note_reaches_the_harmonic_units(offset):
    """No note may be lost or duplicated on its way into the constraint."""
    from ml.reharm.skeleton import skeleton_from_rules

    melody = _offset_melody(offset)
    skeleton = skeleton_from_rules(melody)
    seen = [pitch for unit in skeleton.units for _, pitch, _ in unit.melody]
    assert seen == [note.pitch for note in melody.notes]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.id)
def test_zero_temperature_is_deterministic(engine):
    first = engine.harmonize(TRADITIONAL["twinkle"], temperature=0.0, seed=None)
    second = engine.harmonize(TRADITIONAL["twinkle"], temperature=0.0, seed=99)
    assert [chord.roman for chord in first.chords] == [chord.roman for chord in second.chords]


def test_same_seed_reproduces_the_same_music():
    first = JAZZ_REHARM.harmonize(TRADITIONAL["greensleeves"], temperature=1.2, seed=42)
    second = JAZZ_REHARM.harmonize(TRADITIONAL["greensleeves"], temperature=1.2, seed=42)
    assert [chord.roman for chord in first.chords] == [chord.roman for chord in second.chords]
    assert [
        (note.pitch, note.start, note.duration) for voice in first.voices for note in voice.notes
    ] == [(note.pitch, note.start, note.duration) for voice in second.voices for note in voice.notes]


def test_different_seeds_give_different_reharmonizations():
    outputs = {
        tuple(chord.roman for chord in JAZZ_REHARM.harmonize(
            TRADITIONAL["twinkle"], temperature=1.3, seed=seed
        ).chords)
        for seed in range(6)
    }
    assert len(outputs) > 1


# ---------------------------------------------------------------------------
# Musical properties of the output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["twinkle", "greensleeves", "amazing_grace", "blues_riff"])
def test_no_voiced_note_sounds_a_semitone_under_the_melody(name):
    """The one chorale rule that survives, because it is acoustics not style."""
    result = JAZZ_REHARM.harmonize(TRADITIONAL[name], voice_count=4, temperature=1.1, seed=8)
    soprano = next(voice for voice in result.voices if voice.name == "soprano")
    others = [voice for voice in result.voices if voice.name != "soprano"]
    windows = [(note.start, note.start + note.duration, note.pitch) for note in soprano.notes]
    for voice in others:
        if voice.name == "bass":
            continue
        for note in voice.notes:
            for start, stop, pitch in windows:
                if not (start - 1e-6 <= note.start < stop - 1e-6):
                    continue
                gap = pitch - note.pitch
                assert abs(gap) >= 2, f"{voice.name} collides with the melody at {note.start}"
                assert not (0 < gap <= 13 and gap % 12 == 1), (
                    f"{voice.name} sounds a minor ninth under the melody at {note.start}"
                )


@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.id)
def test_reharmonization_states_sevenths(engine):
    """The rules engine emits triads; jazz states sevenths. That is the floor."""
    result = engine.harmonize(TRADITIONAL["twinkle"], temperature=1.0, seed=5)
    sevenths = sum(1 for chord in result.chords if chord.quality not in ("maj", "min", "dim", "aug"))
    assert sevenths / len(result.chords) > 0.9


def test_substitutions_are_explained():
    result = JAZZ_REHARM.harmonize(TRADITIONAL["twinkle"], temperature=1.3, seed=4)
    substituted = [chord for chord in result.chords if chord.substitutionKind not in (None, "extension")]
    assert substituted, "a reharmonization with no substitutions is not one"
    for chord in substituted:
        assert chord.substitutionOf
        assert chord.roman != chord.substitutionOf


def test_violations_are_jazz_violations_not_chorale_ones():
    for name in TRADITIONAL:
        result = JAZZ_REHARM.harmonize(TRADITIONAL[name], temperature=1.2, seed=6)
        for violation in result.violations:
            assert violation.kind in ("melody_conflict", "unresolved_substitution")


# ---------------------------------------------------------------------------
# Register robustness
# ---------------------------------------------------------------------------


def _octave(melody: Melody, octaves: int) -> Melody:
    return melody.model_copy(update={
        "notes": [note.model_copy(update={"pitch": note.pitch + 12 * octaves}) for note in melody.notes]
    })


@pytest.mark.parametrize("engine", ENGINES, ids=lambda engine: engine.id)
def test_octave_shifts_do_not_change_the_harmony(engine):
    """Harmony is pitch classes, so an octave cannot touch it.

    It could, before: the rules engine that supplies the skeleton voices actual
    SATB parts and its ranges are absolute, so a melody two octaves low made its
    voicing search fail and it returned ONE chord for thirteen bars. The
    skeleton is now octave-normalized before that engine sees it, which is
    sound precisely because chords are pitch classes. Transcription hands us
    normalized melodies, but MIDI upload does not.
    """
    melody = TRADITIONAL["shenandoah"]
    reference = [
        (chord.root, chord.quality) for chord in engine.harmonize(melody, temperature=0.0).chords
    ]
    assert reference
    for octaves in (-2, -1, 1, 2):
        moved = engine.harmonize(_octave(melody, octaves), temperature=0.0)
        assert [(chord.root, chord.quality) for chord in moved.chords] == reference


@pytest.mark.parametrize("octaves", [-2, -1, 0, 1])
def test_voicing_never_beats_against_a_low_melody(octaves):
    """The last-resort voicing used to skip the melody check entirely."""
    melody = _octave(TRADITIONAL["shenandoah"], octaves)
    result = JAZZ_REHARM.harmonize(melody, voice_count=4, temperature=1.0, seed=3)
    soprano = next(voice for voice in result.voices if voice.name == "soprano")
    windows = [(note.start, note.start + note.duration, note.pitch) for note in soprano.notes]
    for voice in result.voices[1:-1]:
        for note in voice.notes:
            for start, stop, pitch in windows:
                if not (start - 1e-6 <= note.start < stop - 1e-6):
                    continue
                gap = pitch - note.pitch
                assert abs(gap) >= 2
                assert not (0 < gap <= 13 and gap % 12 == 1)


def test_voicing_follows_the_melody_register():
    """Where the left hand goes is relative to where the tune is."""
    def centre(octaves: int) -> float:
        result = JAZZ_REHARM.harmonize(
            _octave(TRADITIONAL["shenandoah"], octaves), voice_count=4, temperature=0.0
        )
        pitches = [note.pitch for voice in result.voices[1:-1] for note in voice.notes]
        return sum(pitches) / len(pitches)

    assert centre(1) > centre(0) > centre(-1)
