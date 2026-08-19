"""Contract tests every engine must satisfy.

Applied to all registered engines automatically, so a new engine cannot be added
without meeting them. Determinism in particular is load-bearing: the eval
harness is worthless if two runs of the same engine on the same input disagree.
"""

import pytest

from contracts.schema import KeySignature, Melody, Note, TimeSignature
from ml.data.melody import voices_to_grid
from ml.engines.base import all_engines
from ml.theory.voicing import VOICE_RANGES

import ml.engines.baselines  # noqa: F401  (registers fixed_thirds)
import ml.engines.rules  # noqa: F401  (registers rules)
import ml.engines.neural  # noqa: F401  (registers neural, if a checkpoint exists)

ENGINES = [engine for engine in all_engines() if engine.is_available()]
ENGINE_IDS = [engine.id for engine in ENGINES]


#: A longer, harmonically determined tune. Transposition behaviour is not
#: meaningfully testable on a seven-note fragment where almost any harmony fits.
LONG_TUNE = (72, 72, 72, 71, 69, 69, 71, 72, 74, 76, 76, 74, 72, 74, 72, 71, 69, 67, 69, 71, 72, 72)


def long_melody(shift: int = 0) -> Melody:
    notes = [Note(pitch=p + shift, start=float(i), duration=1.0) for i, p in enumerate(LONG_TUNE)]
    return Melody(
        notes=notes,
        tempo=90.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=shift % 12, mode="major"),
    )


def simple_melody(pitches=(72, 74, 76, 77, 76, 74, 72), key=(0, "major")) -> Melody:
    notes = [Note(pitch=p, start=float(i), duration=1.0) for i, p in enumerate(pitches)]
    return Melody(
        notes=notes,
        tempo=90.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=key[0], mode=key[1]),
    )


@pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
class TestEngineContract:
    def test_returns_four_voices_in_order(self, engine):
        result = engine.harmonize(simple_melody())
        assert [voice.name for voice in result.voices] == ["soprano", "alto", "tenor", "bass"]

    def test_soprano_is_the_given_melody(self, engine):
        melody = simple_melody()
        result = engine.harmonize(melody)
        soprano = [(note.start, note.pitch) for note in result.voices[0].notes]
        expected = [(note.start, note.pitch) for note in melody.notes]
        assert soprano == expected

    def test_is_deterministic_at_temperature_zero(self, engine):
        melody = simple_melody()
        first = engine.harmonize(melody, temperature=0.0, seed=7)
        second = engine.harmonize(melody, temperature=0.0, seed=7)
        assert _fingerprint(first) == _fingerprint(second)

    def test_voices_stay_in_range(self, engine):
        result = engine.harmonize(simple_melody())
        for index, voice in enumerate(result.voices):
            low, high = VOICE_RANGES[index]
            for note in voice.notes:
                assert low - 2 <= note.pitch <= high + 2, f"{voice.name} out of range: {note.pitch}"

    def test_voices_do_not_cross_on_average(self, engine):
        result = engine.harmonize(simple_melody())
        lines = voices_to_grid(result.voices)
        crossings = 0
        steps = 0
        for t in range(len(lines[0])):
            column = [line[t] for line in lines if t < len(line) and line[t] != -1]
            if len(column) < 4:
                continue
            steps += 1
            crossings += any(column[i] < column[i + 1] for i in range(3))
        assert steps == 0 or crossings / steps < 0.25

    def test_reports_a_key(self, engine):
        result = engine.harmonize(simple_melody(key=(9, "minor")))
        assert 0 <= result.key.tonic <= 11
        assert result.key.mode in ("major", "minor")

    def test_detects_the_key_when_not_supplied(self, engine):
        melody = simple_melody()
        melody = melody.model_copy(update={"key": None})
        result = engine.harmonize(melody)
        assert result.key.tonic == 0 and result.key.mode == "major"

    def test_handles_an_empty_melody(self, engine):
        result = engine.harmonize(Melody(notes=[], tempo=90.0))
        assert all(voice.notes == [] for voice in result.voices)

    def test_handles_a_single_note(self, engine):
        result = engine.harmonize(simple_melody(pitches=(67,)))
        assert result.voices[0].notes

    def test_handles_a_rest_in_the_middle(self, engine):
        notes = [
            Note(pitch=72, start=0.0, duration=1.0),
            Note(pitch=76, start=3.0, duration=1.0),
            Note(pitch=72, start=4.0, duration=1.0),
        ]
        result = engine.harmonize(Melody(notes=notes, tempo=90.0, key=KeySignature(tonic=0, mode="major")))
        assert result.voices[0].notes

    def test_handles_an_eighth_note_grid(self, engine):
        notes = [Note(pitch=p, start=0.5 * i, duration=0.5) for i, p in enumerate([72, 74, 76, 77, 79, 77, 76, 74])]
        result = engine.harmonize(Melody(notes=notes, tempo=90.0, key=KeySignature(tonic=0, mode="major")))
        assert len(result.voices) == 4

    def test_handles_triple_metre(self, engine):
        notes = [Note(pitch=p, start=float(i), duration=1.0) for i, p in enumerate([67, 69, 71, 72, 71, 69])]
        melody = Melody(
            notes=notes,
            tempo=90.0,
            timeSignature=TimeSignature(numerator=3, denominator=4),
            key=KeySignature(tonic=0, mode="major"),
        )
        assert len(engine.harmonize(melody).voices) == 4

    def test_respects_voice_count(self, engine):
        for count in (2, 3, 4):
            result = engine.harmonize(simple_melody(), voice_count=count)
            assert len(result.voices) == count

    def test_violations_reference_real_voices(self, engine):
        result = engine.harmonize(simple_melody())
        for violation in result.violations:
            assert violation.severity in ("info", "warning", "error")
            for name in violation.voices:
                assert name in ("soprano", "alto", "tenor", "bass")

    def test_chords_are_ordered_and_non_overlapping(self, engine):
        result = engine.harmonize(simple_melody())
        for previous, chord in zip(result.chords, result.chords[1:]):
            assert chord.start >= previous.start + previous.duration - 1e-6

    def test_transposing_the_melody_does_not_scramble_the_harmony(self, engine):
        """A sanity floor, not an invariance claim.

        Exact equivariance is impossible for any range-aware engine: SATB ranges
        are absolute, so a tune near the top of the soprano range genuinely needs
        different octave placement from the same tune a fourth lower, and that
        feeds back into which chords are reachable. But a transposed tune must
        still get *broadly* the same harmony. This floor is what catches real
        bugs — the octave error in tonic normalization that this found scored
        0.06 here, against 1.00 once fixed.
        """
        reference = [chord.root % 12 for chord in engine.harmonize(long_melody(0)).chords]
        if not reference:
            pytest.skip("engine does not report chords")
        for shift in (2, 4, 5, 7, -3, -5):
            moved = engine.harmonize(long_melody(shift))
            roots = [(chord.root - shift) % 12 for chord in moved.chords]
            span = min(len(reference), len(roots))
            matches = sum(a == b for a, b in zip(reference[:span], roots[:span]))
            assert matches / max(1, span) >= 0.4, f"shift {shift}: {reference} vs {roots}"


LEARNED_ENGINES = [e for e in ENGINES if e.id in ("neural", "neural_vl")]


@pytest.mark.skipif(not LEARNED_ENGINES, reason="no trained checkpoint")
@pytest.mark.parametrize("engine", LEARNED_ENGINES, ids=[e.id for e in LEARNED_ENGINES])
def test_learned_engine_is_exactly_transposition_equivariant(engine):
    """The payoff of the tonic-relative representation, asserted exactly.

    The model never sees the key: every piece is transposed so the tonic is C
    before it is tokenised, so the same tune in twelve keys is literally the same
    input and must produce literally the same output. v1 fed absolute pitch
    classes and had to learn all twelve transpositions separately from ~400
    chorales; this is the handicap that removes.

    Requires the inference-time register correction as well as the normalization
    — without it a tune transposed up a fifth normalizes an octave high and this
    drops from 1.00 to 0.06.
    """
    reference = [chord.roman for chord in engine.harmonize(long_melody(0)).chords]
    assert reference
    for shift in (1, 2, 3, 4, 5, 6, 7, -2, -3, -5):
        romans = [chord.roman for chord in engine.harmonize(long_melody(shift)).chords]
        assert romans == reference, f"shift {shift}"


def _fingerprint(harmonization):
    return [
        [(note.pitch, note.start, note.duration) for note in voice.notes]
        for voice in harmonization.voices
    ]


def test_registry_ids_are_unique():
    ids = [engine.id for engine in all_engines()]
    assert len(ids) == len(set(ids))


def test_engines_expose_metadata():
    for engine in all_engines():
        assert engine.id and engine.name and engine.description
        assert isinstance(engine.learned, bool)
