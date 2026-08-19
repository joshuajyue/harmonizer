"""The public surface other workstreams depend on.

`ml/reharm/` is owned by another agent, reads this package, and calls the `rules`
engine through the registry. These tests pin the surface it relies on so that a
refactor here fails loudly in my own suite rather than silently in someone
else's, and so that "avoid breaking `HarmonyEngine`'s public interface" is a
check rather than a promise.

Deliberately narrow: it fixes the *contract* — names, signatures, registry
behaviour, the shape of what comes back — and asserts nothing about musical
quality, which is measured in `ml/eval` and free to change.
"""

import inspect

import pytest

from contracts.schema import KeySignature, Melody, Note, TimeSignature
from ml.engines.base import (
    Harmonization,
    HarmonyEngine,
    all_engines,
    get_engine,
    register,
)

import ml.engines.baselines  # noqa: F401
import ml.engines.neural  # noqa: F401
import ml.engines.rules  # noqa: F401


def melody() -> Melody:
    return Melody(
        notes=[Note(pitch=p, start=float(i), duration=1.0) for i, p in enumerate([72, 74, 76, 77])],
        tempo=90.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=0, mode="major"),
    )


class TestRegistry:
    def test_rules_is_reachable_by_id(self):
        """The reharm workstream builds on this engine specifically."""
        engine = get_engine("rules")
        assert isinstance(engine, HarmonyEngine)
        assert engine.id == "rules"

    def test_unknown_id_raises_keyerror_listing_options(self):
        with pytest.raises(KeyError) as info:
            get_engine("does-not-exist")
        assert "rules" in str(info.value)

    def test_all_engines_returns_instances(self):
        engines = all_engines()
        assert engines and all(isinstance(e, HarmonyEngine) for e in engines)

    def test_registering_a_duplicate_id_is_refused(self):
        """Two workstreams registering the same id must fail, not shadow."""

        class Duplicate(HarmonyEngine):
            id = "rules"

            def harmonize(self, melody, *, voice_count=4, temperature=0.0, seed=None):
                raise NotImplementedError

        with pytest.raises(ValueError):
            register(Duplicate())


class TestEngineSignature:
    def test_harmonize_keeps_its_keyword_only_options(self):
        signature = inspect.signature(HarmonyEngine.harmonize)
        assert list(signature.parameters) == ["self", "melody", "voice_count", "temperature", "seed"]
        for name in ("voice_count", "temperature", "seed"):
            assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    def test_optional_hooks_exist_with_safe_defaults(self):
        assert hasattr(HarmonyEngine, "is_available")
        assert hasattr(HarmonyEngine, "log_likelihood")

        class Minimal(HarmonyEngine):
            id = "minimal-probe"

            def harmonize(self, melody, *, voice_count=4, temperature=0.0, seed=None):
                return Harmonization(key=KeySignature(tonic=0, mode="major"), voices=[])

        probe = Minimal()
        assert probe.is_available() is True
        assert probe.log_likelihood(melody(), []) is None

    def test_engine_metadata_fields_exist(self):
        for engine in all_engines():
            assert isinstance(engine.id, str) and engine.id
            assert isinstance(engine.name, str)
            assert isinstance(engine.description, str)
            assert isinstance(engine.learned, bool)


class TestHarmonizationShape:
    def test_fields_and_types(self):
        result = get_engine("rules").harmonize(melody(), voice_count=4, temperature=0.0, seed=0)
        assert isinstance(result, Harmonization)
        assert {"key", "voices", "chords", "violations"} <= set(vars(result))
        assert result.key.mode in ("major", "minor")
        assert [v.name for v in result.voices] == ["soprano", "alto", "tenor", "bass"]

    def test_chords_carry_the_fields_a_reharmonizer_needs(self):
        """A reharmonizer substitutes chords, so it needs the harmonic reading,
        not just the notes."""
        result = get_engine("rules").harmonize(melody(), seed=0)
        assert result.chords
        for chord in result.chords:
            assert 0 <= chord.root <= 11
            assert chord.quality
            assert chord.inversion >= 0
            assert chord.roman
            assert chord.duration > 0
            assert chord.extensions == []
            assert chord.substitutionOf is None and chord.substitutionKind is None

    def test_chords_tile_the_melody_without_gaps_or_overlaps(self):
        """Substitution assumes a chord sequence it can index by time."""
        result = get_engine("rules").harmonize(melody(), seed=0)
        for previous, chord in zip(result.chords, result.chords[1:]):
            assert chord.start == pytest.approx(previous.start + previous.duration)

    def test_the_rules_engine_is_deterministic_for_a_caller(self):
        first = get_engine("rules").harmonize(melody(), seed=0)
        second = get_engine("rules").harmonize(melody(), seed=0)
        assert [(c.roman, c.start) for c in first.chords] == [(c.roman, c.start) for c in second.chords]

    def test_temperature_is_honoured_rather_than_ignored(self):
        """A silently ignored knob is worse than an absent one: the caller
        believes it has control it does not have."""
        engine = get_engine("rules")
        tune = Melody(
            notes=[Note(pitch=p, start=float(i), duration=1.0)
                   for i, p in enumerate([72, 72, 71, 69, 71, 72, 74, 76, 74, 72, 71, 69, 67, 69, 71, 72])],
            tempo=90.0,
            timeSignature=TimeSignature(numerator=4, denominator=4),
            key=KeySignature(tonic=0, mode="major"),
        )
        cold = [c.roman for c in engine.harmonize(tune, temperature=0.0, seed=0).chords]
        hot = [c.roman for c in engine.harmonize(tune, temperature=1.5, seed=0).chords]
        assert cold != hot
