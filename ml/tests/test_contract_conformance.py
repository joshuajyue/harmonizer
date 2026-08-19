"""Conformance tests against the shared API contract.

`contracts/` is owned by the lead and is read-only here, so this suite is the
other half of the drift guard: `contracts/test_contract_sync.py` proves the
TypeScript and Pydantic definitions mirror each other, and this proves the
engines actually satisfy the Pydantic side.

The specific failures it exists to catch:

* A chord quality that the contract does not document. `quality` is a bare `str`
  in the schema, so Pydantic will happily accept "min11" and the UI will render
  nothing recognisable. The vocabulary is fixed in `ml/theory/chords.py` and is
  checked here against the list the contract documents.
* A response-side field left to its default. The contract's rule is that the
  server always populates response fields so the frontend never null-checks;
  a default that happens to produce the right value is not the same as
  populating it, and stops being right the moment the default changes.
"""

import re

import pytest

from contracts.schema import Chord, HarmonizeResponse, KeySignature, Melody, Note, TimeSignature, Voice
from ml.data.melody import chorale_to_melody
from ml.theory.chords import QUALITY_TEMPLATES
from ml.tests._engines import chorale_engines, engine_ids

ENGINES = chorale_engines()
ENGINE_IDS = engine_ids(ENGINES)


def documented_qualities() -> set[str]:
    """Quality strings the contract's `Chord.quality` description enumerates."""
    description = Chord.model_fields["quality"].description or ""
    return set(re.findall(r'"([a-z0-9#]+)"', description))


def melody(pitches=(72, 74, 76, 77, 76, 74, 72, 72)) -> Melody:
    return Melody(
        notes=[Note(pitch=p, start=float(i), duration=1.0) for i, p in enumerate(pitches)],
        tempo=100.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=0, mode="major"),
    )


class TestQualityVocabulary:
    def test_every_quality_the_theory_layer_can_emit_is_documented(self):
        """`quality` is a free-form str in the schema, so nothing but this test
        stops the engines emitting a value the UI cannot render."""
        documented = documented_qualities()
        assert documented, "contract no longer documents its quality strings"
        undocumented = set(QUALITY_TEMPLATES) - documented
        assert not undocumented, f"qualities not in the contract: {sorted(undocumented)}"

    @pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
    def test_engines_emit_only_documented_qualities(self, engine):
        documented = documented_qualities()
        for chord in engine.harmonize(melody()).chords:
            assert chord.quality in documented, f"{engine.id} emitted {chord.quality!r}"


class TestRequiredRequestFields:
    """The contract made these required because a silent default is a
    silently-wrong answer. Assert the constraint really bites."""

    def test_melody_rejects_a_missing_tempo(self):
        with pytest.raises(Exception):
            Melody(notes=[Note(pitch=60, start=0.0, duration=1.0)])

    def test_melody_rejects_a_non_positive_tempo(self):
        with pytest.raises(Exception):
            Melody(notes=[Note(pitch=60, start=0.0, duration=1.0)], tempo=0.0)

    def test_time_signature_rejects_partial_specification(self):
        with pytest.raises(Exception):
            TimeSignature(numerator=3)
        with pytest.raises(Exception):
            TimeSignature()

    def test_time_signature_defaults_to_four_four_when_omitted_entirely(self):
        built = Melody(notes=[Note(pitch=60, start=0.0, duration=1.0)], tempo=90.0)
        assert (built.timeSignature.numerator, built.timeSignature.denominator) == (4, 4)

    def test_corpus_melodies_carry_an_explicit_time_signature(self):
        from ml.data.corpus import load_chorales

        chorale = load_chorales(limit=1)[0]
        built = chorale_to_melody(chorale)
        assert built.tempo > 0
        assert built.timeSignature.numerator == chorale.time_signature[0]
        assert built.timeSignature.denominator == chorale.time_signature[1]


@pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
class TestResponseSideFieldsArePopulated:
    """Response-side models are always fully populated, so the UI never
    null-checks. A default that happens to be right is not the same thing."""

    def test_chords_populate_every_field_explicitly(self, engine):
        for chord in engine.harmonize(melody()).chords:
            dumped = chord.model_dump()
            for field in Chord.model_fields:
                assert field in dumped, f"{engine.id} chord missing {field}"
            assert isinstance(chord.extensions, list)
            assert chord.roman, "roman numeral is the only text the UI renders"
            assert 0 <= chord.root <= 11
            assert chord.inversion >= 0

    def test_chord_extensions_are_empty_for_common_practice_harmony(self, engine):
        # These engines write chorale harmony. If this ever fails, the contract's
        # `extensions` field has become load-bearing here and the UI needs to
        # render it.
        for chord in engine.harmonize(melody()).chords:
            assert chord.extensions == []

    def test_no_substitution_provenance_is_claimed(self, engine):
        for chord in engine.harmonize(melody()).chords:
            assert chord.substitutionOf is None
            assert chord.substitutionKind is None

    def test_harmonization_serializes_into_a_response(self, engine):
        result = engine.harmonize(melody())
        response = HarmonizeResponse(
            key=result.key,
            chords=result.chords,
            voices=result.voices,
            violations=result.violations,
            engine=engine.id,
            latencyMs=1.0,
        )
        payload = response.model_dump()
        assert payload["violations"] is not None
        for voice in payload["voices"]:
            for note in voice["notes"]:
                assert note["velocity"] is not None

    def test_voices_and_violations_use_contract_names(self, engine):
        result = engine.harmonize(melody())
        allowed = set(Voice.model_fields["name"].annotation.__args__)
        for voice in result.voices:
            assert voice.name in allowed
        for violation in result.violations:
            assert set(violation.voices) <= allowed
