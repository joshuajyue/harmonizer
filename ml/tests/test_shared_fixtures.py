"""Voice-leading detectors checked against the shared fixture in `contracts/examples/`.

This is the most valuable test in the suite, because it is the only one where the
expected answer was produced by someone else. Every other check of these
detectors was written by the same person who wrote the detectors, so a shared
misconception would pass silently — which is precisely how v1 ended up unable to
tell whether its model was better than its rules.

`contracts/examples/harmonize.response.json` carries three deliberate,
independently verified defects. The detectors in `ml/theory/voicing.py` must find
exactly those three, at exactly those beats, between exactly those voices, and
must invent no additional parallel fifths, parallel octaves or voice crossings
anywhere else in the piece.

The defects are also re-derived here from raw pitch arithmetic on the fixture's
own note data, independently of both the detector and the fixture's labels. If
the detector and the fixture happened to share a mistake, that check still fails.
"""

import json
from pathlib import Path

import pytest

from ml.data.melody import voices_to_grid
from ml.theory.pitch import Key
from ml.theory.voicing import ALTO, BASS, SOPRANO, TENOR, VOICE_NAMES, VOICE_RANGES, analyze_texture

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"
REQUEST_PATH = EXAMPLES / "melody.request.json"
RESPONSE_PATH = EXAMPLES / "harmonize.response.json"

if not (REQUEST_PATH.exists() and RESPONSE_PATH.exists()):  # pragma: no cover
    pytest.skip(
        f"shared fixtures not found under {EXAMPLES}; they are owned by the lead, "
        "so this suite skips rather than failing someone else's refactor",
        allow_module_level=True,
    )

VOICE_INDEX = {"soprano": SOPRANO, "alto": ALTO, "tenor": TENOR, "bass": BASS}

#: Ranges the fixture is documented to respect. Deliberately re-stated rather
#: than imported: if `ml/theory/voicing.py` ever widens a range, this still
#: pins the fixture to what the contract says about it.
FIXTURE_RANGES = {SOPRANO: (60, 79), ALTO: (55, 74), TENOR: (48, 67), BASS: (40, 60)}


@pytest.fixture(scope="module")
def response():
    from contracts.schema import HarmonizeResponse

    return HarmonizeResponse(**json.loads(RESPONSE_PATH.read_text()))


@pytest.fixture(scope="module")
def request_body():
    from contracts.schema import HarmonizeRequest

    return HarmonizeRequest(**json.loads(REQUEST_PATH.read_text()))


@pytest.fixture(scope="module")
def grid(response):
    """The fixture's four voices on the 16th grid, as the detectors see them."""
    return voices_to_grid(response.voices)


@pytest.fixture(scope="module")
def defects(response, grid):
    from ml.eval.metrics import step_chords, to_texture

    key = Key(response.key.tonic, response.key.mode)
    return analyze_texture(to_texture(grid), key, step_chords(grid, key))


def at(grid, beat: float):
    """The sounding sonority at a beat, as (S, A, T, B)."""
    step = int(round(beat * 4))
    return tuple(line[step] for line in grid)


def of_kind(defects, kind):
    return [d for d in defects if d.kind == kind]


class TestAdvertisedDefectsAreFound:
    """Exactly the three the fixture declares, exactly where it declares them."""

    def test_parallel_fifths_at_beat_nine_between_tenor_and_bass(self, defects):
        found = of_kind(defects, "parallel_fifths")
        assert len(found) == 1, [(d.offset, d.voices) for d in found]
        assert found[0].offset == 9.0
        assert set(found[0].voices) == {TENOR, BASS}

    def test_parallel_octaves_at_beat_twentytwo_between_soprano_and_bass(self, defects):
        found = of_kind(defects, "parallel_octaves")
        assert len(found) == 1, [(d.offset, d.voices) for d in found]
        assert found[0].offset == 22.0
        assert set(found[0].voices) == {SOPRANO, BASS}

    def test_voice_crossing_at_beat_twentyfive_between_alto_and_tenor(self, defects):
        found = of_kind(defects, "voice_crossing")
        assert len(found) == 1, [(d.offset, d.voices) for d in found]
        assert found[0].offset == 25.0
        assert set(found[0].voices) == {ALTO, TENOR}

    def test_every_declared_violation_is_reproduced(self, response, defects):
        """Generic form: whatever the fixture declares, the detectors must find."""
        for violation in response.violations:
            matches = [
                d for d in defects
                if d.kind == violation.kind
                and d.offset == pytest.approx(violation.start)
                and set(d.voices) == {VOICE_INDEX[name] for name in violation.voices}
            ]
            assert matches, (
                f"fixture declares {violation.kind} at beat {violation.start} between "
                f"{violation.voices}, detectors did not find it"
            )


class TestNoFalsePositives:
    """The detectors must not invent defects the fixture does not contain.

    A detector that fires everywhere would pass the tests above and be useless.
    """

    def test_no_extra_parallels_or_crossings(self, defects):
        counts = {kind: len(of_kind(defects, kind)) for kind in
                  ("parallel_fifths", "parallel_octaves", "voice_crossing")}
        assert counts == {"parallel_fifths": 1, "parallel_octaves": 1, "voice_crossing": 1}

    def test_no_range_violations(self, defects):
        assert of_kind(defects, "range") == []

    def test_fixture_respects_the_ranges_it_documents(self, response):
        for voice in response.voices:
            low, high = FIXTURE_RANGES[VOICE_INDEX[voice.name]]
            for note in voice.notes:
                assert low <= note.pitch <= high, f"{voice.name} {note.pitch} outside {low}-{high}"

    def test_only_the_crossing_produces_knock_on_defects(self, defects):
        """The extra defects are consequences of the planted crossing, not noise.

        The tenor has to leap back down after rising above the alto, which is a
        real melodic defect and a real overlap. Pinning them means a future
        change to the detectors that silently stops reporting them is caught.
        """
        extra = [d for d in defects if d.kind not in
                 ("parallel_fifths", "parallel_octaves", "voice_crossing")]
        kinds = sorted({d.kind for d in extra})
        assert kinds == ["awkward_melodic_interval", "frustrated_leading_tone", "voice_overlap"]
        for defect in extra:
            if defect.kind != "frustrated_leading_tone":
                assert defect.offset == 26.0, f"{defect.kind} at {defect.offset}, expected beat 26"


class TestDefectsAreGenuineNotJustLabelled:
    """Re-derive each defect from raw pitch arithmetic on the fixture's notes.

    Independent of both the detector and the fixture's own labels: if the two
    shared a misconception, these still fail.
    """

    def test_beat_nine_really_is_consecutive_perfect_fifths(self, grid):
        before, after = at(grid, 8.0), at(grid, 9.0)
        assert (after[TENOR] - before[TENOR]) != 0 and (after[BASS] - before[BASS]) != 0
        assert (before[TENOR] - before[BASS]) % 12 == 7
        assert (after[TENOR] - after[BASS]) % 12 == 7
        # Same direction is what makes it parallel rather than contrary.
        assert (after[TENOR] - before[TENOR]) > 0 and (after[BASS] - before[BASS]) > 0

    def test_beat_twentytwo_really_is_consecutive_octaves(self, grid):
        before, after = at(grid, 21.0), at(grid, 22.0)
        assert (before[SOPRANO] - before[BASS]) % 12 == 0
        assert (after[SOPRANO] - after[BASS]) % 12 == 0
        assert (after[SOPRANO] - before[SOPRANO]) == (after[BASS] - before[BASS]) != 0

    def test_beat_twentyfive_really_is_a_crossing(self, grid):
        sonority = at(grid, 25.0)
        assert sonority[TENOR] > sonority[ALTO]

    def test_the_beats_either_side_are_correctly_ordered(self, grid):
        """The crossing is a single event, not a persistent mis-ordering."""
        for beat in (24.0, 26.0):
            sonority = at(grid, beat)
            assert sonority[SOPRANO] >= sonority[ALTO] >= sonority[TENOR] >= sonority[BASS]


class TestFixtureShape:
    def test_voices_share_one_onset_grid(self, response):
        """Independently re-checked because the UI's lane alignment depends on it."""
        onsets = [{note.start for note in voice.notes} for voice in response.voices]
        union = set().union(*onsets)
        for voice, starts in zip(response.voices, onsets):
            assert starts <= union
        assert onsets[SOPRANO] == union, "soprano should carry every onset in the grid"

    def test_soprano_retains_the_input_melody(self, request_body, response):
        melody = [(n.pitch, n.start, n.duration) for n in request_body.melody.notes]
        soprano = [(n.pitch, n.start, n.duration) for n in response.voices[SOPRANO].notes]
        assert soprano == melody

    def test_voices_are_in_satb_order(self, response):
        assert [v.name for v in response.voices] == ["soprano", "alto", "tenor", "bass"]


class TestEnginesOnTheSharedMelody:
    """Smoke test on non-Bach input.

    Every other engine test uses a chorale or a toy scale. This is an original
    tune the engines have never seen, in a shape a user would actually send.
    """

    @staticmethod
    def engines():
        import ml.engines.baselines  # noqa: F401
        import ml.engines.neural  # noqa: F401
        import ml.engines.rules  # noqa: F401
        from ml.engines.base import all_engines

        return [e for e in all_engines() if e.is_available()]

    def test_every_engine_harmonizes_it(self, request_body):
        from contracts.schema import HarmonizeResponse

        for engine in self.engines():
            result = engine.harmonize(
                request_body.melody,
                voice_count=request_body.options.voiceCount,
                temperature=request_body.options.temperature,
                seed=request_body.options.seed,
            )
            assert len(result.voices) == request_body.options.voiceCount
            assert result.chords, f"{engine.id} produced no chord labels"
            # Must serialize into the contract's response model unchanged.
            HarmonizeResponse(
                key=result.key, chords=result.chords, voices=result.voices,
                violations=result.violations, engine=engine.id, latencyMs=1.0,
            )

    def test_every_engine_retains_the_melody_as_soprano(self, request_body):
        expected = [(n.pitch, n.start, n.duration) for n in request_body.melody.notes]
        for engine in self.engines():
            result = engine.harmonize(request_body.melody, voice_count=4)
            soprano = [(n.pitch, n.start, n.duration) for n in result.voices[SOPRANO].notes]
            assert soprano == expected, engine.id

    def test_every_engine_finds_the_declared_key(self, request_body):
        melody = request_body.melody.model_copy(update={"key": None})
        for engine in self.engines():
            result = engine.harmonize(melody, voice_count=4)
            assert (result.key.tonic, result.key.mode) == (0, "major"), engine.id

    def test_the_rule_engine_writes_no_parallels_on_this_melody(self, request_body):
        """The fixture's own harmonization contains parallels by design. A real
        engine should not reproduce them on the same tune."""
        from ml.eval.metrics import step_chords, to_texture
        from ml.engines.base import get_engine

        import ml.engines.rules  # noqa: F401

        result = get_engine("rules").harmonize(request_body.melody, voice_count=4)
        lines = voices_to_grid(result.voices)
        key = Key(result.key.tonic, result.key.mode)
        found = analyze_texture(to_texture(lines), key, step_chords(lines, key))
        parallels = [d for d in found if d.kind in ("parallel_fifths", "parallel_octaves")]
        assert parallels == [], [(d.offset, [VOICE_NAMES[v] for v in d.voices]) for d in parallels]
