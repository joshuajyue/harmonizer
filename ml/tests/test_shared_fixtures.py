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


#: Kinds the fixture's `violations` array is authoritative about. For these, the
#: detector's findings and the fixture's declarations must agree exactly in both
#: directions. Everything else the detector reports (overlap, melodic intervals,
#: tendency tones) is real but not something the fixture undertakes to enumerate.
STRUCTURAL_KINDS = ("parallel_fifths", "parallel_octaves", "voice_crossing", "range")


def as_key(kind, offset, voices):
    return (kind, round(float(offset), 6), tuple(sorted(voices)))


def declared_structural(response):
    return {
        as_key(v.kind, v.start, [VOICE_INDEX[n] for n in v.voices])
        for v in response.violations if v.kind in STRUCTURAL_KINDS
    }


def detected_structural(defects):
    return {as_key(d.kind, d.offset, d.voices) for d in defects if d.kind in STRUCTURAL_KINDS}


def previous_distinct(grid, beat: float):
    """The sonority immediately before `beat`, skipping steps where nothing moved.

    Voice-leading rules apply between successive *sonorities*, not grid steps, so
    a held chord is one chord however many sixteenths it occupies.
    """
    step = int(round(beat * 4))
    current = tuple(line[step] for line in grid)
    for earlier in range(step - 1, -1, -1):
        sonority = tuple(line[earlier] for line in grid)
        if sonority != current:
            return sonority
    return None


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

    def test_detected_and_declared_agree_exactly(self, response, defects):
        """Bidirectional, and derived from the fixture rather than hardcoded.

        Whatever the fixture declares is what the detectors must find, and
        nothing more. Written this way so a re-voicing of the fixture needs no
        edit here: a mismatch in EITHER direction fails loudly, which is the
        failure mode that let a fixture carrying five unintended parallels
        describe itself as carrying none.
        """
        declared = declared_structural(response)
        detected = detected_structural(defects)
        missed = declared - detected
        invented = detected - declared
        assert not missed, f"declared but not detected: {sorted(missed)}"
        assert not invented, f"detected but not declared: {sorted(invented)}"

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
    """Re-derive every declared defect from raw pitch arithmetic on the fixture.

    Independent of both the detector and the fixture's labels: if the two shared
    a misconception, these still fail. Driven off the declared `violations`
    array rather than hardcoded beats, so it keeps working after a re-voicing —
    and so a defect the fixture merely *claims* cannot pass unchecked.
    """

    def test_every_declared_parallel_is_arithmetically_real(self, response, grid):
        checked = 0
        for violation in response.violations:
            if violation.kind not in ("parallel_fifths", "parallel_octaves"):
                continue
            upper, lower = sorted(
                (VOICE_INDEX[n] for n in violation.voices), key=lambda v: v
            )
            after = at(grid, violation.start)
            before = previous_distinct(grid, violation.start)
            assert before is not None, f"nothing precedes beat {violation.start}"

            step = 7 if violation.kind == "parallel_fifths" else 0
            gap_before = abs(before[upper] - before[lower])
            gap_after = abs(after[upper] - after[lower])
            assert gap_before % 12 == step, f"beat {violation.start}: interval before is {gap_before}"
            assert gap_after % 12 == step, f"beat {violation.start}: interval after is {gap_after}"

            moved_upper = after[upper] - before[upper]
            moved_lower = after[lower] - before[lower]
            assert moved_upper != 0 and moved_lower != 0, "a sustained interval is not a parallel"
            assert (moved_upper > 0) == (moved_lower > 0), "contrary motion is not a parallel"
            checked += 1
        assert checked, "fixture declares no parallels to verify"

    def test_every_declared_crossing_is_arithmetically_real(self, response, grid):
        checked = 0
        for violation in response.violations:
            if violation.kind != "voice_crossing":
                continue
            upper, lower = sorted(VOICE_INDEX[n] for n in violation.voices)
            sonority = at(grid, violation.start)
            assert sonority[lower] > sonority[upper], (
                f"beat {violation.start}: {VOICE_NAMES[lower]} {sonority[lower]} is not above "
                f"{VOICE_NAMES[upper]} {sonority[upper]}"
            )
            checked += 1
        assert checked, "fixture declares no crossings to verify"

    def test_no_undeclared_beat_contains_a_parallel(self, response, grid):
        """The exhaustive sweep the fixture's own generator did not do.

        Walks every successive pair of distinct sonorities and every voice pair,
        by hand, and asserts any perfect parallel it finds is one the fixture
        declares. This is what would have caught the five unintended parallels
        without needing the detector to be trusted at all.
        """
        declared = {
            (round(float(v.start), 6), tuple(sorted(VOICE_INDEX[n] for n in v.voices)))
            for v in response.violations
            if v.kind in ("parallel_fifths", "parallel_octaves")
        }
        length = len(grid[0])
        unexpected = []
        previous, previous_step = None, None
        for step in range(length):
            sonority = tuple(line[step] for line in grid)
            if sonority == previous:
                continue
            if previous is not None:
                for high in range(4):
                    for low in range(high + 1, 4):
                        if -1 in (previous[high], previous[low], sonority[high], sonority[low]):
                            continue
                        moved_high = sonority[high] - previous[high]
                        moved_low = sonority[low] - previous[low]
                        if moved_high == 0 or moved_low == 0:
                            continue
                        if (moved_high > 0) != (moved_low > 0):
                            continue
                        gap_before = abs(previous[high] - previous[low]) % 12
                        gap_after = abs(sonority[high] - sonority[low]) % 12
                        if gap_before == gap_after and gap_before in (0, 7):
                            beat = round(step / 4, 6)
                            if (beat, (high, low)) not in declared:
                                unexpected.append(
                                    (beat, VOICE_NAMES[high], VOICE_NAMES[low], gap_before)
                                )
            previous, previous_step = sonority, step
        assert not unexpected, f"undeclared parallels found by hand: {unexpected}"

    def test_the_beats_either_side_of_a_crossing_are_ordered(self, response, grid):
        """A declared crossing is a single event, not a persistent mis-ordering."""
        for violation in response.violations:
            if violation.kind != "voice_crossing":
                continue
            for offset in (-1.0, 1.0):
                beat = violation.start + offset
                if not 0 <= beat * 4 < len(grid[0]):
                    continue
                sonority = at(grid, beat)
                if -1 in sonority:
                    continue
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
        from ml.tests._engines import chorale_engines

        return chorale_engines()

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
