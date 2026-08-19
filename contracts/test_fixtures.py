"""Validates the shared fixtures in contracts/examples/ against the schema.

If the contract changes and the fixtures are not regenerated, this fails loudly here
rather than silently breaking the frontend's mock mode and the backend's tests at the
same time. Also asserts the deliberate voice-leading defects are still genuinely
present, since the frontend develops its violation rendering against them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent / "examples"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts.schema import HarmonizeRequest, HarmonizeResponse  # noqa: E402

VOICE_RANGES = {
    "soprano": (60, 79),
    "alto": (55, 74),
    "tenor": (48, 67),
    "bass": (40, 60),
}


def load_request() -> HarmonizeRequest:
    return HarmonizeRequest.model_validate(json.loads((EXAMPLES / "melody.request.json").read_text()))


def load_response() -> HarmonizeResponse:
    return HarmonizeResponse.model_validate(
        json.loads((EXAMPLES / "harmonize.response.json").read_text())
    )


def test_fixtures_validate_against_schema():
    assert load_request().melody.notes, "melody fixture is empty"
    assert load_response().voices, "response fixture has no voices"


def test_soprano_retains_the_input_melody():
    """The engine contract says the melody is kept as the soprano voice."""
    melody = load_request().melody.notes
    soprano = next(v for v in load_response().voices if v.name == "soprano")
    assert [(n.start, n.duration, n.pitch) for n in soprano.notes] == [
        (n.start, n.duration, n.pitch) for n in melody
    ]


def test_all_voices_are_in_range():
    for voice in load_response().voices:
        low, high = VOICE_RANGES[voice.name]
        for note in voice.notes:
            assert low <= note.pitch <= high, f"{voice.name} note {note.pitch} out of range at {note.start}"


def test_voices_are_rhythmically_aligned():
    """Every voice must share the same onset grid, or the piano roll cannot align lanes."""
    voices = load_response().voices
    grid = [(n.start, n.duration) for n in voices[0].notes]
    for voice in voices[1:]:
        assert [(n.start, n.duration) for n in voice.notes] == grid, f"{voice.name} is off-grid"


def test_deliberate_defects_are_actually_present():
    """The fixture advertises three defects; verify they are real, not just labelled."""
    response = load_response()
    pitch_at = {
        v.name: {n.start: n.pitch for n in v.notes} for v in response.voices
    }
    kinds = {v.kind for v in response.violations}
    assert {"parallel_fifths", "parallel_octaves", "voice_crossing"} <= kinds

    # Parallel fifths, tenor/bass, beat 8 -> 9.
    assert pitch_at["tenor"][8] - pitch_at["bass"][8] == 7
    assert pitch_at["tenor"][9] - pitch_at["bass"][9] == 7
    assert (pitch_at["tenor"][9] - pitch_at["tenor"][8]) == (pitch_at["bass"][9] - pitch_at["bass"][8]) != 0

    # Parallel octaves, soprano/bass, beat 20 -> 22.
    assert (pitch_at["soprano"][20] - pitch_at["bass"][20]) % 12 == 0
    assert (pitch_at["soprano"][22] - pitch_at["bass"][22]) % 12 == 0
    assert (pitch_at["soprano"][22] - pitch_at["soprano"][20]) == (
        pitch_at["bass"][22] - pitch_at["bass"][20]
    ) != 0

    # Voice crossing, tenor above alto, beat 25.
    assert pitch_at["tenor"][25] > pitch_at["alto"][25]


def test_violations_point_at_real_beats():
    response = load_response()
    onsets = {n.start for n in response.voices[0].notes}
    for violation in response.violations:
        assert violation.start in onsets, f"violation at {violation.start} is not on an onset"
        assert violation.voices, "violation must name the voices involved"


def test_declared_violations_match_the_detector_exactly():
    """The fixture's violations list must be exhaustive, not merely illustrative.

    An earlier draft declared three defects while the voicing actually contained
    eight. That is dangerous in both directions: it makes the fixture useless as a
    detector test corpus, and it invites someone to "fix" a correct detector to
    match wrong data.

    Voice identity is compared, not just (beat, kind). Comparing only the latter let
    every declared voice pair be replaced with ["soprano","alto"] while the test
    still passed — yet the frontend highlights exactly those lanes in the piano roll.
    Skipped if the ml package is unavailable.
    """
    try:
        from ml.theory.voicing import find_parallels, find_voice_crossings, texture_from_voices
    except ImportError:  # pragma: no cover - ml deps not installed
        import pytest

        pytest.skip("ml package not importable")

    response = load_response()
    order = ["soprano", "alto", "tenor", "bass"]
    by_name = {v.name: v.notes for v in response.voices}
    starts = [n.start for n in by_name["soprano"]]
    texture = texture_from_voices([[n.pitch for n in by_name[name]] for name in order])

    detected: set[tuple[float, str, frozenset[str]]] = set()
    for i in range(1, len(texture.grid)):
        for upper, lower, kind in find_parallels(texture.grid[i - 1], texture.grid[i]):
            detected.add((starts[i], kind, frozenset({order[upper], order[lower]})))
    for i in range(len(texture.grid)):
        for upper, lower in find_voice_crossings(texture.grid[i]):
            detected.add(
                (starts[i], "voice_crossing", frozenset({order[upper], order[lower]}))
            )

    declared = {(v.start, v.kind, frozenset(v.voices)) for v in response.violations}
    assert detected == declared, (
        "fixture voicing and declared violations disagree.\n"
        f"  undeclared defects: {sorted(map(str, detected - declared))}\n"
        f"  declared but absent: {sorted(map(str, declared - detected))}"
    )


if __name__ == "__main__":
    import traceback

    tests = [fn for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception:
            failures += 1
            print(f"  FAIL  {test.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failures}/{len(tests)} fixture checks passed")
    raise SystemExit(1 if failures else 0)
