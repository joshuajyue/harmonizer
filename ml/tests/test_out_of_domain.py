"""Engine behaviour on melodies that are not Bach chorale sopranos.

Every other test in this suite, and every piece in `ml/eval`, uses a Bach
soprano. That is a real gap: the harness measures distance from Bach on input
drawn from Bach, so it cannot see an engine that only works on Bach. The product
harmonizes whatever a user plays.

These tests are deliberately weak — they assert robustness, not quality, because
there is no ground truth for an invented tune. What they pin is that an engine
does not silently degenerate off-distribution, and they document the one place
where the engines currently do.
"""

import pytest

from contracts.schema import Melody, Note, TimeSignature
from ml.data.melody import voices_to_grid
from ml.eval.metrics import beat_chords, collect_activity, collect_defects
from ml.theory.pitch import Key

import ml.engines.baselines  # noqa: F401
import ml.engines.neural  # noqa: F401
import ml.engines.rules  # noqa: F401
from ml.engines.base import all_engines

ENGINES = [e for e in all_engines() if e.is_available() and e.id != "fixed_thirds"]
ENGINE_IDS = [e.id for e in ENGINES]


def melody(pitches, durations=None) -> Melody:
    durations = durations or [2.0] * len(pitches)
    notes, start = [], 0.0
    for pitch, duration in zip(pitches, durations):
        notes.append(Note(pitch=pitch, start=start, duration=duration))
        start += duration
    return Melody(notes=notes, tempo=100.0, timeSignature=TimeSignature(numerator=4, denominator=4))


#: Idioms a chorale corpus contains little or none of.
TUNES = {
    "pentatonic": melody([72, 74, 76, 79, 81, 79, 76, 74, 72, 74, 76, 79, 76, 74, 72, 72]),
    "blues_inflected": melody([72, 75, 76, 77, 79, 77, 76, 75, 72, 70, 72, 75, 76, 75, 72, 72]),
    "syncopated": melody(
        [72, 76, 79, 77, 76, 72, 74, 76, 72, 79, 77, 76, 74, 72, 71, 72],
        [1.5, 0.5, 1, 1, 1.5, 0.5, 1, 1, 1.5, 0.5, 1, 1, 1.5, 0.5, 1, 1],
    ),
    "modal_dorian": melody([62, 64, 65, 67, 69, 71, 69, 67, 65, 64, 65, 67, 65, 64, 62, 62]),
    "repeated_note": melody([72] * 12 + [71, 72, 74, 72]),
}


@pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
@pytest.mark.parametrize("name", sorted(TUNES))
class TestRobustnessOffDistribution:
    def test_produces_a_complete_harmonization(self, engine, name):
        result = engine.harmonize(TUNES[name], voice_count=4, seed=0)
        assert len(result.voices) == 4
        assert result.chords, f"{engine.id} produced no chords for {name}"
        for voice in result.voices:
            assert voice.notes, f"{engine.id} left {voice.name} empty on {name}"

    def test_stays_structurally_valid(self, engine, name):
        """Structural defects are the ones a listener notices instantly. An
        unfamiliar tune is no excuse for failing to resolve."""
        from ml.eval.metrics import STRUCTURAL_DEFECTS

        result = engine.harmonize(TUNES[name], voice_count=4, seed=0)
        lines = voices_to_grid(result.voices)
        counts = collect_defects(lines, Key(result.key.tonic, result.key.mode))
        structural = sum(counts.counts.get(kind, 0) for kind in STRUCTURAL_DEFECTS)
        assert structural == 0, f"{engine.id} on {name}: {counts.counts}"

    def test_does_not_collapse_to_one_chord(self, engine, name):
        """The degenerate failure: when the melody stops looking like a chorale,
        a search engine can find nothing worth moving to and simply sits still."""
        result = engine.harmonize(TUNES[name], voice_count=4, seed=0)
        lines = voices_to_grid(result.voices)
        activity = collect_activity(lines, Key(result.key.tonic, result.key.mode))
        assert activity.mean_classes_per_piece() >= 3, (
            f"{engine.id} used {activity.mean_classes_per_piece():.0f} distinct chords on {name}"
        )


class TestKnownWeaknessWideRange:
    """A melody with octave leaps is where both engines currently degrade.

    Pinned rather than asserted-away: the soprano is the user's and cannot be
    changed, so when it leaps across the whole staff the accompanying voices are
    forced into spacing and parallel problems that no amount of re-voicing can
    avoid. `rules` additionally goes nearly static, because a leaping melody
    makes every chord fit equally badly and the transition cost then dominates.

    This is a real limitation, documented in eval/REPORT.md. The test exists so
    that if someone fixes it the number moves and the test fails loudly, rather
    than the weakness quietly persisting.
    """

    WIDE = melody([60, 72, 64, 76, 67, 79, 72, 84, 79, 72, 67, 64, 62, 60, 67, 60])

    @pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
    def test_still_produces_valid_output(self, engine):
        result = engine.harmonize(self.WIDE, voice_count=4, seed=0)
        assert len(result.voices) == 4 and result.chords

    @pytest.mark.parametrize("engine", ENGINES, ids=ENGINE_IDS)
    def test_defect_rate_is_elevated_but_bounded(self, engine):
        result = engine.harmonize(self.WIDE, voice_count=4, seed=0)
        lines = voices_to_grid(result.voices)
        counts = collect_defects(lines, Key(result.key.tonic, result.key.mode))
        # Currently 5.9 (rules), 12.5 (neural_vl) and 32.0 (unpolished neural)
        # per 100 chords, against ~0.1 on chorale sopranos. Bounded so a
        # regression is still caught.
        assert counts.hard_error_rate() < 40.0, f"{engine.id}: {counts.hard_error_rate():.1f}"

    def test_the_voice_leading_veto_is_what_makes_the_model_robust(self):
        """An argument for the polish that on-distribution results do not make.

        On chorale sopranos the veto looks like a tidy-up: 10.4 hard defects down
        to 0.1. On a melody the model has never seen anything like, the
        unpolished model degrades to 32.0 while the polished one holds at 12.5.
        The constraint is not cosmetic, it is what stops the model falling apart
        outside its training distribution — which is where a user's melody lives.
        """
        from ml.engines.base import get_engine

        available = {e.id for e in ENGINES}
        if not {"neural", "neural_vl"} <= available:
            pytest.skip("needs both the polished and unpolished learned engines")

        rates = {}
        for engine_id in ("neural", "neural_vl"):
            result = get_engine(engine_id).harmonize(self.WIDE, voice_count=4, seed=0)
            lines = voices_to_grid(result.voices)
            key = Key(result.key.tonic, result.key.mode)
            rates[engine_id] = collect_defects(lines, key).hard_error_rate()
        assert rates["neural_vl"] < rates["neural"] * 0.75, rates
