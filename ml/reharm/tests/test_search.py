"""Walking the lattice: search versus sampling.

The two properties that matter are the ones the strategic argument rests on.
Sampling has to actually produce different answers — otherwise it buys nothing
over search — and it has to be exactly reproducible from a seed, otherwise it
cannot be shipped behind an API that promises determinism. Both are tested
here, along with the identity that makes the comparison fair: as the
temperature goes to zero, the sampler converges on the search.
"""

import pytest

from ml.reharm.melodies import TRADITIONAL
from ml.reharm.model import ChordNGram
from ml.reharm.search import (
    HybridScorer,
    ModelScorer,
    ReharmConfig,
    RuleScorer,
    build_lattice,
    realize,
    sample,
    viterbi,
)
from ml.reharm.skeleton import skeleton_from_rules


@pytest.fixture(scope="module")
def model():
    loaded = ChordNGram.load()
    if loaded is None:
        pytest.skip("no trained model: run `python -m ml.reharm.model`")
    return loaded


@pytest.fixture(scope="module")
def lattice():
    skeleton = skeleton_from_rules(TRADITIONAL["twinkle"])
    return build_lattice(skeleton, ReharmConfig()), skeleton


def test_every_unit_has_at_least_one_candidate(lattice):
    built, _ = lattice
    assert built.candidates
    assert all(candidates for candidates in built.candidates)


def test_identity_is_always_available(lattice):
    built, _ = lattice
    for unit, candidates in zip(built.units, built.candidates):
        assert any(
            candidate.first.root == unit.base.root for candidate in candidates
        ), "the base chord must always remain choosable"


def test_viterbi_is_deterministic(lattice):
    built, _ = lattice
    scorer = RuleScorer(built, ReharmConfig())
    first = [candidate.label() for candidate in viterbi(built, scorer)]
    second = [candidate.label() for candidate in viterbi(built, scorer)]
    assert first == second


def test_sampling_is_reproducible_from_a_seed(lattice, model):
    built, _ = lattice
    scorer = ModelScorer(built, model, ReharmConfig())
    first = [c.label() for c in sample(built, scorer, temperature=1.2, top_p=0.95, seed=11)]
    second = [c.label() for c in sample(built, scorer, temperature=1.2, top_p=0.95, seed=11)]
    assert first == second


def test_sampling_actually_samples(lattice, model):
    """One-to-many is the whole premise: different seeds, different answers."""
    built, _ = lattice
    scorer = ModelScorer(built, model, ReharmConfig())
    outputs = {
        tuple(c.label() for c in sample(built, scorer, temperature=1.3, top_p=0.98, seed=seed))
        for seed in range(8)
    }
    assert len(outputs) > 1


def test_zero_temperature_falls_back_to_the_argmax(lattice, model):
    built, _ = lattice
    scorer = ModelScorer(built, model, ReharmConfig())
    sampled = [c.label() for c in sample(built, scorer, temperature=0.0, seed=3)]
    searched = [c.label() for c in viterbi(built, scorer)]
    assert sampled == searched


def test_cold_sampling_converges_on_the_argmax(lattice, model):
    """Temperature is a real dial, not a switch: T -> 0 approaches Viterbi.

    0.02 is not cold enough to be a fair test of that, and finding out why was
    worth the test: at T = 0.02 a path 0.035 nats worse than the best still
    carries about a sixth of the probability mass, so the sampler is right to
    pick it sometimes. The convergence is genuine, it just needs a temperature
    where the odds ratio is actually decisive.
    """
    built, _ = lattice
    scorer = ModelScorer(built, model, ReharmConfig())
    searched = [c.label() for c in viterbi(built, scorer)]
    cold = [
        [c.label() for c in sample(built, scorer, temperature=0.002, top_p=1.0, seed=seed)]
        for seed in range(4)
    ]
    assert all(path == searched for path in cold)


def test_higher_temperature_is_more_adventurous(lattice, model):
    built, _ = lattice
    scorer = ModelScorer(built, model, ReharmConfig())

    def distinct(temperature: float) -> int:
        return len({
            tuple(c.label() for c in sample(built, scorer, temperature=temperature, top_p=1.0, seed=seed))
            for seed in range(12)
        })

    assert distinct(2.0) >= distinct(0.3)


def test_realize_covers_the_melody_without_gaps(lattice, model):
    built, skeleton = lattice
    scorer = HybridScorer(built, model, ReharmConfig())
    result = realize(built, sample(built, scorer, temperature=1.0, seed=4), skeleton)
    assert result.spans
    for previous, current in zip(result.spans, result.spans[1:]):
        assert current.start == pytest.approx(previous.stop, abs=1e-6)
    assert result.spans[0].start == pytest.approx(skeleton.units[0].start)
    assert result.spans[-1].stop == pytest.approx(skeleton.units[-1].stop)


def test_adventure_dial_moves_the_result(lattice, model):
    built, skeleton = lattice
    from ml.reharm.metrics import distance

    def travelled(adventure: float) -> float:
        config = ReharmConfig(adventure=adventure)
        local = build_lattice(skeleton, config)
        result = realize(local, viterbi(local, RuleScorer(local, config)), skeleton)
        return distance(skeleton.progression(), result.progression()).root_change_rate

    assert travelled(1.0) > travelled(0.0)
