"""The chord language model.

Two properties carry everything downstream. The distribution has to be a
distribution — the lattice sampler compares candidates whose available history
differs in length, so an unnormalised score would systematically prefer one
shape of candidate over another for no musical reason. And the model has to
survive a save/load round trip exactly, because the shipped asset is what the
engine actually runs; a lossy round trip would mean the thing measured in
REPORT.md and the thing served by the API are different models.

Fitting tests use synthetic sequences so they run without the 46 MB of corpora.
"""

import math

import pytest

from ml.reharm.chords import JazzChord
from ml.reharm.data import ChordSpan, Progression
from ml.reharm.model import BOS, EOS, ChordNGram, token_of, tokens_of

C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B = range(12)

II_V_I = [(2, "min7"), (7, "dom7"), (0, "maj7")]
IV_V_I = [(5, "maj7"), (7, "dom7"), (0, "maj7")]


@pytest.fixture(scope="module")
def shipped():
    model = ChordNGram.load()
    if model is None:
        pytest.skip("no shipped model: run `python -m ml.reharm.model`")
    return model


def fitted(order: int = 3) -> ChordNGram:
    model = ChordNGram(order=order)
    model.fit([II_V_I] * 20 + [IV_V_I] * 5)
    return model


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------


def test_tokens_are_key_relative():
    """The same ii-V-I in two keys is the same token sequence."""
    def progression(tonic: int) -> Progression:
        roots = [(tonic + 2) % 12, (tonic + 7) % 12, tonic]
        qualities = ["min7", "dom7", "maj7"]
        spans = [
            ChordSpan(float(i) * 4, 4.0, JazzChord(root=root, quality=quality))
            for i, (root, quality) in enumerate(zip(roots, qualities))
        ]
        return Progression(spans=spans, tonic=tonic, mode="major")

    assert tokens_of(progression(0)) == tokens_of(progression(7)) == II_V_I


def test_token_drops_extensions():
    plain = JazzChord(root=G, quality="dom7")
    altered = JazzChord(root=G, quality="dom7", extensions=("b9", "b13"))
    assert token_of(plain, C) == token_of(altered, C) == (7, "dom7")


# ---------------------------------------------------------------------------
# It has to be a distribution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "history",
    [
        [],
        [BOS, BOS],
        [(2, "min7")],
        [(2, "min7"), (7, "dom7")],
        [(6, "dim7")],
        [(11, "halfdim7"), (4, "dom7")],
    ],
)
def test_probabilities_sum_to_one(shipped, history):
    """Including for histories the model has never seen, which is the point.

    A short history backs off by substituting the lower-order distribution into
    the higher-order slot rather than dropping the term, so the mixture stays a
    mixture of proper distributions. The residual is the uniform term over one
    unused slot, not a modelling error.
    """
    total = sum(shipped.probability(token, history) for token in sorted(shipped.vocabulary))
    assert total == pytest.approx(1.0, abs=1e-3)


def test_probabilities_are_strictly_positive(shipped):
    """Nothing is impossible, or the lattice can lose a legal path to -inf."""
    unseen = (6, "minmaj7")
    assert shipped.probability(unseen, [(6, "dim7"), (1, "aug")]) > 0
    assert math.isfinite(shipped.log_probability(unseen, []))


def test_history_length_does_not_change_the_normalisation(shipped):
    """Candidates with different history lengths must be comparable."""
    vocabulary = sorted(shipped.vocabulary)
    short = sum(shipped.probability(token, [(2, "min7")]) for token in vocabulary)
    long = sum(shipped.probability(token, [(0, "maj7"), (2, "min7")]) for token in vocabulary)
    assert short == pytest.approx(long, abs=1e-3)


# ---------------------------------------------------------------------------
# It has to have learned something
# ---------------------------------------------------------------------------


def test_the_model_knows_that_ii_goes_to_v(shipped):
    """68% of the mass, from 87,000 chords of real jazz."""
    best = max(shipped.vocabulary, key=lambda token: shipped.probability(token, [(2, "min7")]))
    assert best == (7, "dom7")
    assert shipped.probability((7, "dom7"), [(2, "min7")]) > 0.5


def test_context_actually_changes_the_prediction():
    model = fitted()
    after_two = model.probability((7, "dom7"), [(2, "min7")])
    after_five = model.probability((7, "dom7"), [(5, "maj7")])
    assert after_two > 0 and after_five > 0
    assert model.probability((0, "maj7"), [(7, "dom7")]) > model.probability((0, "maj7"), [(2, "min7")])


def test_higher_order_fits_a_repetitive_corpus_better():
    sequences = [II_V_I] * 20 + [IV_V_I] * 5
    orders = {order: ChordNGram(order=order).fit(sequences).perplexity(sequences) for order in (1, 2, 3)}
    assert orders[3] <= orders[2] <= orders[1]


def test_perplexity_is_finite_on_unseen_material():
    model = fitted()
    assert math.isfinite(model.perplexity([[(1, "aug"), (6, "minmaj7")]]))


def test_sequences_are_padded_and_terminated():
    model = fitted()
    assert model.unigram[EOS] == 25
    assert BOS not in model.vocabulary, "BOS is context, never a prediction"


# ---------------------------------------------------------------------------
# The shipped asset has to be the model that was measured
# ---------------------------------------------------------------------------


def test_round_trip_is_exact(tmp_path):
    model = fitted()
    path = model.save(tmp_path / "ngram.json")
    reloaded = ChordNGram.load(path)
    assert reloaded is not None
    assert reloaded.order == model.order
    assert reloaded.lambdas == model.lambdas
    assert reloaded.unigram == model.unigram
    for token in sorted(model.vocabulary):
        for history in ([], [(2, "min7")], [(2, "min7"), (7, "dom7")]):
            assert reloaded.probability(token, history) == pytest.approx(
                model.probability(token, history)
            )


def test_missing_or_corrupt_model_returns_none_rather_than_raising(tmp_path):
    """The engine degrades to unavailable; it must not take discovery down."""
    assert ChordNGram.load(tmp_path / "absent.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert ChordNGram.load(broken) is None


def test_shipped_model_is_the_configuration_the_report_quotes(shipped):
    assert shipped.order == 3
    assert shipped.trained_on.get("treebank_chords", 0) > 50_000
    assert shipped.trained_on.get("weimar_chords", 0) > 20_000
    assert len(shipped.vocabulary) > 100
