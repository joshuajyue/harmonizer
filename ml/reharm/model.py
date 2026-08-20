"""The learned half: a chord-sequence model estimated from real jazz.

Deliberately small. The Jazz Harmony Treebank is 1170 tunes and about 59,000
chords; the Weimar changes add another 28,000. That is enough data to estimate
an interpolated trigram over key-relative chord tokens well, and not remotely
enough to justify a transformer — the honest thing to do with 87,000 tokens and
a vocabulary under 200 is to fit the model the data supports and say so. The
comparison of orders 1/2/3 on a held-out split is printed by `main()`, so the
choice is a measurement rather than an assertion.

Tokens are key-relative `(degree, quality)` pairs. That is the same correction
the rest of the project already made for pitch: an absolute representation
would force the model to learn the same ii-V-I 12 times, once per key, from a
corpus that cannot afford it.

What this model is *for* matters for how it is used. It is not asked to
generate a progression from nothing — it scores candidate substitutions in a
lattice that is already anchored to the tune and already melody-safe. So its
job is narrow and its data is adequate for that job.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .chords import JazzChord
from .data import Progression

ASSETS = Path(__file__).resolve().parent / "assets"
MODEL_PATH = ASSETS / "jazz_ngram.json"

#: Sequence boundary tokens.
BOS: tuple[int, str] = (-1, "<s>")
EOS: tuple[int, str] = (-2, "</s>")

Token = tuple[int, str]


def token_of(chord: JazzChord, tonic: int) -> Token:
    """Key-relative token for a chord. Extensions are deliberately dropped.

    Lead sheets notate extensions only 1.5% of the time (the oracle measures
    it), so modelling them here would be modelling the notation habits of the
    transcriber rather than the harmony. Tensions are chosen later, from the
    melody, where there is actual evidence for them.
    """
    return ((chord.root - tonic) % 12, chord.quality)


def tokens_of(progression: Progression) -> list[Token]:
    return [token_of(chord, progression.tonic) for chord in progression.chords]


@dataclass
class ChordNGram:
    """Interpolated trigram over key-relative chord tokens."""

    order: int = 3
    #: Interpolation weights for (trigram, bigram, unigram, uniform).
    lambdas: tuple[float, float, float, float] = (0.45, 0.35, 0.19, 0.01)
    unigram: Counter = field(default_factory=Counter)
    bigram: dict[Token, Counter] = field(default_factory=lambda: defaultdict(Counter))
    trigram: dict[tuple[Token, Token], Counter] = field(default_factory=lambda: defaultdict(Counter))
    vocabulary: set[Token] = field(default_factory=set)
    trained_on: dict[str, int] = field(default_factory=dict)

    # -- fitting -----------------------------------------------------------

    def fit(self, sequences: Iterable[Sequence[Token]]) -> ChordNGram:
        for sequence in sequences:
            padded = [BOS, BOS, *sequence, EOS]
            for index in range(2, len(padded)):
                token = padded[index]
                self.unigram[token] += 1
                self.vocabulary.add(token)
                self.bigram[padded[index - 1]][token] += 1
                self.trigram[(padded[index - 2], padded[index - 1])][token] += 1
        self.vocabulary.discard(BOS)
        return self

    # -- scoring -----------------------------------------------------------

    def probability(self, token: Token, history: Sequence[Token] = ()) -> float:
        """Interpolated P(token | history), always strictly positive.

        A short history backs off by *substituting* the lower-order
        distribution into the higher-order slot rather than by dropping the
        term, so the result stays a mixture of proper distributions and
        therefore still sums to one. That matters here because the lattice
        search compares candidates whose available history differs in length,
        and an unnormalised score would systematically prefer one shape.
        """
        size = max(1, len(self.vocabulary) + 1)
        l3, l2, l1, l0 = self.lambdas

        total = sum(self.unigram.values()) or 1
        p_uniform = 1.0 / size
        p_unigram = self.unigram.get(token, 0) / total

        if history:
            previous = history[-1]
            context = self.bigram.get(previous)
            p_bigram = (context.get(token, 0) / sum(context.values())) if context else p_unigram
        else:
            p_bigram = p_unigram

        if len(history) >= 2:
            context3 = self.trigram.get((history[-2], history[-1]))
            p_trigram = (context3.get(token, 0) / sum(context3.values())) if context3 else p_bigram
        else:
            p_trigram = p_bigram

        if self.order == 1:
            return max(1e-12, (1.0 - l0) * p_unigram + l0 * p_uniform)
        if self.order == 2:
            return max(1e-12, (l3 + l2) * p_bigram + l1 * p_unigram + l0 * p_uniform)
        return max(1e-12, l3 * p_trigram + l2 * p_bigram + l1 * p_unigram + l0 * p_uniform)

    def log_probability(self, token: Token, history: Sequence[Token] = ()) -> float:
        return math.log(self.probability(token, history))

    def perplexity(self, sequences: Iterable[Sequence[Token]]) -> float:
        total_log = 0.0
        count = 0
        for sequence in sequences:
            history: list[Token] = [BOS, BOS]
            for token in [*sequence, EOS]:
                total_log += self.log_probability(token, history)
                history.append(token)
                count += 1
        return math.exp(-total_log / count) if count else float("inf")

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "lambdas": list(self.lambdas),
            "trained_on": self.trained_on,
            "unigram": [[list(token), count] for token, count in sorted(self.unigram.items(), key=repr)],
            "bigram": [
                [list(previous), [[list(token), count] for token, count in sorted(counter.items(), key=repr)]]
                for previous, counter in sorted(self.bigram.items(), key=repr)
            ],
            "trigram": [
                [
                    [list(pair[0]), list(pair[1])],
                    [[list(token), count] for token, count in sorted(counter.items(), key=repr) if count >= 2],
                ]
                for pair, counter in sorted(self.trigram.items(), key=repr)
                if any(count >= 2 for count in counter.values())
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ChordNGram:
        model = cls(order=int(payload.get("order", 3)), lambdas=tuple(payload.get("lambdas", (0.45, 0.35, 0.19, 0.01))))
        model.trained_on = dict(payload.get("trained_on", {}))
        for token, count in payload.get("unigram", []):
            model.unigram[(int(token[0]), token[1])] = int(count)
            model.vocabulary.add((int(token[0]), token[1]))
        for previous, entries in payload.get("bigram", []):
            key = (int(previous[0]), previous[1])
            for token, count in entries:
                model.bigram[key][(int(token[0]), token[1])] = int(count)
        for pair, entries in payload.get("trigram", []):
            key = ((int(pair[0][0]), pair[0][1]), (int(pair[1][0]), pair[1][1]))
            for token, count in entries:
                model.trigram[key][(int(token[0]), token[1])] = int(count)
        model.vocabulary.discard(BOS)
        return model

    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()))
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> ChordNGram | None:
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text()))
        except (OSError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def corpus_sequences(*, download: bool = True, include_weimar: bool = True) -> tuple[list[list[Token]], dict[str, int]]:
    """Token sequences from both corpora, with provenance counts."""
    from .data import load_solos, solo_progression, treebank_progressions

    sequences: list[list[Token]] = []
    counts: dict[str, int] = {}
    tunes = treebank_progressions(download=download)
    sequences.extend(tokens_of(tune) for tune in tunes)
    counts["treebank_tunes"] = len(tunes)
    counts["treebank_chords"] = sum(len(tune.spans) for tune in tunes)
    if include_weimar:
        played = [solo_progression(solo) for solo in load_solos(download=download)]
        played = [progression for progression in played if progression.spans]
        sequences.extend(tokens_of(progression) for progression in played)
        counts["weimar_solos"] = len(played)
        counts["weimar_chords"] = sum(len(progression.spans) for progression in played)
    return sequences, counts


def split(sequences: Sequence[Sequence[Token]], *, seed: int = 17, fraction: float = 0.15):
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    cut = max(1, int(len(indices) * fraction))
    validation = {index for index in indices[:cut]}
    train = [sequences[i] for i in indices[cut:]]
    held_out = [sequences[i] for i in sorted(validation)]
    return train, held_out


_LAMBDA_GRID = (
    (0.80, 0.14, 0.05, 0.01),
    (0.70, 0.20, 0.09, 0.01),
    (0.60, 0.25, 0.14, 0.01),
    (0.45, 0.35, 0.19, 0.01),
    (0.35, 0.40, 0.24, 0.01),
    (0.25, 0.45, 0.29, 0.01),
    (0.15, 0.50, 0.34, 0.01),
)


def train(
    *,
    download: bool = True,
    include_weimar: bool = True,
    seed: int = 17,
) -> tuple[ChordNGram, dict[str, float]]:
    """Fit the model, choose the order and interpolation on held-out data."""
    sequences, counts = corpus_sequences(download=download, include_weimar=include_weimar)
    train_set, held_out = split(sequences, seed=seed)

    report: dict[str, float] = {}
    best: tuple[float, ChordNGram] | None = None
    for order in (1, 2, 3):
        for lambdas in _LAMBDA_GRID if order == 3 else (_LAMBDA_GRID[1],):
            model = ChordNGram(order=order, lambdas=lambdas)
            model.fit(train_set)
            score = model.perplexity(held_out)
            key = f"order{order}" + ("" if order < 3 else f"_l{lambdas[0]:.2f}")
            report[key] = score
            if best is None or score < best[0]:
                best = (score, model)
    assert best is not None
    report["best_perplexity"] = best[0]

    # Refit the winning configuration on everything: the split existed to pick
    # the configuration, not to throw away 15% of a small corpus.
    final = ChordNGram(order=best[1].order, lambdas=best[1].lambdas)
    final.fit(sequences)
    final.trained_on = counts
    return final, report


def main() -> None:
    model, report = train()
    path = model.save()
    print("=" * 72)
    print("JAZZ CHORD LANGUAGE MODEL")
    print("=" * 72)
    print("held-out perplexity (lower is better)")
    for key, value in sorted(report.items(), key=lambda item: item[1]):
        print(f"  {key:<24} {value:8.3f}")
    print(f"\nchosen: order {model.order}, lambdas {model.lambdas}")
    print(f"trained on: {model.trained_on}")
    print(f"vocabulary: {len(model.vocabulary)} key-relative chord types")
    print(f"saved: {path} ({path.stat().st_size / 1024:.0f} KB)")

    print("\nmost likely continuations of a ii7 (degree 2 min7):")
    context = (2, "min7")
    scored = sorted(
        ((model.probability(token, [context]), token) for token in model.vocabulary),
        reverse=True,
    )[:8]
    for probability, token in scored:
        print(f"  degree {token[0]:>2} {token[1]:<10} {probability:.3f}")


if __name__ == "__main__":
    main()
