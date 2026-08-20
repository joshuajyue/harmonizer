"""The comparison: skeleton vs rule reharmonization vs sampling vs humans.

Four things are scored with identical code, which is the only way any of this
means anything:

  * **skeleton** — the rules engine's own diatonic harmony, unreharmonized. The
    control. If reharmonization does not beat this, there is no workstream.
  * **rules** — the hand-written substitution vocabulary, Viterbi argmax.
  * **sampled** — the same lattice, learned scores, drawn at a temperature.
  * **human** — for Weimar tunes, the changes a real rhythm section played
    under that exact melody. Not a proxy, not a taste judgement: the reference.

The diversity number is the one that tests the strategic argument directly. If
five samples of the same tune are five identical progressions, then sampling
buys nothing over search and the argument is wrong.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field

from .data import Progression, treebank_progressions
from .melodies import TestTune, jazz_tunes, traditional_tunes
from .metrics import (
    collect_corpus_syntax,
    collect_syntax,
    distance,
    js_divergence,
    score,
)
from .model import ChordNGram
from .search import (
    HybridScorer,
    ModelScorer,
    ReharmConfig,
    RuleScorer,
    build_lattice,
    realize,
    sample,
    viterbi,
)
from .skeleton import skeleton_from_rules

REPORTED = (
    "headline",
    "hard_conflict_rate",
    "soft_conflict_rate",
    "chord_tone_rate",
    "tension_rate",
    "seventh_rate",
    "extension_rate",
    "dominant_resolution_rate",
    "ii_v_per_16_bars",
    "chromatic_tone_rate",
    "mean_chord_beats",
    "root_change_rate",
    "changed_rate",
    "pc_distance",
    "human_pc_distance",
)


@dataclass
class Row:
    """One engine's scores across the evaluation set."""

    name: str
    values: dict[str, list[float]] = field(default_factory=dict)
    diversity: list[float] = field(default_factory=list)
    style_divergence: list[float] = field(default_factory=list)

    def add(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            self.values.setdefault(key, []).append(value)

    def mean(self, key: str) -> float:
        values = self.values.get(key) or []
        return statistics.fmean(values) if values else float("nan")

    def summary(self) -> dict[str, float]:
        out = {key: self.mean(key) for key in REPORTED if key in self.values}
        if self.diversity:
            out["sample_diversity"] = statistics.fmean(self.diversity)
        if self.style_divergence:
            out["style_divergence"] = statistics.fmean(self.style_divergence)
        return out


def style_reference(*, download: bool = True) -> dict:
    """Key-relative chord distribution of the treebank — the style target."""
    return collect_corpus_syntax(treebank_progressions(download=download)).vocabulary


def vocabulary_divergence(progression: Progression, reference) -> float:
    """How far a result's chord vocabulary sits from real jazz, in bits."""
    return js_divergence(collect_syntax(progression).vocabulary, reference)


def _metrics(tune: TestTune, base: Progression, result: Progression, reference) -> dict[str, float]:
    from .skeleton import melody_notes

    scored = score(melody_notes(tune.melody), base, result)
    values = scored.as_dict()
    values["style_divergence"] = vocabulary_divergence(result, reference)
    if tune.reference is not None and tune.reference.spans:
        # Distance to what a human actually played under this melody. The one
        # criterion in the table that no engine here optimises for, and
        # therefore the one that cannot be gamed by whichever engine happens to
        # share an objective with the metric.
        values["human_pc_distance"] = distance(tune.reference, result).pc_distance
    return values


@dataclass
class Engines:
    model: ChordNGram
    config: ReharmConfig
    samples: int = 5
    seed: int = 7


def evaluate(
    tunes: Sequence[TestTune],
    engines: Engines,
    *,
    reference_vocabulary=None,
    download: bool = True,
    include_human: bool = True,
) -> dict[str, Row]:
    reference = reference_vocabulary if reference_vocabulary is not None else style_reference(download=download)
    rows: dict[str, Row] = {
        "skeleton": Row("skeleton"),
        "rules": Row("rules"),
        "sampled": Row("sampled"),
        "hybrid": Row("hybrid"),
    }
    if include_human:
        rows["human"] = Row("human")

    for tune in tunes:
        skeleton = skeleton_from_rules(tune.melody)
        if not skeleton.units:
            continue
        base = skeleton.progression()
        lattice = build_lattice(skeleton, engines.config)

        metrics = _metrics(tune, base, base, reference)
        rows["skeleton"].add(metrics)
        rows["skeleton"].style_divergence.append(metrics["style_divergence"])

        rule_result = realize(lattice, viterbi(lattice, RuleScorer(lattice, engines.config)), skeleton)
        metrics = _metrics(tune, base, rule_result.progression(), reference)
        rows["rules"].add(metrics)
        rows["rules"].style_divergence.append(metrics["style_divergence"])

        for name, scorer in (
            ("sampled", ModelScorer(lattice, engines.model, engines.config)),
            ("hybrid", HybridScorer(lattice, engines.model, engines.config)),
        ):
            sampled: list[Progression] = []
            per_sample: list[dict[str, float]] = []
            for index in range(engines.samples):
                path = sample(
                    lattice,
                    scorer,
                    temperature=engines.config.temperature,
                    top_p=engines.config.top_p,
                    seed=engines.seed + index,
                )
                result = realize(lattice, path, skeleton).progression()
                sampled.append(result)
                per_sample.append(_metrics(tune, base, result, reference))
            # One row entry per TUNE, averaged over its samples, so every row in
            # the table is indexed the same way and a paired per-tune
            # comparison between engines is possible at all.
            averaged = {
                key: statistics.fmean([metrics[key] for metrics in per_sample if key in metrics])
                for key in per_sample[0]
            }
            rows[name].add(averaged)
            rows[name].style_divergence.append(averaged["style_divergence"])
            rows[name].diversity.append(_diversity(sampled))

        if include_human and tune.reference is not None and tune.reference.spans:
            metrics = _metrics(tune, base, tune.reference, reference)
            rows["human"].add(metrics)
            rows["human"].style_divergence.append(metrics["style_divergence"])

    return rows


def _diversity(progressions: Sequence[Progression]) -> float:
    """Mean pairwise root-change rate between samples of the same tune."""
    if len(progressions) < 2:
        return 0.0
    scores = [
        distance(progressions[i], progressions[j]).root_change_rate
        for i in range(len(progressions))
        for j in range(i + 1, len(progressions))
    ]
    return statistics.fmean(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def paired(rows: dict[str, Row], a: str, b: str, keys: Sequence[str]) -> dict[str, tuple[float, float, int]]:
    """Per-tune paired comparison of two engines.

    Means across a 20-tune set hide whether a difference is systematic or one
    tune shouting. Pairing by tune and counting wins does not.
    """
    out: dict[str, tuple[float, float, int]] = {}
    for key in keys:
        left = rows[a].values.get(key) or []
        right = rows[b].values.get(key) or []
        if not left or len(left) != len(right):
            continue
        differences = [x - y for x, y in zip(left, right)]
        wins = sum(1 for difference in differences if difference > 1e-9)
        out[key] = (statistics.fmean(differences), statistics.pstdev(differences), wins)
    return out


def print_paired(rows: dict[str, Row], a: str, b: str, keys: Sequence[str]) -> None:
    results = paired(rows, a, b, keys)
    if not results:
        return
    total = len(rows[a].values.get(next(iter(results)), []))
    print(f"\npaired per tune: {a} minus {b}  (n = {total})")
    print(f"{'metric':<26} {'mean diff':>10} {'sd':>8} {'wins':>6}")
    for key, (mean, deviation, wins) in results.items():
        print(f"{key:<26} {mean:>10.3f} {deviation:>8.3f} {wins:>4}/{total}")


def print_table(rows: dict[str, Row], *, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    names = list(rows)
    keys = [key for key in REPORTED if any(key in rows[name].values for name in names)]
    keys += ["sample_diversity", "style_divergence"]
    width = max(len(key) for key in keys) + 2
    header = " ".join(f"{name:>12}" for name in names)
    print(f"{'metric':<{width}} {header}")
    print("-" * (width + len(header) + 1))
    summaries = {name: rows[name].summary() for name in names}
    for key in keys:
        cells = []
        for name in names:
            value = summaries[name].get(key)
            cells.append(f"{value:>12.3f}" if value is not None else f"{'-':>12}")
        print(f"{key:<{width}} {' '.join(cells)}")


def build_tunes(kind: str, *, limit: int, download: bool) -> list[TestTune]:
    if kind == "traditional":
        return traditional_tunes()
    if kind == "jazz":
        return jazz_tunes(limit=limit, download=download)
    return traditional_tunes() + jazz_tunes(limit=limit, download=download)


def sweep(
    tunes: Sequence[TestTune],
    model: ChordNGram,
    *,
    parameter: str,
    values: Sequence[float],
    samples: int,
    download: bool,
) -> None:
    """How a dial actually moves the music. The point of having a dial."""
    reference = style_reference(download=download)
    print(f"\nsweep over {parameter}")
    print(f"{parameter:>10} {'engine':>10} {'headline':>9} {'hard':>7} {'rootchg':>8} {'chrom':>7} {'res':>6} {'div':>6} {'style':>7}")
    for value in values:
        config = ReharmConfig(**{parameter: value})
        rows = evaluate(
            tunes,
            Engines(model=model, config=config, samples=samples),
            reference_vocabulary=reference,
            download=download,
            include_human=False,
        )
        for name in ("rules", "sampled", "hybrid"):
            summary = rows[name].summary()
            print(
                f"{value:>10.2f} {name:>10} {summary.get('headline', float('nan')):>9.3f} "
                f"{summary.get('hard_conflict_rate', float('nan')):>7.3f} "
                f"{summary.get('root_change_rate', float('nan')):>8.3f} "
                f"{summary.get('chromatic_tone_rate', float('nan')):>7.3f} "
                f"{summary.get('dominant_resolution_rate', float('nan')):>6.3f} "
                f"{summary.get('sample_diversity', 0.0):>6.3f} "
                f"{summary.get('style_divergence', float('nan')):>7.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare jazz reharmonization engines.")
    parser.add_argument("--cases", choices=("traditional", "jazz", "both"), default="both")
    parser.add_argument("--limit", type=int, default=20, help="jazz tunes to evaluate")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--adventure", type=float, default=None)
    parser.add_argument(
        "--sweep", choices=("adventure", "temperature", "top_p", "rule_weight"), default=None
    )
    parser.add_argument("--offline", action="store_true", help="use the cache, never download")
    parser.add_argument("--json", type=str, default=None, help="write the summary to a JSON file")
    args = parser.parse_args()

    download = not args.offline
    model = ChordNGram.load()
    if model is None:
        raise SystemExit("no model: run `python -m ml.reharm.model` first")

    tunes = build_tunes(args.cases, limit=args.limit, download=download)
    overrides: dict[str, float] = {}
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.adventure is not None:
        overrides["adventure"] = args.adventure
    config = ReharmConfig(**overrides)

    if args.sweep:
        values = {
            "adventure": (0.0, 0.25, 0.5, 0.75, 1.0),
            "temperature": (0.05, 0.3, 0.6, 0.9, 1.3, 1.8),
            "top_p": (0.5, 0.75, 0.9, 0.98, 1.0),
            "rule_weight": (0.0, 0.5, 0.9, 1.5, 2.5, 4.0),
        }[args.sweep]
        sweep(tunes, model, parameter=args.sweep, values=values, samples=args.samples, download=download)
        return

    rows = evaluate(tunes, Engines(model=model, config=config, samples=args.samples), download=download)
    print(f"tunes: {len(tunes)} ({args.cases})   samples per tune: {args.samples}")
    print(f"config: adventure={config.adventure} temperature={config.temperature} top_p={config.top_p}")
    print_table(rows, title="jazz reharmonization comparison")
    paired_keys = ("headline", "hard_conflict_rate", "chromatic_tone_rate", "human_pc_distance", "style_divergence")
    print_paired(rows, "hybrid", "rules", paired_keys)
    print_paired(rows, "sampled", "rules", paired_keys)

    if args.json:
        payload = {name: row.summary() for name, row in rows.items()}
        with open(args.json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
