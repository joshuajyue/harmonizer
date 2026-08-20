"""Tune the rule engine's harmonic weights on the validation split.

The eval harness is the objective function. That is the whole point of building
it first: v1 had no way to tell whether a change helped, so its rule engine's
numbers were whatever the author happened to type, and its model's numbers were
measured against a proxy that could not distinguish good from bad.

Coordinate descent over a dozen scalars on ~24 validation chorales. Deliberately
coarse: the search is a few passes over a small grid, not an optimiser, because
the objective is measured on a few hundred cadences and squeezing it harder
would fit the validation split rather than the style.

    python -m ml.training.tune_rules [--pieces 24] [--passes 2]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields, replace
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.corpus import Chorale, load_chorales, split_chorales  # noqa: E402
from ml.data.melody import chorale_to_melody, voices_to_grid  # noqa: E402
from ml.engines.rules import DEFAULT_CONFIG, RuleConfig, RuleHarmonyEngine  # noqa: E402
from ml.eval.harness import reference_style  # noqa: E402
from ml.eval.metrics import (  # noqa: E402
    DefectCounts,
    StyleCounts,
    collect_defects,
    collect_style,
    js_divergence,
)

#: Candidate values per parameter. Coarse on purpose.
GRID: dict[str, list[float]] = {
    "chord_tone_bonus": [1.6, 2.2, 2.8],
    "seventh_penalty": [-0.7, -0.45, -0.2, 0.05],
    "applied_prior": [-1.8, -1.4, -1.0, -0.6],
    "first_inversion_prior": [-0.9, -0.66, -0.4, -0.15],
    "six_four_prior": [-3.4, -2.9, -2.4],
    "weak_beat_64_penalty": [-3.2, -2.6, -1.8],
    "cadence_tonic_bonus": [1.2, 2.0, 2.8],
    "cadence_dominant_bonus": [0.6, 1.2, 1.8],
    "cadence_inversion_penalty": [-2.2, -1.5, -0.8],
    "pre_cadence_dominant_bonus": [0.4, 0.9, 1.5],
    "root_position_dominant_bonus": [0.4, 0.8, 1.4, 2.0],
    "harmony_weight": [0.9, 1.35, 1.9],
}

#: Objective weights. Voice-leading errors and style distance only: chord
#: agreement with Bach is deliberately excluded, because optimising it is
#: exactly the mistake v1 made.
WEIGHTS = {
    "hard": 3.0,
    "tendency": 0.35,
    "chord_bigram": 8.0,
    "root_motion": 4.0,
    "cadence": 4.0,
    "inversion": 4.0,
    "melodic_interval": 3.0,
    "outer_motion": 3.0,
}


def objective(config: RuleConfig, pieces: Sequence[Chorale], reference: StyleCounts) -> tuple[float, dict]:
    engine = RuleHarmonyEngine(config=config)
    defects, style = DefectCounts(), StyleCounts()
    for chorale in pieces:
        harmonization = engine.harmonize(chorale_to_melody(chorale))
        lines = voices_to_grid(harmonization.voices, length=chorale.length)[:4]
        while len(lines) < 4:
            lines.append([-1] * chorale.length)
        defects.merge(collect_defects(lines, chorale.key))
        style.merge(collect_style(lines, chorale.key, phrase_ends=chorale.fermatas))

    divergences = {
        name: js_divergence(getattr(style, name), getattr(reference, name))
        for name in ("chord_bigram", "root_motion", "cadence", "inversion", "melodic_interval", "outer_motion")
    }
    tendency = defects.per_hundred("unresolved_leading_tone") + defects.per_hundred("unresolved_seventh")
    total = WEIGHTS["hard"] * defects.hard_error_rate() + WEIGHTS["tendency"] * tendency
    for name, value in divergences.items():
        total += WEIGHTS[name] * value
    detail = {"hard": defects.hard_error_rate(), "tendency": tendency, **divergences}
    return total, detail


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", type=int, default=24)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "engines" / "_rule_config.json"))
    args = parser.parse_args(argv)

    chorales = load_chorales()
    train, val, _ = split_chorales(chorales)
    pieces = val[: args.pieces]
    reference = reference_style(train)
    print(f"tuning on {len(pieces)} validation chorales against {len(train)} training chorales\n")

    best = DEFAULT_CONFIG
    best_score, detail = objective(best, pieces, reference)
    print(f"start  J={best_score:.4f}  {json.dumps({k: round(v, 4) for k, v in detail.items()})}")

    for sweep in range(args.passes):
        improved = False
        for name, values in GRID.items():
            current = getattr(best, name)
            for value in values:
                if value == current:
                    continue
                candidate = replace(best, **{name: value})
                score, detail = objective(candidate, pieces, reference)
                if score < best_score - 1e-4:
                    best, best_score, improved = candidate, score, True
                    print(f"  pass {sweep}: {name} {current} -> {value}   J={score:.4f}")
                    current = value
        if not improved:
            print(f"  pass {sweep}: no improvement, stopping")
            break

    final_score, detail = objective(best, pieces, reference)
    print(f"\nfinal  J={final_score:.4f}  {json.dumps({k: round(v, 4) for k, v in detail.items()})}")
    payload = {field.name: getattr(best, field.name) for field in fields(best)}
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.out}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
