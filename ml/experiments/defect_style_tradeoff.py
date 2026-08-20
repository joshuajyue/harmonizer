"""Where to sit on the defect/style trade-off, measured rather than assumed.

The project owner's correction, and the reason this exists: a defect rate of
zero is not the goal. Bach himself breaks these rules ~3.7 times per 100 chord
changes, and an engine that never does is not better than Bach, it is stiffer
than Bach. Defects are a guardrail against degenerating into parallel thirds;
stylistic fidelity and harmonic interest are the objective.

`neural_vl` exposes exactly one knob for this: `rule_weight` scales the
voice-leading rulebook against the model's own log probabilities during the
constrained polish. At 0 the model is unconstrained; as it rises the rules take
over. This sweeps it and reports, at each setting, the defect rate against
Bach's, and the style divergences against the training corpus, so the operating
point can be chosen on evidence.

    python -m ml.experiments.defect_style_tradeoff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.corpus import load_chorales, split_chorales  # noqa: E402
from ml.data.melody import chorale_to_melody, voices_to_grid  # noqa: E402
from ml.engines.neural import ConstrainedNeuralEngine  # noqa: E402
from ml.eval.harness import reference_style  # noqa: E402
from ml.eval.metrics import (  # noqa: E402
    ActivityCounts,
    DefectCounts,
    StyleCounts,
    collect_activity,
    collect_defects,
    collect_style,
    js_divergence,
)

OUTPUT = Path(__file__).resolve().parents[1] / "models" / "defect_style_tradeoff.json"
WEIGHTS = (0.0, 0.15, 0.3, 0.5, 1.0, 2.0)


def score(engine, pieces, reference) -> dict:
    defects, style, activity = DefectCounts(), StyleCounts(), ActivityCounts()
    for chorale in pieces:
        result = engine.harmonize(chorale_to_melody(chorale), voice_count=4, seed=0)
        lines = voices_to_grid(result.voices, length=chorale.length)[:4]
        while len(lines) < 4:
            lines.append([-1] * chorale.length)
        defects.merge(collect_defects(lines, chorale.key))
        style.merge(collect_style(lines, chorale.key, phrase_ends=chorale.fermatas))
        activity.merge(collect_activity(lines, chorale.key))
    return {
        "hard_errors": round(defects.hard_error_rate(), 3),
        "parallel_fifths": round(defects.per_hundred("parallel_fifths"), 3),
        "parallel_octaves": round(defects.per_hundred("parallel_octaves"), 3),
        "chord_bigram_js": round(js_divergence(style.chord_bigram, reference.chord_bigram), 4),
        "cadence_js": round(js_divergence(style.cadence, reference.cadence), 4),
        "chord_unigram_js": round(js_divergence(style.chord_unigram, reference.chord_unigram), 4),
        "chord_changes_per_100_beats": round(activity.chord_changes_per_100_beats(), 1),
        "sonority_changes_per_100_beats": round(activity.sonority_changes_per_100_beats(), 1),
        "chord_classes_per_piece": round(activity.mean_classes_per_piece(), 2),
        "tonic_dominant_share": round(activity.safe_chord_share(), 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pieces", type=int, default=24)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    args = parser.parse_args(argv)

    chorales = load_chorales()
    train, val, test = split_chorales(chorales)
    pieces = (val if args.split == "val" else test)[: args.pieces]
    reference = reference_style(train)

    # Bach on the same pieces: the calibration every row is read against.
    defects, style, activity = DefectCounts(), StyleCounts(), ActivityCounts()
    for chorale in pieces:
        defects.merge(collect_defects(chorale.voices, chorale.key))
        style.merge(collect_style(chorale.voices, chorale.key, phrase_ends=chorale.fermatas))
        activity.merge(collect_activity(chorale.voices, chorale.key))
    oracle = {
        "hard_errors": round(defects.hard_error_rate(), 3),
        "chord_bigram_js": round(js_divergence(style.chord_bigram, reference.chord_bigram), 4),
        "cadence_js": round(js_divergence(style.cadence, reference.cadence), 4),
        "chord_classes_per_piece": round(activity.mean_classes_per_piece(), 2),
        "tonic_dominant_share": round(activity.safe_chord_share(), 4),
        "sonority_changes_per_100_beats": round(activity.sonority_changes_per_100_beats(), 1),
    }
    print(f"BACH (oracle, {len(pieces)} {args.split} pieces): {json.dumps(oracle)}\n")

    results = {}
    for weight in WEIGHTS:
        engine = ConstrainedNeuralEngine(rule_weight=weight)
        row = score(engine, pieces, reference)
        results[str(weight)] = row
        print(f"rule_weight={weight:<5} hard={row['hard_errors']:7.2f} "
              f"(p5={row['parallel_fifths']:5.2f} p8={row['parallel_octaves']:5.2f})  "
              f"bigramJS={row['chord_bigram_js']:.3f} cadJS={row['cadence_js']:.3f}  "
              f"classes/pc={row['chord_classes_per_piece']:5.2f} I+V={100*row['tonic_dominant_share']:.1f}%")

    OUTPUT.write_text(json.dumps({"oracle": oracle, "sweep": results}, indent=2) + "\n")
    print(f"\nwrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
