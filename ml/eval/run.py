"""One-command evaluation: `python -m ml.eval.run`.

Scores every registered engine, plus the Bach oracle, on the identical held-out
split with identical metrics, prints a comparison table and writes
`ml/eval/REPORT.md`.
"""

from __future__ import annotations

import argparse
import importlib
import io
import pkgutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.corpus import load_chorales, split_chorales  # noqa: E402
from ml.engines.base import all_engines  # noqa: E402
from ml.eval.harness import (  # noqa: E402
    EngineResult,
    defect_table,
    evaluate_engine,
    key_detection_accuracy,
    reference_result,
    reference_style,
)
from ml.eval.metrics import DEFECT_KINDS, StyleCounts, summarize_distribution  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "REPORT.md"
ABLATION_PATH = Path(__file__).resolve().parents[1] / "models" / "ablation.json"


def discover_engines() -> None:
    """Import every engine module so the registry is populated."""
    package = importlib.import_module("ml.engines")
    for module in pkgutil.iter_modules(package.__path__, f"{package.__name__}."):
        leaf = module.name.rsplit(".", 1)[-1]
        if leaf in ("base",) or leaf.startswith("_"):
            continue
        try:
            importlib.import_module(module.name)
        except Exception as error:  # pragma: no cover - surfaced, never swallowed
            print(f"  ! could not import {module.name}: {error}")


def _fmt(value: float | None, places: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{places}f}"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    out = io.StringIO()
    out.write("| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |\n")
    out.write("|" + "|".join("-" * (w + 2) for w in widths) + "|\n")
    for row in rows:
        out.write("| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |\n")
    return out.getvalue()


def load_ablation() -> dict | None:
    if not ABLATION_PATH.exists():
        return None
    try:
        import json

        return json.loads(ABLATION_PATH.read_text())
    except (OSError, ValueError):
        return None


def build_report(
    results: Sequence[EngineResult],
    reference: StyleCounts,
    oracle: EngineResult,
    key_stats: dict,
    detected_results: Sequence[EngineResult] | None,
    split_sizes: tuple[int, int, int],
    ablation: dict | None = None,
) -> str:
    names = [result.engine_id for result in results]
    out = io.StringIO()
    out.write("# HarmonAIzer v2 — engine evaluation\n\n")
    out.write(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.\n\n")
    out.write(
        f"Bach chorale corpus, piece-level split by hash of the piece id: "
        f"{split_sizes[0]} train / {split_sizes[1]} val / {split_sizes[2]} test. "
        f"Every engine sees the same held-out sopranos and nothing else.\n\n"
    )

    out.write("## 1. Every engine against Bach, on the axes that matter\n\n")
    out.write(
        "**The headline is distance from Bach, not a defect leaderboard.** Voice-leading "
        "defects are a guardrail that stops an engine degenerating into parallel thirds; "
        "they are not the objective. Bach himself breaks these rules, and an engine that "
        "never does is not better than Bach — it is stiffer than Bach. Every column below "
        "is therefore read against the `bach_oracle` row, and *under*-shooting is as much "
        "a miss as overshooting.\n\n"
    )
    headline = [
        ("STRUCTURAL defects / piece", lambda r: r.defects.structural_rate(), 3),
        ("HARD defects /100 chords", lambda r: r.defects.hard_error_rate(), 2),
        ("SOFT defects /100 chords", lambda r: r.defects.soft_error_rate(), 1),
        ("chord-bigram JS from Bach", lambda r: r.divergences(reference)["chord_bigram_js"], 3),
        ("cadence JS from Bach", lambda r: r.divergences(reference)["cadence_js"], 3),
        ("distinct chords / piece", lambda r: r.activity.mean_classes_per_piece(), 1),
        ("share of beats on I or V", lambda r: 100 * r.activity.safe_chord_share(), 1),
        ("chord changes /100 beats", lambda r: r.activity.chord_changes_per_100_beats(), 1),
        ("voice moves /100 beats", lambda r: r.activity.sonority_changes_per_100_beats(), 1),
    ]
    rows = [[label] + [_fmt(fn(r), places) for r in results] for label, fn, places in headline]
    out.write(render_table(["metric"] + names, rows))
    out.write(
        "\nThe three defect rows are tiered by **audibility**, not by textbook tradition, "
        "and are never summed. STRUCTURAL is the category a listener notices instantly — a "
        "piece that never resolves, a phrase closing somewhere impossible — and it must "
        "stay near Bach's own ~0.02 whatever else an engine is doing. HARD is the classic "
        "audible errors. SOFT is real to a theorist and largely invisible to a listener; "
        "Bach breaks all of them, so drifting upward there in exchange for more interesting "
        "harmony is a fair trade.\n\n"
        "The bottom four rows are what a defect count cannot see at all. An engine reaches "
        "zero defects either by realising a full harmonic vocabulary carefully, or by "
        "narrowing the vocabulary until nothing can go wrong. Those look identical in the "
        "defect rows and completely different below them.\n\n"
    )

    out.write("## 2. Defects by tier\n\n")
    out.write(
        "Objective and engine-agnostic: counted by the same detectors for every row, "
        "including Bach's. Read as a guardrail — the question is whether an engine is "
        "*materially worse* than the oracle, not whether it is lowest.\n\n"
        "Structural defects are per PIECE; everything else is per 100 chord changes. The "
        "two units are never added together. `half_cadence_ending` is reported and not "
        "scored: Bach ends 9.2% of his chorales on a root-position V, so it is idiomatic — "
        "but an engine doing it half the time is broken, and this is the only way to see "
        "that.\n\n"
    )
    rows = []
    for kind, values, unit in defect_table(results):
        emphasise = kind in ("STRUCTURAL / piece", "HARD TOTAL", "SOFT TOTAL")
        label = f"**{kind}**" if emphasise else kind
        places = 3 if unit == "per piece" else (1 if unit == "% of pieces" else 2)
        suffix = "%" if unit == "% of pieces" else ""
        rows.append([label, unit] + [_fmt(v, places) + suffix for v in values])
    out.write(render_table(["defect", "unit"] + names, rows))
    out.write("\n")

    out.write("## 3. Style distance from the Bach corpus\n\n")
    out.write(
        "Jensen-Shannon divergence in bits (0 = identical, 1 = disjoint) against the "
        "**training** split. `bach_oracle` is held-out Bach measured against training "
        "Bach, so its value is the noise floor: no engine can meaningfully score below it.\n\n"
    )
    divergence_keys = list(results[0].divergences(reference).keys())
    rows = []
    for metric in divergence_keys:
        rows.append([metric] + [_fmt(result.divergences(reference)[metric], 3) for result in results])
    out.write(render_table(["JS divergence"] + names, rows))
    out.write("\n")

    out.write("### Style mix\n\n")
    fraction_keys = list(results[0].style_fractions().keys())
    rows = []
    for metric in fraction_keys:
        rows.append([metric] + [_fmt(100 * result.style_fractions()[metric], 1) + "%" for result in results])
    out.write(render_table(["share of chords"] + names, rows))
    out.write("\n")

    out.write("### Cadence types\n\n")
    cadence_keys = sorted({k for result in results for k in result.style.cadence})
    rows = []
    for cadence in cadence_keys:
        row = [cadence]
        for result in results:
            total = sum(result.style.cadence.values()) or 1
            row.append(_fmt(100 * result.style.cadence.get(cadence, 0) / total, 1) + "%")
        rows.append(row)
    out.write(render_table(["cadence"] + names, rows))
    out.write("\n")

    out.write("## 4. Held-out likelihood\n\n")
    out.write(
        "Negative log-likelihood in nats per predicted note token, and its perplexity, "
        "for Bach's own alto/tenor/bass on the held-out split. Only defined for "
        "probabilistic engines; a rule engine has no likelihood to report.\n\n"
    )
    rows = [
        ["NLL (nats/token)"] + [_fmt(result.nll_per_token(), 4) for result in results],
        ["perplexity"] + [_fmt(result.perplexity(), 3) for result in results],
    ]
    out.write(render_table(["metric"] + names, rows))
    out.write("\n")

    out.write("## 5. Agreement with Bach — reported, NOT the headline\n\n")
    out.write(
        "This is the metric v1 optimised. It is included for continuity and because "
        "watching it move independently of sections 1-3 is itself the argument against "
        "it: a harmonization can disagree with Bach on most beats and still be excellent, "
        "and v1's version of this number additionally counted padded positions.\n\n"
    )
    agreement_keys = list(results[0].agreement.as_dict().keys())
    rows = []
    for metric in agreement_keys:
        rows.append([metric] + [_fmt(100 * result.agreement.as_dict()[metric], 1) + "%" for result in results])
    out.write(render_table(["agreement"] + names, rows))
    out.write("\n")

    out.write("## 6. Cost and robustness\n\n")
    rows = [
        ["pieces scored"] + [str(result.pieces) for result in results],
        ["failures"] + [str(result.failures) for result in results],
        ["seconds / piece"] + [_fmt(result.seconds / max(1, result.pieces), 3) for result in results],
    ]
    out.write(render_table(["metric"] + names, rows))
    out.write("\n")

    out.write("## 7. Melody-only key detection\n\n")
    out.write(
        "The tables above supply the ground-truth key so the comparison isolates the "
        "harmonic decision. In production the engine must find the key from the tune "
        "alone; this is the accuracy of that step on the same held-out melodies.\n\n"
    )
    out.write(render_table(
        ["metric", "value"],
        [
            ["exact key (tonic + mode)", _fmt(100 * key_stats["exact"], 1) + "%"],
            ["correct tonic", _fmt(100 * key_stats["same_tonic"], 1) + "%"],
            ["relative-key confusion", _fmt(100 * key_stats["relative_key_confusion"], 1) + "%"],
        ],
    ))
    out.write("\n")

    if detected_results:
        out.write("### Same engines, key detected instead of supplied\n\n")
        detected_names = [result.engine_id for result in detected_results]
        rows = [
            ["HARD TOTAL / 100 chords"] + [_fmt(r.defects.hard_error_rate()) for r in detected_results],
            ["chord_bigram_js"] + [_fmt(r.divergences(reference)["chord_bigram_js"], 3) for r in detected_results],
            ["chord agreement"] + [_fmt(100 * r.agreement.as_dict()["chord_exact"], 1) + "%" for r in detected_results],
        ]
        out.write(render_table(["metric"] + detected_names, rows))
        out.write("\n")

    if ablation:
        out.write("## 8. Representation ablation — what the v1 handicap actually cost\n\n")
        out.write(
            "Identical architecture, identical data, identical number of gradient steps; "
            "only the pitch representation differs. `absolute` is exactly the information "
            "v1's network had: raw pitch plus a mode flag, no tonic. `absolute_augmented` "
            "adds per-epoch transposition, which is the standard remedy. Validation NLL is "
            "in nats per predicted note token, so lower is better.\n\n"
        )
        rows = []
        for name, values in ablation.items():
            rows.append([
                name,
                _fmt(values["val_loss"], 4),
                _fmt(values["val_perplexity"], 3),
                str(values.get("variants_per_piece", 1)),
                str(values["best_epoch"]),
                _fmt(values["seconds"] / 60.0, 1),
            ])
        out.write(render_table(
            ["representation", "val NLL/token", "perplexity", "transpositions/piece", "best epoch", "minutes"],
            rows,
        ))
        baseline = ablation.get("tonic_relative", {}).get("val_perplexity")
        if baseline:
            out.write("\nRelative to tonic-relative: ")
            deltas = [
                f"`{name}` {100 * (values['val_perplexity'] / baseline - 1):+.1f}% perplexity"
                for name, values in ablation.items() if name != "tonic_relative"
            ]
            out.write(", ".join(deltas) + ".\n\n")

    out.write("## 9. Most frequent chords\n\n")
    for result in list(results):
        top = summarize_distribution(result.style.chord_unigram, [], top=8)
        out.write(f"* `{result.engine_id}`: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in top) + "\n")
    out.write("\n")
    return out.getvalue()


def print_summary(results: Sequence[EngineResult], reference: StyleCounts) -> None:
    headers = ["engine", "hard/100", "par5", "par8", "cross", "range", "unresLT", "unres7", "bigramJS", "cadJS", "agree", "s/piece"]
    rows = []
    for result in results:
        divergences = result.divergences(reference)
        rows.append([
            result.engine_id,
            _fmt(result.defects.hard_error_rate()),
            _fmt(result.defects.per_hundred("parallel_fifths")),
            _fmt(result.defects.per_hundred("parallel_octaves")),
            _fmt(result.defects.per_hundred("voice_crossing")),
            _fmt(result.defects.per_hundred("range")),
            _fmt(result.defects.per_hundred("unresolved_leading_tone")),
            _fmt(result.defects.per_hundred("unresolved_seventh")),
            _fmt(divergences["chord_bigram_js"], 3),
            _fmt(divergences["cadence_js"], 3),
            _fmt(100 * result.agreement.as_dict()["chord_exact"], 1),
            _fmt(result.seconds / max(1, result.pieces), 3),
        ])
    print(render_table(headers, rows))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score every registered harmony engine on held-out Bach chorales.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N test pieces.")
    parser.add_argument("--engines", nargs="*", default=None, help="Restrict to these engine ids.")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Which held-out split to score.")
    parser.add_argument("--detect-key", action="store_true", help="Also run the realistic melody-only-key condition.")
    parser.add_argument("--no-report", action="store_true", help="Print the table but do not write REPORT.md.")
    args = parser.parse_args(argv)

    discover_engines()
    chorales = load_chorales()
    train, val, test = split_chorales(chorales)
    held_out = (test if args.split == "test" else val)
    if args.limit:
        held_out = held_out[: args.limit]

    print(f"Corpus: {len(chorales)} four-part chorales — {len(train)} train / {len(val)} val / {len(test)} test")
    print(f"Scoring on {len(held_out)} held-out pieces ({args.split} split)\n")

    engines = [e for e in all_engines() if e.is_available()]
    if args.engines:
        engines = [e for e in engines if e.id in args.engines]
    engines.sort(key=lambda e: (e.learned, e.id))

    reference = reference_style(train)
    results: list[EngineResult] = []
    for engine in engines:
        print(f"  running {engine.id} ...", flush=True)
        results.append(evaluate_engine(engine, held_out, supply_key=True, verbose=True))

    oracle = reference_result(held_out)
    results.append(oracle)

    print()
    print_summary(results, reference)

    key_stats = key_detection_accuracy(held_out)
    print(f"melody-only key detection: exact {100 * key_stats['exact']:.1f}%, "
          f"tonic {100 * key_stats['same_tonic']:.1f}%, "
          f"relative confusion {100 * key_stats['relative_key_confusion']:.1f}%\n")

    detected_results = None
    if args.detect_key:
        detected_results = []
        for engine in engines:
            print(f"  running {engine.id} (key detected) ...", flush=True)
            detected_results.append(evaluate_engine(engine, held_out, supply_key=False, score_likelihood=False))

    if not args.no_report:
        report = build_report(
            results, reference, oracle, key_stats, detected_results,
            (len(train), len(val), len(test)), load_ablation(),
        )
        existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
        # The file is part hand-written and part generated. Everything before
        # GENERATED is a hand-written orientation for a reader with no context,
        # and everything after NARRATIVE is the discussion; only the middle is
        # rebuilt. Without this the status section would be destroyed by the
        # next eval run, which is exactly when someone would be reading it.
        preamble_marker = "<!-- GENERATED -->\n"
        narrative_marker = "\n<!-- NARRATIVE -->\n"
        preamble = existing.split(preamble_marker, 1)[0] if preamble_marker in existing else ""
        narrative = existing.split(narrative_marker, 1)[1] if narrative_marker in existing else ""
        REPORT_PATH.write_text(preamble + preamble_marker + report + narrative_marker + narrative)
        print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
