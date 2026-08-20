"""The jazz oracle: score real jazz with our own metrics, before generating any.

This is the single most valuable thing in the chorale harness ported across —
not the metrics themselves, which are chorale-specific and wrong here, but the
discipline. Scoring the ground truth with your own measuring stick tells you
what the target actually looks like, and it is the only thing that stops you
optimising toward a number nobody wants.

Three questions it answers, each of which changed a decision in this package:

  1. *What does real jazz harmony measure?* — seventh rate, chromaticism,
     dominant-resolution rate, ii-V density on 1170 treebank lead sheets.
  2. *How often do real jazz melodies sit on an "avoid note"?* — 456 Weimar
     solos against the changes actually played under them. If real players
     score 0, a hard constraint is justified; if they score 8%, then a
     reharmonizer forced to 0 is more conservative than the music it imitates.
  3. *How far apart are two versions of the same tune?* — 139 standards appear
     in both corpora, so the distance between the lead sheet and what the band
     actually played is a measurement, not a guess. That is the distance band.

Run: `python -m ml.reharm.oracle`
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import (
    CACHE,
    LICENCES,
    chorus_progression,
    load_solos,
    solo_melody_beats,
    solo_progression,
    treebank_progressions,
)
from .metrics import (
    MelodyFit,
    SyntaxCounts,
    collect_corpus_syntax,
    collect_syntax,
    distance,
    melody_fit,
)

ORACLE_PATH = CACHE / "oracle.json"


@dataclass
class OracleReport:
    treebank: SyntaxCounts
    weimar: SyntaxCounts
    melody: MelodyFit
    version_distances: list[float]
    version_pc_distances: list[float]
    version_root_distances: list[float]
    tunes: int
    solos: int
    compared: int

    def distance_band(self, low: float = 0.25, high: float = 0.75) -> tuple[float, float]:
        """Quantile band of real version-to-version harmonic change."""
        if not self.version_distances:
            return (0.30, 0.70)
        values = sorted(self.version_distances)
        return (_quantile(values, low), _quantile(values, high))

    def as_dict(self) -> dict:
        band = self.distance_band()
        return {
            "licences": LICENCES,
            "counts": {"treebank_tunes": self.tunes, "weimar_solos": self.solos, "shared_tunes": self.compared},
            "treebank_syntax": self.treebank.as_dict(),
            "weimar_syntax": self.weimar.as_dict(),
            "weimar_melody_fit": self.melody.as_dict(),
            "version_distance": {
                "n": len(self.version_distances),
                "median_changed_rate": statistics.median(self.version_distances) if self.version_distances else 0.0,
                "p25_changed_rate": band[0],
                "p75_changed_rate": band[1],
                "median_pc_distance": (
                    statistics.median(self.version_pc_distances) if self.version_pc_distances else 0.0
                ),
                "median_root_change_rate": (
                    statistics.median(self.version_root_distances) if self.version_root_distances else 0.0
                ),
            },
            "top_qualities": _top(self.treebank.qualities, 10),
            "top_vocabulary": [
                {"degree": degree, "quality": quality, "share": share}
                for (degree, quality), share in _top(self.treebank.vocabulary, 14)
            ],
            "root_motion": _top(self.treebank.root_motion, 12),
        }


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    position = q * (len(values) - 1)
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _top(counter, n: int) -> list:
    total = sum(counter.values()) or 1
    return [(key, round(count / total, 4)) for key, count in counter.most_common(n)]


def build(
    *,
    download: bool = True,
    solo_limit: int | None = None,
    melody_limit: int | None = 120,
) -> OracleReport:
    tunes = treebank_progressions(download=download)
    treebank_counts = collect_corpus_syntax(tunes)

    solos = load_solos(download=download, limit=solo_limit)
    weimar_counts = SyntaxCounts()
    fit = MelodyFit()
    for index, solo in enumerate(solos):
        progression = solo_progression(solo)
        if not progression.spans:
            continue
        weimar_counts.merge(collect_syntax(progression))
        if melody_limit is None or index < melody_limit:
            melody = solo_melody_beats(solo)
            fit.merge(melody_fit(melody, progression, beats_per_bar=solo.meter[0]))

    by_title = {tune.title.strip().lower(): tune for tune in tunes}
    changed: list[float] = []
    pc_distances: list[float] = []
    root_distances: list[float] = []
    compared = 0
    for solo in solos:
        reference = by_title.get(solo.title.strip().lower())
        if reference is None:
            continue
        played = chorus_progression(solo, 1)
        if not played.spans or not reference.spans:
            continue
        ratio = played.duration / reference.duration if reference.duration else 0.0
        # Only compare when the two really are the same form: a solo taken at
        # double time, or a truncated chorus, would report a spurious rewrite.
        if not 0.8 <= ratio <= 1.25:
            continue
        metrics = distance(reference, played)
        changed.append(metrics.changed_rate)
        pc_distances.append(metrics.pc_distance)
        root_distances.append(metrics.root_change_rate)
        compared += 1

    return OracleReport(
        treebank=treebank_counts,
        weimar=weimar_counts,
        melody=fit,
        version_distances=changed,
        version_pc_distances=pc_distances,
        version_root_distances=root_distances,
        tunes=len(tunes),
        solos=len(solos),
        compared=compared,
    )


def load(path: Path = ORACLE_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def main() -> None:
    report = build()
    payload = report.as_dict()
    ORACLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ORACLE_PATH.write_text(json.dumps(payload, indent=2, default=str))

    def show(title: str, stats: dict[str, float], keys: Sequence[str]) -> None:
        print(f"\n{title}")
        for key in keys:
            print(f"  {key:<28} {stats[key]:.3f}")

    print("=" * 72)
    print("JAZZ ORACLE — real jazz scored by our own metrics")
    print("=" * 72)
    print(f"treebank tunes {report.tunes}   weimar solos {report.solos}   shared tunes compared {report.compared}")

    syntax_keys = [
        "chords", "mean_chord_beats", "seventh_rate", "extension_rate", "mean_extensions",
        "dominant_rate", "dominant_resolution_rate", "semitone_resolution_share",
        "ii_v_per_16_bars", "ii_v_i_share", "nondiatonic_root_rate", "chromatic_tone_rate",
        "ends_on_tonic_rate",
    ]
    show("Jazz Harmony Treebank (lead sheets, 1170 tunes)", report.treebank.as_dict(), syntax_keys)
    show("Weimar Jazz Database (changes as played)", report.weimar.as_dict(), syntax_keys)

    melody = report.melody.as_dict()
    print("\nReal jazz melody over real changes (Weimar solos)")
    for key in ("chord_tone_rate", "tension_rate", "soft_conflict_rate", "hard_conflict_rate"):
        print(f"  {key:<28} {melody[key]:.3f}")
    print(f"  {'notes measured':<28} {melody['notes']:.0f}")

    band = report.distance_band()
    print("\nLead sheet vs what the band played (same tune, both corpora)")
    print(f"  compared                     {report.compared}")
    if report.version_distances:
        print(f"  median changed_rate          {statistics.median(report.version_distances):.3f}")
        print(f"  p25..p75 changed_rate        {band[0]:.3f} .. {band[1]:.3f}")
        print(f"  median root_change_rate      {statistics.median(report.version_root_distances):.3f}")
        print(f"  median pc_distance           {statistics.median(report.version_pc_distances):.3f}")

    print("\nTop chord qualities (treebank)")
    for quality, share in payload["top_qualities"]:
        print(f"  {quality:<12} {share:.3f}")
    print("\nTop key-relative chords (treebank)")
    for entry in payload["top_vocabulary"]:
        print(f"  degree {entry['degree']:>2} {entry['quality']:<10} {entry['share']:.3f}")
    print(f"\nwritten: {ORACLE_PATH}")


if __name__ == "__main__":
    main()
