"""Fast iteration helper: score one engine on N held-out pieces."""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.data.corpus import load_chorales, split_chorales
from ml.data.melody import chorale_to_melody, voices_to_grid
from ml.eval.metrics import DEFECT_KINDS, AgreementCounts, DefectCounts, StyleCounts, collect_agreement, collect_defects, collect_style, js_divergence
from ml.eval.harness import reference_style


def score(engine, pieces, reference, label=""):
    d, a, s = DefectCounts(), AgreementCounts(), StyleCounts()
    t0 = time.perf_counter()
    for c in pieces:
        h = engine.harmonize(chorale_to_melody(c))
        lines = voices_to_grid(h.voices, length=c.length)[:4]
        while len(lines) < 4:
            lines.append([-1] * c.length)
        d.merge(collect_defects(lines, c.key))
        a.merge(collect_agreement(lines, c.voices, c.key))
        s.merge(collect_style(lines, c.key, phrase_ends=c.fermatas))
    dt = time.perf_counter() - t0
    hard = d.hard_error_rate()
    div = {k: js_divergence(getattr(s, k), getattr(reference, k)) for k in
           ("chord_bigram", "root_motion", "cadence", "inversion", "melodic_interval", "outer_motion")}
    print(f"[{label or engine.id}] hard={hard:.2f} p5={d.per_hundred('parallel_fifths'):.2f} "
          f"p8={d.per_hundred('parallel_octaves'):.2f} rng={d.per_hundred('range'):.2f} "
          f"cross={d.per_hundred('voice_crossing'):.2f} "
          f"uLT={d.per_hundred('unresolved_leading_tone'):.2f} u7={d.per_hundred('unresolved_seventh'):.2f} "
          f"| bigramJS={div['chord_bigram']:.3f} rootJS={div['root_motion']:.3f} cadJS={div['cadence']:.3f} "
          f"invJS={div['inversion']:.3f} melJS={div['melodic_interval']:.3f} motJS={div['outer_motion']:.3f} "
          f"| agree={100*a.as_dict()['chord_root']:.1f}/{100*a.as_dict()['chord_exact']:.1f} "
          f"| changes={d.chord_changes} | {dt/max(1,len(pieces)):.3f}s/pc")
    return d, a, s
