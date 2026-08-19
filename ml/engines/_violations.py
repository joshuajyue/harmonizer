"""Shared conversion from detected defects to the contract's `Violation` type.

Every engine reported violations with its own near-identical copy of this loop,
which is exactly how the three would drift apart on which defects reach the UI.

Two rules encoded here that the copies did not have:

* Structural defects are surfaced regardless of severity. They are the category a
  listener notices instantly — a piece that never resolves, a phrase that closes
  somewhere impossible — and they were previously invisible because every
  detector in `voicing.py` looks at two sonorities at a time and none of them can
  see the shape of a whole piece.
* `half_cadence_ending` is reported at info level rather than filtered out. Bach
  ends 9.2% of his chorales that way, so it is not a defect; but an engine doing
  it half the time is broken, and the only way to notice is to surface it.
"""

from __future__ import annotations

from typing import Sequence

from contracts.schema import Violation

from ..data.corpus import REST, STEP, STEPS_PER_QUARTER
from ..theory.chords import analyze_chord
from ..theory.pitch import Key
from ..theory.structure import (
    DESCRIPTIVE_KINDS,
    STRUCTURAL_KINDS,
    find_structural_defects,
    phrase_end_beats,
)
from ..theory.voicing import VOICE_NAMES, analyze_texture, texture_from_voices

#: Kinds that reach the UI even at info severity.
ALWAYS_REPORT = frozenset(STRUCTURAL_KINDS) | frozenset(DESCRIPTIVE_KINDS)


def build_violations(
    lines: Sequence[Sequence[int]],
    key: Key,
    *,
    steps_per_beat: int = STEPS_PER_QUARTER,
    phrase_ends: Sequence[bool] | None = None,
) -> list[Violation]:
    """Every violation an engine should report for a finished harmonization."""
    texture = texture_from_voices(
        [[None if pitch == REST else pitch for pitch in line] for line in lines], step=STEP
    )

    per_beat = []
    length = max((len(line) for line in lines), default=0)
    for start in range(0, length, steps_per_beat):
        pitches = [line[start] for line in lines if start < len(line) and line[start] != REST]
        per_beat.append(analyze_chord(pitches, key) if pitches else None)

    per_step = [
        per_beat[min(t // steps_per_beat, len(per_beat) - 1)] if per_beat else None
        for t in range(length)
    ]

    defects = list(analyze_texture(texture, key, per_step))
    beats = phrase_end_beats(phrase_ends, steps_per_beat) if phrase_ends is not None else None
    defects += find_structural_defects(per_beat, key, phrase_ends=beats)

    return [
        Violation(
            kind=defect.kind,
            severity=defect.severity,
            start=defect.offset,
            voices=[VOICE_NAMES[v] for v in defect.voices if v < 4],
            message=defect.message,
        )
        for defect in defects
        if defect.severity != "info" or defect.kind in ALWAYS_REPORT
    ]
