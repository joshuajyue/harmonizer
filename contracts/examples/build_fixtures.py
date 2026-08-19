"""Builds the canonical fixtures in contracts/examples/.

Three agents need example data: the frontend for its offline mock mode, the backend
for endpoint tests, the ML layer for engine smoke tests. If each invents its own,
they will be subtly incompatible and only find out at integration. So the fixtures
are generated here, through the Pydantic models, which makes them schema-valid by
construction rather than by hope.

The response fixture deliberately contains real voice-leading defects (parallel
fifths, parallel octaves, a voice crossing) so the frontend can develop and
demonstrate violation rendering without waiting on a working engine.

Regenerate:  python contracts/examples/build_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.schema import (  # noqa: E402
    Chord,
    HarmonizeRequest,
    HarmonizeResponse,
    KeySignature,
    Melody,
    Note,
    TimeSignature,
    Violation,
    Voice,
)

OUT_DIR = Path(__file__).resolve().parent

# An original 8-bar diatonic melody in C major, 4/4, with a clear final cadence.
# (start_beat, duration_beats, midi_pitch)
MELODY: list[tuple[float, float, int]] = [
    (0, 2, 60), (2, 2, 64),      # m1  C4  E4
    (4, 2, 67), (6, 2, 64),      # m2  G4  E4
    (8, 1, 65), (9, 1, 64), (10, 2, 62),   # m3  F4 E4 D4
    (12, 4, 60),                 # m4  C4
    (16, 2, 64), (18, 2, 67),    # m5  E4  G4
    (20, 2, 69), (22, 2, 67),    # m6  A4  G4
    (24, 1, 65), (25, 1, 64), (26, 2, 62), # m7  F4 E4 D4
    (28, 4, 60),                 # m8  C4
]

# Hand-voiced SATB. Columns: start, duration, roman, root_pc, quality, inversion,
# then the four voices as (soprano, alto, tenor, bass) MIDI pitches.
#
# Voicings are deliberately varied between segments. An earlier draft reused the same
# chord shapes, which manufactured five accidental parallels on top of the three
# intended ones — verified against ml.theory.voicing, which caught all eight. The
# fixture is only useful as a detector test corpus if its declared violations are
# exhaustive, so test_fixtures.py now asserts detector output matches exactly.
#
# Voice ranges: S 60-79, A 55-74, T 48-67, B 40-60.
PROGRESSION: list[tuple] = [
    (0,  2, "I",   0, "maj", 0, (60, 55, 52, 48)),
    (2,  2, "I",   0, "maj", 0, (64, 55, 52, 48)),
    (4,  2, "V6",  7, "maj", 1, (67, 55, 50, 47)),
    (6,  2, "I",   0, "maj", 0, (64, 55, 52, 48)),
    # -- parallel fifths, tenor/bass, at beat 9 (F2+C3 -> C3+G3, both +7) --
    (8,  1, "IV",  5, "maj", 0, (65, 57, 48, 41)),
    (9,  1, "I",   0, "maj", 0, (64, 60, 55, 48)),
    (10, 2, "V",   7, "maj", 0, (62, 59, 55, 43)),
    (12, 4, "I",   0, "maj", 0, (60, 55, 52, 48)),
    (16, 2, "I",   0, "maj", 0, (64, 55, 52, 48)),
    (18, 2, "V6",  7, "maj", 1, (67, 55, 50, 47)),
    # -- parallel octaves, soprano/bass, at beat 22 (A4+A2 -> G4+G2, both -2) --
    (20, 2, "IV6", 5, "maj", 1, (69, 60, 53, 45)),
    (22, 2, "V",   7, "maj", 0, (67, 59, 50, 43)),
    (24, 1, "IV6", 5, "maj", 1, (65, 60, 53, 45)),
    # -- voice crossing: tenor C4 (60) rises above alto G3 (55) at beat 25 --
    (25, 1, "I",   0, "maj", 0, (64, 55, 60, 48)),
    (26, 2, "V6",  7, "maj", 1, (62, 55, 50, 47)),
    (28, 4, "I",   0, "maj", 0, (60, 55, 52, 48)),
]

DEFECTS = [
    Violation(
        kind="parallel_fifths",
        severity="error",
        start=9.0,
        voices=["tenor", "bass"],
        message="Bass and tenor move F2-C3 to C3-G3, a perfect fifth in parallel motion.",
    ),
    Violation(
        kind="parallel_octaves",
        severity="error",
        start=22.0,
        voices=["soprano", "bass"],
        message="Soprano and bass move A4-A2 to G4-G2, an octave in parallel motion.",
    ),
    Violation(
        kind="voice_crossing",
        severity="warning",
        start=25.0,
        voices=["alto", "tenor"],
        message="Tenor (C4) rises above alto (G3).",
    ),
]

VOICE_ORDER = ["soprano", "alto", "tenor", "bass"]


def build_melody() -> Melody:
    return Melody(
        notes=[Note(pitch=p, start=s, duration=d, velocity=80) for s, d, p in MELODY],
        tempo=88.0,
        timeSignature=TimeSignature(numerator=4, denominator=4),
        key=KeySignature(tonic=0, mode="major", confidence=0.97),
    )


def build_response() -> HarmonizeResponse:
    chords = [
        Chord(
            start=start,
            duration=duration,
            roman=roman,
            root=root,
            quality=quality,
            inversion=inversion,
        )
        for start, duration, roman, root, quality, inversion, _ in PROGRESSION
    ]

    voices = []
    for index, name in enumerate(VOICE_ORDER):
        notes = [
            Note(pitch=pitches[index], start=start, duration=duration, velocity=76)
            for start, duration, _, _, _, _, pitches in PROGRESSION
        ]
        voices.append(Voice(name=name, notes=notes))

    return HarmonizeResponse(
        key=KeySignature(tonic=0, mode="major", confidence=0.97),
        chords=chords,
        voices=voices,
        violations=DEFECTS,
        engine="rules",
        latencyMs=12.4,
    )


def main() -> None:
    request = HarmonizeRequest(melody=build_melody(), engine="rules")
    response = build_response()

    # Validate BEFORE writing. Writing first meant a regeneration that broke the
    # invariant exited non-zero but left the bad fixtures on disk, where the dev
    # engine, the backend tests and the frontend mock mode would all read them.
    soprano = next(v for v in response.voices if v.name == "soprano")
    melody_notes = [(n.start, n.duration, n.pitch) for n in request.melody.notes]
    soprano_notes = [(n.start, n.duration, n.pitch) for n in soprano.notes]
    assert soprano_notes == melody_notes, "soprano must retain the input melody exactly"

    written = []
    for filename, model in (
        ("melody.request.json", request),
        ("harmonize.response.json", response),
    ):
        path = OUT_DIR / filename
        path.write_text(json.dumps(model.model_dump(mode="json"), indent=2) + "\n")
        written.append(path.name)

    print(f"Wrote {', '.join(written)}")
    print(f"  melody: {len(request.melody.notes)} notes, {len(PROGRESSION)} chords")
    print(f"  voices: {', '.join(v.name for v in response.voices)}")
    print(f"  deliberate defects: {', '.join(v.kind for v in response.violations)}")


if __name__ == "__main__":
    main()
