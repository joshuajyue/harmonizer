"""Test melodies: traditional tunes and real jazz lines.

Two sources, for two different jobs.

*Traditional tunes* are hand-encoded here (all long out of copyright) and are
the demo and regression material: short, diatonic, well known, and exactly the
kind of thing a user uploads. Reharmonizing a hymn or folk tune in jazz is also
a real practice, not a contrivance.

*Weimar solo choruses* are the honest evaluation material. Each one is a real
jazz line for which the changes an actual rhythm section played are known, so a
reharmonization of it can be compared against a human reference rather than
against taste. Solo lines are more chromatic than heads, which makes them a
harder and more revealing test than a folk tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from contracts.schema import KeySignature, Melody, Note, TimeSignature

from .data import Progression, chorus_melody, chorus_progression, load_solos


@dataclass(frozen=True)
class TestTune:
    name: str
    melody: Melody
    #: The changes a human actually played, when one exists.
    reference: Progression | None = None


def _melody(
    pitches_and_durations: Sequence[tuple[int, float]],
    *,
    tonic: int,
    mode: str = "major",
    tempo: float = 120.0,
    meter: tuple[int, int] = (4, 4),
) -> Melody:
    notes: list[Note] = []
    offset = 0.0
    for pitch, duration in pitches_and_durations:
        if pitch >= 0:
            notes.append(Note(pitch=pitch, start=round(offset, 6), duration=duration))
        offset += duration
    return Melody(
        notes=notes,
        tempo=tempo,
        timeSignature=TimeSignature(numerator=meter[0], denominator=meter[1]),
        key=KeySignature(tonic=tonic, mode=mode),
    )


# Pitch shorthand: middle C = 60.
C4, D4, E4, F4, G4, A4, B4 = 60, 62, 64, 65, 67, 69, 71
C5, D5, E5, F5, G5, A5 = 72, 74, 76, 77, 79, 81
A3, B3, G3, F3, E3 = 57, 59, 55, 53, 52

#: "Twinkle, Twinkle" — traditional (Ah! vous dirai-je, maman, 1761).
TWINKLE = _melody(
    [
        (C4, 1), (C4, 1), (G4, 1), (G4, 1), (A4, 1), (A4, 1), (G4, 2),
        (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 1), (D4, 1), (C4, 2),
        (G4, 1), (G4, 1), (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 2),
        (G4, 1), (G4, 1), (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 2),
        (C4, 1), (C4, 1), (G4, 1), (G4, 1), (A4, 1), (A4, 1), (G4, 2),
        (F4, 1), (F4, 1), (E4, 1), (E4, 1), (D4, 1), (D4, 1), (C4, 4),
    ],
    tonic=0,
)

#: "Amazing Grace" — traditional, New Britain (1835). 3/4, pickup.
AMAZING_GRACE = _melody(
    [
        (G4, 1),
        (C5, 2), (E5, 0.5), (C5, 0.5), (E5, 2), (D5, 1), (C5, 2), (A4, 1),
        (G4, 3), (G4, 1), (C5, 2), (E5, 0.5), (C5, 0.5), (E5, 2), (D5, 1),
        (G5, 3), (G5, 2), (E5, 1), (G5, 2), (E5, 0.5), (G5, 0.5), (E5, 2), (D5, 1),
        (C5, 2), (A4, 1), (G4, 3), (G4, 1), (C5, 2), (E5, 0.5), (C5, 0.5),
        (E5, 2), (D5, 1), (C5, 3),
    ],
    tonic=0,
    meter=(3, 4),
)

#: "Greensleeves" — traditional English, c. 1580. Minor, 3/4.
GREENSLEEVES = _melody(
    [
        (A4, 1),
        (C5, 2), (D5, 1), (E5, 1.5), (F5, 0.5), (E5, 1), (D5, 2), (B4, 1),
        (G4, 1.5), (A4, 0.5), (B4, 1), (C5, 2), (A4, 1), (A4, 1.5), (G4, 0.5),
        (A4, 1), (B4, 2), (G4, 1), (E4, 3),
        (C5, 2), (D5, 1), (E5, 1.5), (F5, 0.5), (E5, 1), (D5, 2), (B4, 1),
        (G4, 1.5), (A4, 0.5), (B4, 1), (C5, 1), (B4, 1), (A4, 1), (G4, 1.5),
        (F4, 0.5), (G4, 1), (A4, 3),
    ],
    tonic=9,
    mode="minor",
    meter=(3, 4),
)

#: "Scarborough Fair" — traditional English. Dorian-inflected minor.
SCARBOROUGH = _melody(
    [
        (A4, 1), (A4, 2), (E5, 2), (E5, 1), (B4, 3),
        (C5, 1), (B4, 1), (A4, 1), (A4, 1), (G4, 1), (A4, 1), (B4, 3),
        (A4, 1), (A4, 2), (E5, 2), (E5, 1), (E5, 3),
        (F5, 1), (E5, 1), (D5, 1), (C5, 1), (B4, 1), (A4, 1), (B4, 3),
    ],
    tonic=9,
    mode="minor",
    meter=(3, 4),
)

#: "House of the Rising Sun" — traditional American folk. Minor, 6/8 written 3/4.
RISING_SUN = _melody(
    [
        (A4, 1), (C5, 1), (E5, 1), (F5, 2), (E5, 1),
        (A4, 1), (C5, 1), (E5, 1), (D5, 3),
        (A4, 1), (C5, 1), (E5, 1), (F5, 2), (E5, 1),
        (C5, 1), (B4, 1), (A4, 1), (A4, 3),
    ],
    tonic=9,
    mode="minor",
    meter=(3, 4),
)

#: "Shenandoah" — traditional American, early 19th century.
SHENANDOAH = _melody(
    [
        (C4, 1),
        (F4, 2), (A4, 1), (C5, 2), (D5, 1), (C5, 1), (A4, 1), (F4, 2),
        (D4, 1), (C4, 3), (C4, 1),
        (F4, 2), (A4, 1), (C5, 2), (D5, 1), (F5, 3), (E5, 1),
        (D5, 2), (C5, 1), (A4, 2), (F4, 1), (G4, 4),
    ],
    tonic=5,
)

#: A blues head shape in F — generic riff, not a copyrighted tune.
BLUES_RIFF = _melody(
    [
        (F4, 1), (A4, 0.5), (C5, 0.5), (D5, 1), (C5, 1),
        (F4, 1), (A4, 0.5), (C5, 0.5), (D5, 1), (C5, 1),
        (F5, 2), (D5, 1), (C5, 1),
        (A4, 2), (F4, 2),
        (Bb4 := 70, 1), (D5, 0.5), (F5, 0.5), (G5, 1), (F5, 1),
        (Bb4, 1), (D5, 0.5), (F5, 0.5), (G5, 1), (F5, 1),
        (F5, 2), (D5, 1), (C5, 1),
        (A4, 2), (F4, 2),
        (C5, 1), (E5, 0.5), (G5, 0.5), (A5, 1), (G5, 1),
        (Bb4, 1), (D5, 0.5), (F5, 0.5), (G5, 1), (F5, 1),
        (F4, 2), (A4, 1), (C5, 1),
        (C5, 2), (F4, 2),
    ],
    tonic=5,
)

TRADITIONAL: dict[str, Melody] = {
    "twinkle": TWINKLE,
    "amazing_grace": AMAZING_GRACE,
    "greensleeves": GREENSLEEVES,
    "scarborough_fair": SCARBOROUGH,
    "rising_sun": RISING_SUN,
    "shenandoah": SHENANDOAH,
    "blues_riff": BLUES_RIFF,
}


def traditional_tunes() -> list[TestTune]:
    return [TestTune(name=name, melody=melody) for name, melody in TRADITIONAL.items()]


def jazz_tunes(
    *,
    limit: int = 40,
    chorus: int = 1,
    min_notes: int = 24,
    download: bool = True,
    tempo: float = 140.0,
) -> list[TestTune]:
    """Real jazz choruses with the changes that were played under them."""
    out: list[TestTune] = []
    for solo in load_solos(download=download):
        line = chorus_melody(solo, chorus)
        reference = chorus_progression(solo, chorus)
        if len(line) < min_notes or not reference.spans:
            continue
        notes = [
            Note(pitch=pitch, start=round(max(0.0, start), 6), duration=round(max(0.125, duration), 6))
            for start, pitch, duration in line
        ]
        melody = Melody(
            notes=notes,
            tempo=tempo,
            timeSignature=TimeSignature(numerator=solo.meter[0], denominator=solo.meter[1]),
            key=KeySignature(tonic=solo.tonic, mode=solo.mode),
        )
        out.append(TestTune(name=f"{solo.title} — {solo.performer}", melody=melody, reference=reference))
        if len(out) >= limit:
            break
    return out
