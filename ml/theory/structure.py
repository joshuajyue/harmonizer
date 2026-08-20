"""Structural defects: the errors a listener notices instantly.

Everything in `voicing.py` is a *local* defect — two sonorities, a voice pair,
one interval. That is the whole of classical voice-leading pedagogy, and it has a
blind spot big enough to drive a piece through: **a harmonization that ends on
the dominant scores zero defects there.** It breaks no voice-leading rule. It is
also obviously, audibly wrong in a way that no parallel fifth ever is.

This module supplies the missing category. It reasons over the chord sequence and
the phrase structure rather than over adjacent sonorities:

    no_tonal_closure           the piece does not end anywhere Bach would end
    implausible_phrase_cadence a phrase closes on a chord that never closes one
    key_not_established        no tonic-rooted chord in the opening

The thresholds are measured, not asserted (`ml/training/calibrate_cadences.py`),
because the obvious rule is wrong. "A piece must end on the tonic" flags 15% of
Bach: he closes ~8% of chorales on a root-position V, rising to 12.8% in minor
keys, and checking the pitch content confirms only 1 of those 22 pieces has a
mistaken key label. Those are Phrygian half cadences, and they are idiomatic.

So ending on V is reported, at info severity, as `half_cadence_ending` — visible,
with Bach's own rate beside it as the yardstick — rather than being scored as a
defect. Anything Bach genuinely never does is what counts as structural.
"""

from __future__ import annotations

from typing import Sequence

from ._cadence_priors import (
    KEY_ESTABLISHMENT_BEATS,
    PLAUSIBLE_FINAL_CHORDS,
    PLAUSIBLE_PHRASE_CHORDS,
)
from .chords import ChordLabel
from .pitch import Key
from .voicing import Defect

#: Defects that are wrong rather than merely irregular. These stay near zero even
#: when an engine is deliberately given licence to be adventurous elsewhere:
#: structure is a constraint, surface detail is where the freedom belongs.
STRUCTURAL_KINDS = ("no_tonal_closure", "implausible_phrase_cadence", "key_not_established")

#: Reported but not a defect: Bach does this, so an engine may too. It is
#: surfaced because an engine doing it 40% of the time is broken even though an
#: engine doing it 8% of the time is being idiomatic.
DESCRIPTIVE_KINDS = ("half_cadence_ending",)

_SEVERITY = {
    "no_tonal_closure": "error",
    "implausible_phrase_cadence": "error",
    "key_not_established": "error",
    "half_cadence_ending": "info",
}


def _chord_class(chord: ChordLabel) -> tuple[int, str]:
    return (chord.relative_root, chord.quality)


def find_structural_defects(
    chords: Sequence[ChordLabel | None],
    key: Key,
    *,
    phrase_ends: Sequence[bool] | None = None,
    beat_duration: float = 1.0,
) -> list[Defect]:
    """Structural defects of a whole harmonization.

    `chords` is one entry per beat (None where nothing sounds). `phrase_ends`
    marks beats inside a phrase-final note; a phrase is judged at the LAST such
    beat, since that is where the cadence lands.
    """
    sounding = [(index, chord) for index, chord in enumerate(chords) if chord is not None]
    if not sounding:
        return []

    defects: list[Defect] = []
    plausible_final = PLAUSIBLE_FINAL_CHORDS[key.mode]
    plausible_phrase = PLAUSIBLE_PHRASE_CHORDS[key.mode]

    final_index, final_chord = sounding[-1]
    final_offset = final_index * beat_duration
    if _chord_class(final_chord) not in plausible_final:
        defects.append(Defect(
            "no_tonal_closure", _SEVERITY["no_tonal_closure"], final_offset, (),
            f"piece ends on {final_chord.roman(key.mode)}, which does not close a piece in "
            f"{key.name()}",
        ))
    elif final_chord.relative_root == 7 and final_chord.inversion == 0:
        defects.append(Defect(
            "half_cadence_ending", _SEVERITY["half_cadence_ending"], final_offset, (),
            f"piece ends on {final_chord.roman(key.mode)} — a half cadence, which Bach also "
            "writes, but check it is intended",
        ))

    if not any(chord.relative_root == 0 for index, chord in sounding if index <= KEY_ESTABLISHMENT_BEATS):
        defects.append(Defect(
            "key_not_established", _SEVERITY["key_not_established"], 0.0, (),
            f"no tonic-rooted chord in the first {KEY_ESTABLISHMENT_BEATS} beats, so "
            f"{key.name()} is never established",
        ))

    if phrase_ends is not None:
        for position, (index, chord) in enumerate(sounding):
            if index >= len(phrase_ends) or not phrase_ends[index]:
                continue
            following = sounding[position + 1][0] if position + 1 < len(sounding) else None
            if following is not None and following < len(phrase_ends) and phrase_ends[following]:
                continue  # not yet the last beat of this phrase
            if index == final_index:
                continue  # the final cadence is judged by no_tonal_closure
            if _chord_class(chord) not in plausible_phrase:
                defects.append(Defect(
                    "implausible_phrase_cadence", _SEVERITY["implausible_phrase_cadence"],
                    index * beat_duration, (),
                    f"phrase ends on {chord.roman(key.mode)}, which does not close a phrase in "
                    f"{key.name()}",
                ))

    defects.sort(key=lambda defect: (defect.offset, defect.kind))
    return defects


def phrase_end_beats(phrase_ends: Sequence[bool], steps_per_beat: int) -> list[bool]:
    """Collapse a per-grid-step phrase mask onto per-beat resolution."""
    length = (len(phrase_ends) + steps_per_beat - 1) // steps_per_beat
    return [
        any(phrase_ends[t] for t in range(beat * steps_per_beat,
                                          min((beat + 1) * steps_per_beat, len(phrase_ends))))
        for beat in range(length)
    ]
