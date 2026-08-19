"""The Bach oracle: the ceiling of the comparison.

Running Bach's own four parts through the identical metric code is the single
most important control in this harness. It answers the question v1 could never
answer — "is 1.2 parallel fifths per 100 chords good?" — because it shows what
Bach himself scores under exactly these detectors. Without it, every defect rate
is measuring the detector as much as the engine.

This is not registered as a servable engine: it only works on corpus pieces, and
exposing it through the API would be meaningless.
"""

from __future__ import annotations

from contracts.schema import KeySignature, Melody, Violation

from ..data.corpus import REST, STEP, Chorale
from ..data.melody import grid_to_voices, melody_to_grid
from ..engines.base import Harmonization, HarmonyEngine
from ..theory.chords import analyze_chord
from ..theory.pitch import Key
from ..theory.voicing import VOICE_NAMES, analyze_texture, texture_from_voices


class BachOracleEngine(HarmonyEngine):
    """Replays Bach's own alto, tenor and bass for a corpus melody."""

    id = "bach_oracle"
    name = "J. S. Bach (ground truth)"
    description = "The original four-part setting, scored by the same metrics. The ceiling, not an engine."
    learned = False

    def __init__(self, chorales: list[Chorale]) -> None:
        self._by_soprano: dict[tuple[int, ...], Chorale] = {}
        for chorale in chorales:
            self._by_soprano[tuple(chorale.voices[0])] = chorale

    def is_available(self) -> bool:
        return bool(self._by_soprano)

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        grid = melody_to_grid(melody)
        chorale = self._by_soprano.get(tuple(grid.pitches))
        if chorale is None:
            raise KeyError("BachOracleEngine only harmonizes melodies drawn from the corpus")

        key = chorale.key
        lines = [list(line) for line in chorale.voices]
        texture = texture_from_voices([[None if p == REST else p for p in line] for line in lines], step=STEP)

        steps_per_beat = grid.steps_per_beat
        per_step = []
        for t in range(chorale.length):
            beat_start = (t // steps_per_beat) * steps_per_beat
            pitches = [line[beat_start] for line in lines if line[beat_start] != REST]
            per_step.append(analyze_chord(pitches, key) if pitches else None)

        violations = [
            Violation(
                kind=defect.kind, severity=defect.severity, start=defect.offset,
                voices=[VOICE_NAMES[v] for v in defect.voices if v < 4], message=defect.message,
            )
            for defect in analyze_texture(texture, key, per_step)
            if defect.severity != "info"
        ]
        return Harmonization(
            key=KeySignature(tonic=key.tonic, mode=key.mode, confidence=1.0),
            voices=grid_to_voices(lines, onsets=chorale.onsets),
            chords=[],
            violations=violations,
        )
