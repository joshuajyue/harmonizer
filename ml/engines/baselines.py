"""Baseline engines that bracket the comparison from below.

`fixed_thirds` is what a commercial harmonizer does: lock to the scale, stack a
fixed diatonic interval under the melody, done. It has no notion of harmonic
function and no voice leading at all, which is exactly the point — it is the
floor, and any engine that does not clear it comfortably is not doing anything.
"""

from __future__ import annotations

from contracts.schema import Chord, KeySignature, Melody, Violation

from ..data.corpus import REST, STEP
from ..data.melody import detect_melody_key, grid_to_voices, melody_to_grid, select_voices
from ..theory.chords import analyze_chord
from ..theory.pitch import MAJOR_SCALE, NATURAL_MINOR, Key
from ..theory.voicing import ALTO, BASS, TENOR, VOICE_NAMES, VOICE_RANGES, analyze_texture, texture_from_voices
from .base import Harmonization, HarmonyEngine, register

#: Diatonic steps below the melody for each generated voice.
DEFAULT_OFFSETS = (2, 4, 7)  # a third, a fifth and an octave below


def _scale_of(key: Key) -> tuple[int, ...]:
    return NATURAL_MINOR if key.is_minor else MAJOR_SCALE


def _shift_diatonic(pitch: int, steps: int, key: Key) -> int:
    """Move `pitch` down `steps` scale degrees, snapping chromatic notes into the key."""
    scale = _scale_of(key)
    relative = (pitch % 12 - key.tonic) % 12
    octave = (pitch - key.tonic - relative) // 12

    if relative in scale:
        degree = scale.index(relative)
        chromatic_offset = 0
    else:
        degree = max(i for i, value in enumerate(scale) if value < relative)
        chromatic_offset = 0  # snap to the scale, as a scale-locked harmonizer does

    target = degree - steps
    target_octave = octave + (target // 7)
    target_degree = target % 7
    return key.tonic + scale[target_degree] + 12 * target_octave + chromatic_offset


def _fit_range(pitch: int, voice: int) -> int:
    low, high = VOICE_RANGES[voice]
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return pitch


class FixedIntervalEngine(HarmonyEngine):
    """Scale-locked parallel harmony: the commodity harmonizer, as the floor."""

    id = "fixed_thirds"
    name = "Parallel Diatonic Intervals"
    description = (
        "The commercial-harmonizer baseline: voices locked a diatonic third, fifth and "
        "octave below the melody. No harmonic function, no voice leading. Present as the "
        "floor of the comparison, not as a recommendation."
    )
    learned = False

    def __init__(self, offsets: tuple[int, ...] = DEFAULT_OFFSETS) -> None:
        self.offsets = offsets

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        grid = melody_to_grid(melody)
        if melody.key is not None:
            key, confidence = Key(melody.key.tonic, melody.key.mode), melody.key.confidence or 1.0
        else:
            key, confidence = detect_melody_key(grid)

        lines = [list(grid.pitches)]
        for index, steps in enumerate(self.offsets):
            voice = (ALTO, TENOR, BASS)[min(index, 2)]
            lines.append([
                REST if pitch == REST else _fit_range(_shift_diatonic(pitch, steps, key), voice)
                for pitch in grid.pitches
            ])

        selected, names = select_voices(lines, voice_count)
        chords = self._chords(selected, key, grid.steps_per_beat)
        violations = self._violations(selected, key, grid.steps_per_beat)
        return Harmonization(
            key=KeySignature(tonic=key.tonic, mode=key.mode, confidence=confidence),
            voices=grid_to_voices(selected, names=names),
            chords=chords,
            violations=violations,
        )

    def _chords(self, lines, key: Key, steps_per_beat: int) -> list[Chord]:
        out: list[Chord] = []
        length = len(lines[0])
        for start in range(0, length, steps_per_beat):
            pitches = [line[start] for line in lines if start < len(line) and line[start] != REST]
            label = analyze_chord(pitches, key) if pitches else None
            if label is None:
                continue
            roman = label.roman(key.mode)
            span = round(min(steps_per_beat, length - start) * STEP, 6)
            if out and out[-1].roman == roman:
                out[-1] = out[-1].model_copy(update={"duration": round(out[-1].duration + span, 6)})
                continue
            out.append(Chord(
                start=round(start * STEP, 6), duration=span, roman=roman,
                root=label.absolute_root(key), quality=label.contract_quality(),
                inversion=label.inversion,
                secondaryOf=None if label.applied_to is None else key.to_absolute(label.applied_to),
            ))
        return out

    def _violations(self, lines, key: Key, steps_per_beat: int) -> list[Violation]:
        from ..eval.metrics import step_chords

        texture = texture_from_voices([[None if p == REST else p for p in line] for line in lines], step=STEP)
        chords = step_chords(lines, key, steps_per_beat=steps_per_beat)
        return [
            Violation(
                kind=defect.kind, severity=defect.severity, start=defect.offset,
                voices=[VOICE_NAMES[v] for v in defect.voices if v < 4], message=defect.message,
            )
            for defect in analyze_texture(texture, key, chords)
            if defect.severity != "info"
        ]


register(FixedIntervalEngine())
