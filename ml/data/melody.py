"""Conversion between the API's `Melody`/`Voice` note lists and the fixed grid
the engines reason on.

Both engines share this so a rule harmonization and a neural harmonization are
literally the same kind of object, which is what makes the eval harness able to
score them with identical code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from contracts.schema import Melody, Note, Voice, VoiceName

from ..theory.pitch import Key, detect_key
from ..theory.voicing import VOICE_NAMES
from .corpus import REST, STEP, STEPS_PER_QUARTER

VOICE_ORDER: tuple[VoiceName, ...] = ("soprano", "alto", "tenor", "bass")


@dataclass
class MelodyGrid:
    """A melody sampled onto the 16th grid, plus the metric context."""

    pitches: list[int]
    onsets: list[bool]
    beat_strength: list[int]
    phrase_end: list[bool]
    step: float = STEP
    time_signature: tuple[int, int] = (4, 4)
    pickup_steps: int = 0

    @property
    def length(self) -> int:
        return len(self.pitches)

    @property
    def steps_per_beat(self) -> int:
        return max(1, int(round(STEPS_PER_QUARTER * 4 / self.time_signature[1])))

    @property
    def steps_per_measure(self) -> int:
        return max(1, self.steps_per_beat * self.time_signature[0])


def _quantize(offset: float) -> int:
    return int(round(float(offset) / STEP))


def melody_to_grid(melody: Melody) -> MelodyGrid:
    """Quantize a `Melody` to the 16th grid used everywhere downstream."""
    if not melody.notes:
        return MelodyGrid(pitches=[], onsets=[], beat_strength=[], phrase_end=[])

    notes = sorted(melody.notes, key=lambda n: (n.start, n.pitch))
    origin = min(n.start for n in notes)
    end = max(n.start + n.duration for n in notes)
    length = max(1, _quantize(end - origin))

    pitches = [REST] * length
    onsets = [False] * length
    for note in notes:
        start = _quantize(note.start - origin)
        stop = max(start + 1, _quantize(note.start + note.duration - origin))
        if start >= length:
            continue
        onsets[start] = True
        for t in range(start, min(stop, length)):
            # A higher simultaneous note wins: chorale melodies are monophonic,
            # but uploaded MIDI is not always clean.
            pitches[t] = max(pitches[t], note.pitch)

    numerator = melody.timeSignature.numerator
    denominator = melody.timeSignature.denominator
    steps_per_beat = max(1, int(round(STEPS_PER_QUARTER * 4 / denominator)))
    steps_per_measure = max(1, steps_per_beat * numerator)

    # Infer a pickup: if the first note is shorter than a full measure and the
    # total length is not a whole number of measures, assume an anacrusis.
    pickup_steps = 0
    remainder = length % steps_per_measure
    if remainder and _quantize(origin) % steps_per_measure == 0:
        pickup_steps = remainder

    beat_strength = []
    for t in range(length):
        position = (t + (steps_per_measure - pickup_steps) % steps_per_measure) % steps_per_measure
        if position == 0:
            beat_strength.append(3)
        elif position % steps_per_beat == 0:
            beat_strength.append(2)
        elif position % max(1, steps_per_beat // 2) == 0:
            beat_strength.append(1)
        else:
            beat_strength.append(0)

    return MelodyGrid(
        pitches=pitches,
        onsets=onsets,
        beat_strength=beat_strength,
        phrase_end=infer_phrase_ends(pitches, onsets, steps_per_beat),
        time_signature=(numerator, denominator),
        pickup_steps=pickup_steps,
    )


def infer_phrase_ends(pitches: Sequence[int], onsets: Sequence[bool], steps_per_beat: int) -> list[bool]:
    """Mark grid steps where a phrase closes.

    Uploaded melodies have no fermatas, so phrase ends are inferred from long
    notes and from rests — the same cues a listener uses. Cadence handling in
    the rule engine depends on getting these roughly right.
    """
    length = len(pitches)
    out = [False] * length
    if not length:
        return out
    starts = [t for t in range(length) if onsets[t]]
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else length
        duration = stop - start
        followed_by_rest = stop < length and pitches[stop] == REST
        if duration >= 2 * steps_per_beat or followed_by_rest or index == len(starts) - 1:
            for t in range(start, stop):
                out[t] = True
    return out


def detect_melody_key(grid: MelodyGrid) -> tuple[Key, float]:
    """Key of a gridded melody, duration-weighted and biased by the final note."""
    weights: list[tuple[int, float]] = []
    run_pitch, run_length = None, 0
    for t in range(grid.length):
        pitch = grid.pitches[t]
        if pitch == REST:
            if run_pitch is not None:
                weights.append((run_pitch, run_length * STEP))
            run_pitch, run_length = None, 0
            continue
        if pitch != run_pitch or grid.onsets[t]:
            if run_pitch is not None:
                weights.append((run_pitch, run_length * STEP))
            run_pitch, run_length = pitch, 0
        run_length += 1
    if run_pitch is not None:
        weights.append((run_pitch, run_length * STEP))

    if not weights:
        return Key(0, "major"), 0.0
    # Weight the final and first notes more heavily; chorale phrases resolve.
    final_pitch = weights[-1][0]
    return detect_key(weights, final_bonus_pitch=final_pitch)


def grid_to_voices(
    lines: Sequence[Sequence[int]],
    *,
    step: float = STEP,
    origin: float = 0.0,
    onsets: Sequence[Sequence[bool]] | None = None,
    velocity: int = 80,
    names: Sequence[VoiceName] | None = None,
) -> list[Voice]:
    """Turn per-voice pitch grids into `Voice` objects with merged note runs."""
    out: list[Voice] = []
    for index, line in enumerate(lines):
        notes: list[Note] = []
        t = 0
        while t < len(line):
            pitch = line[t]
            if pitch == REST:
                t += 1
                continue
            run = t + 1
            while run < len(line) and line[run] == pitch:
                if onsets is not None and index < len(onsets) and run < len(onsets[index]) and onsets[index][run]:
                    break
                run += 1
            notes.append(Note(
                pitch=int(pitch),
                start=round(origin + t * step, 6),
                duration=round((run - t) * step, 6),
                velocity=velocity,
            ))
            t = run
        if names is not None and index < len(names):
            name = names[index]
        else:
            name = VOICE_ORDER[index] if index < len(VOICE_ORDER) else VOICE_ORDER[-1]
        out.append(Voice(name=name, notes=notes))
    return out


#: Which of the four parts to keep for a reduced voice count. Two voices means
#: melody plus bass — the harmonic frame — not melody plus an inner part.
VOICE_SUBSETS = {1: (0,), 2: (0, 3), 3: (0, 1, 3), 4: (0, 1, 2, 3)}


def select_voices(lines: Sequence[Sequence[int]], voice_count: int) -> tuple[list[list[int]], list[VoiceName]]:
    """Pick `voice_count` parts from a full SATB grid, keeping the outer frame."""
    indices = VOICE_SUBSETS.get(max(1, min(voice_count, 4)), (0, 1, 2, 3))
    return [list(lines[i]) for i in indices], [VOICE_ORDER[i] for i in indices]


def voices_to_grid(voices: Sequence[Voice], *, length: int | None = None, origin: float = 0.0) -> list[list[int]]:
    """Inverse of `grid_to_voices`, used by the eval harness."""
    end = 0.0
    for voice in voices:
        for note in voice.notes:
            end = max(end, note.start + note.duration - origin)
    size = length if length is not None else max(1, _quantize(end))
    grids: list[list[int]] = []
    for voice in voices:
        line = [REST] * size
        for note in voice.notes:
            start = _quantize(note.start - origin)
            stop = max(start + 1, _quantize(note.start + note.duration - origin))
            for t in range(max(0, start), min(stop, size)):
                line[t] = note.pitch
        grids.append(line)
    return grids


def chorale_to_melody(chorale, *, tempo: float = 90.0) -> Melody:
    """The soprano of a corpus chorale, as an API `Melody`.

    This is how the harness hands a held-out chorale to an engine: the engine
    sees exactly what a user would upload — a bare tune — and nothing else.
    """
    from contracts.schema import KeySignature, TimeSignature

    notes: list[Note] = []
    line = chorale.voices[0]
    onset = chorale.onsets[0]
    t = 0
    while t < len(line):
        pitch = line[t]
        if pitch == REST:
            t += 1
            continue
        run = t + 1
        while run < len(line) and line[run] == pitch and not onset[run]:
            run += 1
        notes.append(Note(pitch=int(pitch), start=round(t * STEP, 6), duration=round((run - t) * STEP, 6)))
        t = run

    numerator, denominator = chorale.time_signature
    return Melody(
        notes=notes,
        tempo=tempo,
        timeSignature=TimeSignature(numerator=numerator, denominator=denominator),
        key=KeySignature(tonic=chorale.key.tonic, mode=chorale.key.mode),
    )
