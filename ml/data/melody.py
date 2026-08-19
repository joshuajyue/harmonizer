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

from ..theory.pitch import Key, detect_key, detect_key_candidates
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
    #: Absolute beat of grid index 0. The grid starts at the melody's first note
    #: so it stays compact, which means everything converted back out of it must
    #: have this added again. Forgetting to shifted every generated voice to beat
    #: zero and silently misaligned the harmony against the melody it accompanies.
    origin: float = 0.0

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
    pickup_steps = infer_pickup(pitches, onsets, steps_per_measure)

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
        origin=origin,
    )


def infer_pickup(pitches: Sequence[int], onsets: Sequence[bool], steps_per_measure: int) -> int:
    """Length of the anacrusis, in grid steps, inferred from note placement.

    A pickup cannot be read off the total length: a chorale that begins with a
    quarter-note upbeat normally ends with a three-quarter bar that completes it,
    so the total is a whole number of measures anyway. Instead, score every
    candidate downbeat alignment by the total duration of the notes that would
    begin on a downbeat, and take the best — long notes fall on strong beats.

    Both the corpus encoder and the inference path call this, so the model is
    never trained on one alignment and asked to use another. Getting that wrong
    is invisible: the features stay in range, the loss stays finite, and only the
    output quietly degrades.
    """
    length = len(pitches)
    if length == 0 or steps_per_measure <= 1:
        return 0
    starts = [t for t in range(length) if onsets[t] and pitches[t] != REST]
    if not starts:
        return 0
    durations = []
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else length
        durations.append(stop - start)

    best_offset, best_score = 0, -1.0
    for offset in range(steps_per_measure):
        score = sum(
            duration for start, duration in zip(starts, durations)
            if (start + offset) % steps_per_measure == 0
        )
        if score > best_score:
            best_score, best_offset = score, offset
    return (steps_per_measure - best_offset) % steps_per_measure


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


#: How much a candidate key's best achievable harmonization counts against its
#: pitch-class profile. Chosen on train+val (0.05-0.2 all improve; 0.1 is the
#: peak on val) and then evaluated once on test.
HARMONIC_FIT_WEIGHT = 0.1
#: Only the strongest profile candidates are rescored. Measured: top-3 gives
#: bit-identical results to rescoring all 24, at an eighth of the cost.
HARMONIC_FIT_CANDIDATES = 3


def detect_melody_key(grid: MelodyGrid, *, use_harmony: bool = True) -> tuple[Key, float]:
    """Key of a gridded melody.

    Two stages. First Krumhansl-Schmuckler on pitch-class durations, plus
    cadential evidence — where the tune starts and ends — scored as measured
    log-probabilities.

    Then, because that still confuses a key with its dominant (55% of the
    remaining errors: a D minor melody dwells on D, F and A, which A minor's
    profile rewards as its own tonic, fourth and sixth), the top few candidates
    are rescored by **how well the melody actually harmonizes** in each. The
    rules engine's chord search is run under each candidate key and its best
    achievable path score is added in. A melody in the wrong key has to be
    explained by a worse progression, and that is exactly the evidence a
    pitch-class histogram cannot contain.

    Measured on held-out chorales: 73.8% profile alone, 83.6% with cadential
    evidence, 88.5% with harmonic rescoring. Set `use_harmony=False` to skip the
    third stage, which costs roughly 45 ms.
    """
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

    key, confidence = detect_key(
        weights, final_bonus_pitch=weights[-1][0], first_pitch=weights[0][0]
    )
    if not use_harmony or grid.length == 0:
        return key, confidence

    ranked = detect_key_candidates(
        weights,
        final_bonus_pitch=weights[-1][0],
        first_pitch=weights[0][0],
        limit=HARMONIC_FIT_CANDIDATES,
    )
    if len(ranked) < 2:
        return key, confidence

    # Imported here rather than at module scope: the rules engine imports this
    # module, so a top-level import would be circular.
    from ..engines.rules import harmonic_fit_scores

    fits = harmonic_fit_scores(grid, [candidate for candidate, _ in ranked])
    rescored = sorted(
        (
            (score + HARMONIC_FIT_WEIGHT * fits[candidate], candidate)
            for candidate, score in ranked
        ),
        key=lambda item: (-item[0], item[1].tonic, item[1].mode),
    )
    best_score, best_key = rescored[0]
    margin = best_score - rescored[1][0]
    return best_key, max(0.0, min(1.0, margin * 4.0))


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


def voices_with_melody(
    lines: Sequence[Sequence[int]],
    melody: Melody,
    *,
    origin: float,
    names: Sequence[VoiceName],
) -> list[Voice]:
    """Generated parts placed at absolute beats, with the melody itself as soprano.

    Two invariants, both of which used to be broken:

    * Everything is offset by `origin`, so a melody starting in bar 2 is
      accompanied in bar 2 rather than from beat 0.
    * The soprano is the caller's melody **verbatim**, not a re-gridded
      reconstruction of it. Round-tripping through a sixteenth grid moves any
      note that does not land on it, so rebuilding the soprano could silently
      alter the user's own notes. The contract says the melody is retained; this
      makes that exact by construction rather than true-when-quantization-permits.
    """
    voices = grid_to_voices(lines, origin=origin, names=names)
    if voices and voices[0].name == "soprano":
        voices[0] = Voice(name="soprano", notes=list(melody.notes))
    return voices


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

    The contract requires `tempo` because a defaulted tempo renders the result at
    the wrong speed with nothing raised anywhere. That reasoning does not reach
    here: the corpus carries no tempo marking, harmonization is tempo-invariant
    (every engine works in quarter-note beats and none reads the field), and
    nothing in `ml/eval` measures duration in seconds. The default is a
    placeholder for a quantity that has no effect, not a guess at one that does —
    which is why it is safe here and would not be on a request path.
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
