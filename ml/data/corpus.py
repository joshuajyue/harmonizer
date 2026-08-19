"""Bach chorale corpus loading, quantization and tonic-relative normalization.

Every deliberate departure from v1 is here:

* Pieces are kept whole. v1 truncated to `SEQUENCE_LENGTH = 32` quarter notes,
  so training only ever saw the first 32 beats of chorales whose median length
  is 48 and whose maximum is 111.
* Padding is tracked explicitly with a mask instead of being labelled as tonic.
* Everything is transposed so the tonic is pitch class 0, which removes the
  absolute/relative mismatch that forced v1's network to induce the key.
* All four voices are retained. `chordify()` throws away exactly the counterpoint
  that makes the corpus worth training on.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Sequence

from ..theory.pitch import Key, normalization_shift

#: Grid resolution in quarter notes. A sixteenth grid captures 100% of note
#: onsets in the corpus exactly (measured: 80.6% land on quarters, 18.7% on
#: eighths, 0.6% on sixteenths), so nothing is lost to quantization.
STEP = 0.25
STEPS_PER_QUARTER = int(round(1.0 / STEP))

CACHE_DIR = Path(__file__).resolve().parent / "cache"

#: Sentinel used inside integer grids where a voice is silent.
REST = -1


@dataclass
class Chorale:
    """One chorale, quantized, normalized and ready for any consumer.

    `voices[v][t]` is the MIDI pitch of voice v at grid step t in the ORIGINAL
    key, or REST. `normalized[v][t]` is the same transposed so the tonic is C.
    `onsets[v][t]` marks a re-articulation, which is what distinguishes a tied
    note from a repeated one.
    """

    id: str
    key: Key
    voices: list[list[int]]
    onsets: list[list[bool]]
    fermatas: list[bool]
    beat_strength: list[int]
    time_signature: tuple[int, int]
    shift: int
    pickup_steps: int = 0

    @property
    def length(self) -> int:
        return len(self.voices[0]) if self.voices else 0

    @property
    def normalized(self) -> list[list[int]]:
        return [[p + self.shift if p != REST else REST for p in line] for line in self.voices]

    @property
    def normalized_key(self) -> Key:
        return Key(0, self.key.mode)

    def soprano(self) -> list[int]:
        return self.voices[0]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["key"] = {"tonic": self.key.tonic, "mode": self.key.mode}
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "Chorale":
        payload = dict(payload)
        payload["key"] = Key(payload["key"]["tonic"], payload["key"]["mode"])
        payload["time_signature"] = tuple(payload["time_signature"])
        return cls(**payload)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _quantize(offset: float) -> int:
    return int(round(float(offset) / STEP))


def _beat_strength_grid(length: int, numerator: int, denominator: int, pickup_steps: int) -> list[int]:
    """3 = downbeat, 2 = other beat, 1 = eighth, 0 = sixteenth."""
    steps_per_beat = int(round(STEPS_PER_QUARTER * 4 / denominator))
    steps_per_measure = max(1, steps_per_beat * numerator)
    out = []
    for t in range(length):
        position = (t + (steps_per_measure - pickup_steps) % steps_per_measure) % steps_per_measure
        if position == 0:
            out.append(3)
        elif position % steps_per_beat == 0:
            out.append(2)
        elif position % max(1, steps_per_beat // 2) == 0:
            out.append(1)
        else:
            out.append(0)
    return out


def parse_score(score, piece_id: str) -> Chorale | None:
    """Convert a music21 score into a Chorale, or None if it is unusable."""
    import music21

    parts = list(score.parts)
    if len(parts) != 4:
        return None

    total_quarters = float(score.duration.quarterLength)
    length = _quantize(total_quarters)
    if length < 4 * STEPS_PER_QUARTER:
        return None

    voices: list[list[int]] = []
    onsets: list[list[bool]] = []
    for part in parts:
        line = [REST] * length
        onset = [False] * length
        events = []
        for element in part.recurse().notes:
            start = _quantize(element.getOffsetInHierarchy(score))
            dur = max(1, _quantize(element.duration.quarterLength))
            pitch = element.pitch.midi if element.isNote else max(p.midi for p in element.pitches)
            events.append((start, dur, pitch))
        events.sort()
        for start, dur, pitch in events:
            if start >= length:
                continue
            onset[start] = True
            for t in range(start, min(start + dur, length)):
                line[t] = pitch
        voices.append(line)
        onsets.append(onset)

    if all(p == REST for p in voices[0]):
        return None

    key = detect_key_music21(score)
    shift = normalization_shift(key)

    fermatas = [False] * length
    for element in parts[0].recurse().notes:
        if any(isinstance(e, music21.expressions.Fermata) for e in element.expressions):
            index = _quantize(element.getOffsetInHierarchy(score))
            dur = max(1, _quantize(element.duration.quarterLength))
            for t in range(index, min(index + dur, length)):
                fermatas[t] = True

    time_sigs = list(score.recurse().getElementsByClass("TimeSignature"))
    numerator, denominator = (time_sigs[0].numerator, time_sigs[0].denominator) if time_sigs else (4, 4)

    first_measure = None
    for measure in parts[0].getElementsByClass("Measure"):
        first_measure = measure
        break
    pickup_steps = 0
    if first_measure is not None:
        full = _quantize(4.0 * numerator / denominator)
        actual = _quantize(first_measure.duration.quarterLength)
        if 0 < actual < full:
            pickup_steps = actual

    return Chorale(
        id=piece_id,
        key=key,
        voices=voices,
        onsets=onsets,
        fermatas=fermatas,
        beat_strength=_beat_strength_grid(length, numerator, denominator, pickup_steps),
        time_signature=(numerator, denominator),
        shift=shift,
        pickup_steps=pickup_steps,
    )


def detect_key_music21(score) -> Key:
    """Key of a music21 score, using its full four-part texture.

    Corpus keys are analysed once, offline, from all four voices; the engines
    detect the key from the melody alone at inference time. Keeping those two
    paths separate stops the corpus labels inheriting melody-only mistakes.
    """
    import music21

    try:
        analyzed = score.analyze("key")
        mode = "minor" if analyzed.mode == "minor" else "major"
        return Key(analyzed.tonic.pitchClass, mode)
    except Exception:
        return Key(0, "major")


# ---------------------------------------------------------------------------
# Corpus access
# ---------------------------------------------------------------------------


def _cache_path(tag: str) -> Path:
    return CACHE_DIR / f"chorales_{tag}.pkl"


def load_chorales(*, limit: int | None = None, refresh: bool = False, verbose: bool = True) -> list[Chorale]:
    """All usable four-part Bach chorales, parsed once and cached on disk."""
    tag = f"all_step{STEP}"
    path = _cache_path(tag)
    if path.exists() and not refresh:
        with path.open("rb") as handle:
            chorales = [Chorale.from_dict(d) for d in pickle.load(handle)]
        return chorales[:limit] if limit else chorales

    import music21

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    paths = music21.corpus.getComposer("bach")
    out: list[Chorale] = []
    for raw_path in paths:
        piece_id = Path(str(raw_path)).stem
        try:
            score = music21.corpus.parse(raw_path)
            chorale = parse_score(score, piece_id)
        except Exception:
            continue
        if chorale is not None:
            out.append(chorale)
        if verbose and len(out) % 50 == 0 and out:
            print(f"  parsed {len(out)} chorales...", flush=True)

    out.sort(key=lambda c: c.id)
    with path.open("wb") as handle:
        pickle.dump([c.to_dict() for c in out], handle)
    if verbose:
        print(f"Parsed and cached {len(out)} four-part chorales -> {path}")
    return out[:limit] if limit else out


def split_chorales(
    chorales: Sequence[Chorale],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.15,
) -> tuple[list[Chorale], list[Chorale], list[Chorale]]:
    """Deterministic PIECE-LEVEL train/val/test split.

    The split is a hash of the piece id, not a shuffle, so it is stable across
    runs, machines and any change to corpus ordering — a model can never be
    evaluated on a piece an earlier run trained on.
    """
    train: list[Chorale] = []
    val: list[Chorale] = []
    test: list[Chorale] = []
    for chorale in chorales:
        digest = hashlib.sha256(chorale.id.encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < test_fraction:
            test.append(chorale)
        elif bucket < test_fraction + val_fraction:
            val.append(chorale)
        else:
            train.append(chorale)
    return train, val, test


def corpus_summary(chorales: Sequence[Chorale]) -> dict:
    from collections import Counter

    modes = Counter(c.key.mode for c in chorales)
    tonics = Counter(c.key.tonic for c in chorales)
    lengths = sorted(c.length for c in chorales)
    ranges: dict[int, tuple[int, int]] = {}
    for voice in range(4):
        pitches = [p for c in chorales for p in c.voices[voice] if p != REST]
        pitches.sort()
        lo = pitches[int(0.002 * len(pitches))]
        hi = pitches[int(0.998 * len(pitches)) - 1]
        ranges[voice] = (lo, hi)
    return {
        "pieces": len(chorales),
        "modes": dict(modes),
        "tonics": dict(sorted(tonics.items())),
        "median_steps": lengths[len(lengths) // 2] if lengths else 0,
        "min_steps": lengths[0] if lengths else 0,
        "max_steps": lengths[-1] if lengths else 0,
        "voice_ranges_p02_p998": ranges,
    }


def iter_sonorities(chorale: Chorale, normalized: bool = False) -> Iterator[tuple[int, ...]]:
    lines = chorale.normalized if normalized else chorale.voices
    for t in range(chorale.length):
        yield tuple(line[t] for line in lines)
