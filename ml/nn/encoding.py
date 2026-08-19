"""Tensor encoding for the learned engine.

Three of the four v1 failures are prevented structurally in this file:

* **Padding.** Sequences carry an explicit `valid` mask. v1 padded to 32 steps
  and labelled the padding as tonic, so the loss rewarded predicting I on
  nothing and the reported accuracy was partly fiction. Nothing here is ever
  scored outside `valid`.
* **Representation.** `Encoding.TONIC_RELATIVE` transposes every piece so the
  tonic is C before tokenising. v1 fed absolute pitch classes and trained on
  tonic-relative targets, forcing the network to induce the key from ~400
  chorales given a single `is_minor` bit. `Encoding.ABSOLUTE` is kept so that
  handicap can be measured rather than asserted.
* **Truncation.** Pieces are kept whole and batched with a length mask, so the
  model sees complete chorales rather than their first 32 quarter notes.

Voices are modelled directly. Chord labels are derived from the generated voices
afterwards, so the label space cannot be too small to represent the corpus —
there is no label space.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from ..data.corpus import REST, STEPS_PER_QUARTER, Chorale
from ..data.melody import infer_phrase_ends, infer_pickup

N_VOICES = 4
#: Largest metric position index kept; 4/4 at a sixteenth grid needs 16.
MAX_POSITION = 16

#: Median soprano pitch of the corpus AFTER tonic normalization, measured over
#: all 368 chorales. Used at inference to keep an incoming melody in the
#: register the model was trained in: `normalization_shift` is chosen by key
#: alone, so a tune written high and then transposed up a fifth normalizes an
#: octave above anything the model has seen, and the output degrades silently.
NORMALIZED_SOPRANO_MEDIAN = 72
#: Only correct by whole octaves, and only when the melody is clearly outside
#: the trained register — a real high or low setting is information, not noise.
REGISTER_CORRECTION_THRESHOLD = 7.0


class Encoding(str, Enum):
    """Which pitch representation to tokenise in.

    TONIC_RELATIVE is the design; ABSOLUTE exists only so the v1 handicap can be
    measured under an otherwise identical model, data and budget.
    """

    TONIC_RELATIVE = "tonic_relative"
    ABSOLUTE = "absolute"


@dataclass
class VoiceVocab:
    """Pitch alphabet for one voice. Range is enforced by construction."""

    pitches: list[int]

    def __post_init__(self) -> None:
        self._index = {pitch: i for i, pitch in enumerate(self.pitches)}

    @property
    def rest_id(self) -> int:
        return len(self.pitches)

    @property
    def mask_id(self) -> int:
        return len(self.pitches) + 1

    @property
    def size(self) -> int:
        return len(self.pitches) + 2

    def encode(self, pitch: int) -> int:
        if pitch == REST:
            return self.rest_id
        index = self._index.get(pitch)
        if index is not None:
            return index
        # Snap by octaves, then clamp: an out-of-alphabet pitch must still map
        # somewhere deterministic rather than crash a request.
        candidate = pitch
        while candidate < self.pitches[0]:
            candidate += 12
        while candidate > self.pitches[-1]:
            candidate -= 12
        if candidate in self._index:
            return self._index[candidate]
        return min(range(len(self.pitches)), key=lambda i: abs(self.pitches[i] - pitch))

    def decode(self, token: int) -> int:
        if token >= len(self.pitches):
            return REST
        return self.pitches[token]


@dataclass
class Vocabulary:
    voices: list[VoiceVocab]
    encoding: Encoding

    @property
    def sizes(self) -> list[int]:
        return [voice.size for voice in self.voices]

    def to_dict(self) -> dict:
        return {"pitches": [voice.pitches for voice in self.voices], "encoding": self.encoding.value}

    @classmethod
    def from_dict(cls, payload: dict) -> "Vocabulary":
        return cls(
            voices=[VoiceVocab(list(p)) for p in payload["pitches"]],
            encoding=Encoding(payload["encoding"]),
        )


def register_correction(median_pitch: float, target: int = NORMALIZED_SOPRANO_MEDIAN) -> int:
    """Octave shift that brings a normalized melody back into the trained register.

    Returns a multiple of 12 (usually 0). Applied on top of the tonic
    normalization at inference and undone on output, so it changes nothing about
    the harmonization except keeping the model in distribution.
    """
    offset = target - median_pitch
    if abs(offset) < REGISTER_CORRECTION_THRESHOLD:
        return 0
    return int(round(offset / 12.0)) * 12


def build_vocabulary(chorales: Sequence[Chorale], encoding: Encoding, *, margin: int = 2) -> Vocabulary:
    """Pitch alphabets covering every note in the corpus, plus a little headroom.

    The margin matters at inference: a user melody can sit slightly outside the
    corpus range, and the accompanying voices then need pitches Bach never wrote
    in that part.
    """
    voices: list[VoiceVocab] = []
    for voice in range(N_VOICES):
        low, high = 127, 0
        for chorale in chorales:
            line = chorale.normalized[voice] if encoding is Encoding.TONIC_RELATIVE else chorale.voices[voice]
            for pitch in line:
                if pitch == REST:
                    continue
                low, high = min(low, pitch), max(high, pitch)
        voices.append(VoiceVocab(list(range(low - margin, high + margin + 1))))
    return Vocabulary(voices=voices, encoding=encoding)


@dataclass
class Example:
    """One encoded chorale."""

    tokens: np.ndarray        # (4, T) int64
    valid: np.ndarray         # (T,) bool — everything outside is padding
    metric: np.ndarray        # (T,) int64, 0-3 beat strength
    position: np.ndarray      # (T,) int64, index within the measure
    phrase: np.ndarray        # (T,) int64, 0/1 phrase-end marker
    mode: int                 # 0 major, 1 minor
    shift: int                # semitones applied by normalization
    piece_id: str

    @property
    def length(self) -> int:
        return int(self.valid.sum())


def phrase_feature(chorale: Chorale) -> np.ndarray:
    """Phrase-end marker, inferred from the soprano exactly as at inference time.

    The corpus has real fermatas, and using them would train the model on a
    feature no uploaded melody carries. Inferring the marker from note lengths
    in both settings keeps train and inference honest, at the cost of a noisier
    feature.
    """
    soprano = chorale.voices[0]
    onsets = chorale.onsets[0]
    flags = infer_phrase_ends(soprano, onsets, STEPS_PER_QUARTER)
    return np.array(flags, dtype=np.int64)


def metric_features(chorale: Chorale) -> tuple[np.ndarray, np.ndarray]:
    """Beat strength and position-in-measure, derived exactly as at inference.

    The pickup is INFERRED from the soprano rather than read from
    `chorale.pickup_steps`, even though the corpus knows the true value. A user
    melody carries no barlines, so training on the notated alignment and running
    on an inferred one is a silent distribution shift — measured at 41 of 61
    held-out chorales before this was shared.
    """
    numerator, denominator = chorale.time_signature
    steps_per_beat = max(1, int(round(STEPS_PER_QUARTER * 4 / denominator)))
    steps_per_measure = max(1, steps_per_beat * numerator)
    pickup = infer_pickup(chorale.voices[0], chorale.onsets[0], steps_per_measure)
    offset = (steps_per_measure - pickup) % steps_per_measure
    position = np.array(
        [min((t + offset) % steps_per_measure, MAX_POSITION - 1) for t in range(chorale.length)],
        dtype=np.int64,
    )
    metric = np.array([
        3 if (t + offset) % steps_per_measure == 0
        else 2 if (t + offset) % steps_per_beat == 0
        else 1 if (t + offset) % max(1, steps_per_beat // 2) == 0
        else 0
        for t in range(chorale.length)
    ], dtype=np.int64)
    return metric, position


def encode_chorale(chorale: Chorale, vocabulary: Vocabulary) -> Example:
    lines = chorale.normalized if vocabulary.encoding is Encoding.TONIC_RELATIVE else chorale.voices
    tokens = np.zeros((N_VOICES, chorale.length), dtype=np.int64)
    for voice in range(N_VOICES):
        vocab = vocabulary.voices[voice]
        tokens[voice] = [vocab.encode(pitch) for pitch in lines[voice]]
    metric, position = metric_features(chorale)
    return Example(
        tokens=tokens,
        valid=np.ones(chorale.length, dtype=bool),
        metric=metric,
        position=position,
        phrase=phrase_feature(chorale),
        mode=1 if chorale.key.is_minor else 0,
        shift=chorale.shift if vocabulary.encoding is Encoding.TONIC_RELATIVE else 0,
        piece_id=chorale.id,
    )


def transpose_example(example: Example, semitones: int, vocabulary: Vocabulary) -> Example | None:
    """Shift an ABSOLUTE-encoding example, or None if it leaves any voice's range.

    Only meaningful for `Encoding.ABSOLUTE`. Under a tonic-relative encoding
    transposition is a no-op by construction — normalising and then transposing
    just undoes the normalisation — which is the whole argument for the
    representation, so `transposition_variants` refuses to do it.
    """
    if semitones == 0:
        return example
    tokens = example.tokens.copy()
    for voice in range(N_VOICES):
        vocab = vocabulary.voices[voice]
        n_pitches = len(vocab.pitches)
        line = example.tokens[voice]
        pitched = line < n_pitches
        shifted = line[pitched] + semitones
        if shifted.size and (shifted.min() < 0 or shifted.max() >= n_pitches):
            return None
        tokens[voice][pitched] = shifted
    return Example(
        tokens=tokens,
        valid=example.valid,
        metric=example.metric,
        position=example.position,
        phrase=example.phrase,
        mode=example.mode,
        shift=example.shift,
        piece_id=f"{example.piece_id}+{semitones}",
    )


def transposition_variants(
    examples: Sequence[Example], vocabulary: Vocabulary, *, half_range: int
) -> list[list[Example]]:
    """Every in-range transposition of each example, grouped per piece.

    Used for *per-epoch* augmentation: one variant is drawn per piece per epoch,
    so an augmented run takes exactly as many gradient steps as an unaugmented
    one. Expanding the dataset instead would give augmentation more compute and
    make the ablation unreadable.

    Returns singletons under a tonic-relative encoding, because there is nothing
    to augment: normalising to a common tonic and then transposing away from it
    simply undoes the normalisation. That redundancy is the point — a
    tonic-relative model gets the benefit of all twelve keys for free, from the
    representation rather than from twelve times the compute.
    """
    if vocabulary.encoding is Encoding.TONIC_RELATIVE or half_range <= 0:
        return [[example] for example in examples]
    groups: list[list[Example]] = []
    for example in examples:
        variants = []
        for semitones in range(-half_range, half_range + 1):
            moved = transpose_example(example, semitones, vocabulary)
            if moved is not None:
                variants.append(moved)
        groups.append(variants or [example])
    return groups


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@dataclass
class Batch:
    tokens: np.ndarray        # (B, 4, T)
    valid: np.ndarray         # (B, T)
    metric: np.ndarray        # (B, T)
    position: np.ndarray      # (B, T)
    phrase: np.ndarray        # (B, T)
    mode: np.ndarray          # (B,)


def collate(examples: Sequence[Example]) -> Batch:
    length = max(example.tokens.shape[1] for example in examples)
    size = len(examples)
    tokens = np.zeros((size, N_VOICES, length), dtype=np.int64)
    valid = np.zeros((size, length), dtype=bool)
    metric = np.zeros((size, length), dtype=np.int64)
    position = np.zeros((size, length), dtype=np.int64)
    phrase = np.zeros((size, length), dtype=np.int64)
    mode = np.zeros(size, dtype=np.int64)
    for index, example in enumerate(examples):
        span = example.tokens.shape[1]
        tokens[index, :, :span] = example.tokens
        valid[index, :span] = example.valid
        metric[index, :span] = example.metric
        position[index, :span] = example.position
        phrase[index, :span] = example.phrase
        mode[index] = example.mode
    return Batch(tokens=tokens, valid=valid, metric=metric, position=position, phrase=phrase, mode=mode)


def sample_mask(
    valid: np.ndarray,
    rng: np.random.Generator,
    *,
    soprano_visible_probability: float = 0.5,
    free_voices: Sequence[int] = (1, 2, 3),
) -> np.ndarray:
    """Random orderless-NADE mask: True where a token is HIDDEN and must be predicted.

    The mask *rate* is drawn uniformly per example, which is what makes the
    trained model usable at every stage of blocked-Gibbs decoding — early sweeps
    condition on almost nothing, late sweeps on almost everything.

    With probability `soprano_visible_probability` the soprano is fully visible,
    matching the harmonization task, where the melody is always given. The rest
    of the time all four voices are maskable, which keeps the model a proper
    generative model over the whole texture rather than a conditional one.
    """
    size, length = valid.shape
    mask = np.zeros((size, N_VOICES, length), dtype=bool)
    for index in range(size):
        span = int(valid[index].sum())
        if span == 0:
            continue
        soprano_visible = rng.random() < soprano_visible_probability
        voices = list(free_voices) if soprano_visible else list(range(N_VOICES))
        total = len(voices) * span
        # Uniform over 1..total hidden sites, per the orderless-NADE objective.
        hidden = int(rng.integers(1, total + 1))
        flat = rng.permutation(total)[:hidden]
        for site in flat:
            voice = voices[site // span]
            step = site % span
            mask[index, voice, step] = True
    return mask


def apply_mask(tokens: np.ndarray, mask: np.ndarray, vocabulary: Vocabulary) -> np.ndarray:
    """Replace hidden tokens with each voice's MASK symbol."""
    out = tokens.copy()
    for voice in range(N_VOICES):
        out[:, voice, :][mask[:, voice, :]] = vocabulary.voices[voice].mask_id
    return out
