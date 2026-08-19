"""Reproduce v1's four failures and measure what each one cost.

The main ablation in `train_neural.py --ablation` compares pitch representations
for a model that predicts *pitches from pitches*, and finds little difference —
which is informative, but it is not v1's situation. v1's network saw only the
melody's ABSOLUTE pitch classes and was asked for TONIC-RELATIVE chord degrees.
The target was not a function of the input: two melodies with identical pitch
classes in different keys had different correct answers, and nothing in the
input distinguished them beyond a single `is_minor` bit.

So this script rebuilds v1's actual setup — its features, its BiLSTM, its label
extraction, its padding scheme — and flips one factor at a time:

    representation   absolute pitch classes  vs  tonic-relative pitch classes
    labels           7 diatonic triads       vs  root x quality
    sequence         v1 (truncate/pad to 32 beats, unmasked loss)
                                             vs  whole pieces with a mask

EVERY arm is evaluated identically: tonic-relative chord-root accuracy on all
real beats of held-out pieces. Only training differs, so the numbers are
directly comparable across label spaces and sequence schemes.

Two model-free measurements come first, because they bound what any model in
v1's setup could have achieved regardless of architecture: how much of the
corpus the seven-triad label space can actually represent, and how much of each
piece the 32-beat window let the model see.

    python -m ml.training.v1_diagnosis
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.corpus import REST, STEPS_PER_QUARTER, Chorale, load_chorales, split_chorales  # noqa: E402
from ml.theory.chords import analyze_chord  # noqa: E402
from ml.theory.pitch import Key  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "models" / "v1_diagnosis.json"

#: v1's constants, verbatim from `git show main:backend/data_processor.py`.
V1_SEQUENCE_LENGTH = 32
V1_FEATURE_DIM = 14
V1_NUM_CHORD_TYPES = 7

V1_MAJOR_CHORD_TONES = {
    0: [0, 4, 7], 1: [2, 5, 9], 2: [4, 7, 11], 3: [5, 9, 0],
    4: [7, 11, 2], 5: [9, 0, 4], 6: [11, 2, 5],
}
V1_MINOR_CHORD_TONES = {
    0: [0, 3, 7], 1: [2, 5, 8], 2: [3, 7, 10], 3: [5, 8, 0],
    4: [7, 11, 2], 5: [8, 0, 3], 6: [10, 2, 5],
}
#: Scale-degree root of each v1 class, so every arm can be scored on chord root.
V1_DEGREE_ROOT = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def v1_chord_degree(pitch_classes: set[int], key: Key) -> int:
    """v1's `extract_real_chord_labels` vote, reproduced exactly.

    Every seventh chord, secondary dominant, borrowed chord and suspension is
    projected onto whichever diatonic triad shares the most pitch classes.
    """
    table = V1_MINOR_CHORD_TONES if key.is_minor else V1_MAJOR_CHORD_TONES
    scale_degrees = [(pc - key.tonic) % 12 for pc in pitch_classes]
    best_degree, best_score = 0, float("-inf")
    for degree in range(V1_NUM_CHORD_TYPES):
        tones = table[degree]
        score = sum(1 if d in tones else -1 for d in scale_degrees)
        if score > best_score:
            best_score, best_degree = score, degree
    return best_degree


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

#: Chord classes for the rich label space: (tonic-relative root, quality).
RICH_QUALITIES = ("maj", "min", "dim", "dom7", "min7", "halfdim7", "dim7", "maj7", "aug", "minmaj7")


@dataclass
class Piece:
    features_absolute: np.ndarray   # (beats, 14)
    features_relative: np.ndarray   # (beats, 14)
    root: np.ndarray                # (beats,) tonic-relative chord root, the common metric
    v1_label: np.ndarray            # (beats,) 0-6
    rich_label: np.ndarray          # (beats,) index into RICH_QUALITIES x 12
    mode: str


def rich_index(relative_root: int, quality: str) -> int:
    return relative_root * len(RICH_QUALITIES) + RICH_QUALITIES.index(quality)


def rich_root(index: int) -> int:
    return index // len(RICH_QUALITIES)


def label_space_fidelity(chorales: Sequence[Chorale]) -> dict:
    """How much of the corpus v1's seven diatonic triads can represent.

    Model-free: for every beat, compare the true chord against what v1's
    +1/-1 vote projects it onto. Whatever this loses, no model trained on those
    labels can recover — it is an upper bound on v1's ceiling, not a training
    problem.
    """
    total = root_kept = exact_kept = seventh = applied = chromatic = 0
    for chorale in chorales:
        key = chorale.key
        for beat in range(chorale.length // STEPS_PER_QUARTER):
            start = beat * STEPS_PER_QUARTER
            sounding = [line[start] for line in chorale.voices if line[start] != REST]
            if not sounding:
                continue
            label = analyze_chord(sounding, key)
            if label is None:
                continue
            total += 1
            if label.is_seventh:
                seventh += 1
            if label.applied_to is not None:
                applied += 1
            table = V1_MINOR_CHORD_TONES if key.is_minor else V1_MAJOR_CHORD_TONES
            degree = v1_chord_degree({p % 12 for p in sounding}, key)
            projected_root = V1_DEGREE_ROOT[key.mode][degree]
            projected_tones = set(table[degree])
            if projected_root == label.relative_root:
                root_kept += 1
                if projected_tones == set(label.pitch_classes):
                    exact_kept += 1
            if not set(label.pitch_classes) <= set(V1_DEGREE_ROOT[key.mode]) | {
                (r + 3) % 12 for r in V1_DEGREE_ROOT[key.mode]
            }:
                chromatic += 1
    return {
        "beats": total,
        "root_preserved": round(root_kept / max(1, total), 4),
        "chord_preserved_exactly": round(exact_kept / max(1, total), 4),
        "true_seventh_chords": round(seventh / max(1, total), 4),
        "true_applied_chords": round(applied / max(1, total), 4),
    }


def sequence_coverage(chorales: Sequence[Chorale]) -> dict:
    """How much of the corpus v1's 32-beat window actually saw."""
    lengths = np.array([c.length // STEPS_PER_QUARTER for c in chorales])
    seen = np.minimum(lengths, V1_SEQUENCE_LENGTH)
    return {
        "pieces": int(len(lengths)),
        "median_beats": int(np.median(lengths)),
        "max_beats": int(lengths.max()),
        "pieces_shorter_than_window": int((lengths < V1_SEQUENCE_LENGTH).sum()),
        "padded_fraction_of_training_grid": round(
            float((V1_SEQUENCE_LENGTH - seen).sum()) / float(V1_SEQUENCE_LENGTH * len(lengths)), 4
        ),
        "mean_fraction_of_piece_seen": round(float((seen / lengths).mean()), 4),
        "fraction_of_corpus_never_seen": round(float((lengths - seen).sum()) / float(lengths.sum()), 4),
    }


def build_piece(chorale: Chorale) -> Piece | None:
    key = chorale.key
    beats = chorale.length // STEPS_PER_QUARTER
    if beats < 8:
        return None

    numerator = chorale.time_signature[0]
    absolute = np.zeros((beats, V1_FEATURE_DIM), dtype=np.float32)
    relative = np.zeros((beats, V1_FEATURE_DIM), dtype=np.float32)
    roots = np.zeros(beats, dtype=np.int64)
    v1_labels = np.zeros(beats, dtype=np.int64)
    rich_labels = np.zeros(beats, dtype=np.int64)
    is_minor = 1.0 if key.is_minor else 0.0

    for beat in range(beats):
        start = beat * STEPS_PER_QUARTER
        stop = start + STEPS_PER_QUARTER
        melody_pcs = {p % 12 for p in chorale.voices[0][start:stop] if p != REST}
        for pitch_class in melody_pcs:
            absolute[beat, pitch_class] = 1.0
            relative[beat, key.to_relative(pitch_class)] = 1.0
        strong = 1.0 if (beat % max(1, numerator) == 0) else 0.0
        absolute[beat, 12] = relative[beat, 12] = strong
        absolute[beat, 13] = relative[beat, 13] = is_minor

        sounding = [line[start] for line in chorale.voices if line[start] != REST]
        label = analyze_chord(sounding, key) if sounding else None
        if label is None:
            roots[beat] = roots[beat - 1] if beat else 0
            v1_labels[beat] = v1_labels[beat - 1] if beat else 0
            rich_labels[beat] = rich_labels[beat - 1] if beat else 0
            continue
        roots[beat] = label.relative_root
        v1_labels[beat] = v1_chord_degree({p % 12 for p in sounding}, key)
        quality = label.quality if label.quality in RICH_QUALITIES else "maj"
        rich_labels[beat] = rich_index(label.relative_root, quality)

    return Piece(absolute, relative, roots, v1_labels, rich_labels, key.mode)


# ---------------------------------------------------------------------------
# v1's model
# ---------------------------------------------------------------------------


class V1ChordLSTM(nn.Module):
    """v1's `ChordLSTM`, with only the output width made configurable."""

    def __init__(self, output_dim: int, input_dim: int = V1_FEATURE_DIM, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden * 2, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.classifier(self.dropout(out))


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass
class Arm:
    representation: str   # "absolute" | "tonic_relative"
    labels: str           # "v1_triads" | "rich"
    sequence: str         # "v1" | "masked"

    @property
    def name(self) -> str:
        return f"{self.representation}/{self.labels}/{self.sequence}"


def pack(pieces: Sequence[Piece], arm: Arm, *, for_training: bool) -> tuple[torch.Tensor, ...]:
    """Returns (features, labels, valid, roots).

    Training follows the arm's sequence scheme. Evaluation NEVER does: every arm
    is tested on complete pieces with a mask, so a model trained on 32-beat
    windows is not also graded on a shorter, easier test set.
    """
    label_field = "v1_label" if arm.labels == "v1_triads" else "rich_label"
    feature_field = "features_absolute" if arm.representation == "absolute" else "features_relative"

    truncate = for_training and arm.sequence == "v1"
    length = V1_SEQUENCE_LENGTH if truncate else max(piece.root.shape[0] for piece in pieces)

    features = np.zeros((len(pieces), length, V1_FEATURE_DIM), dtype=np.float32)
    labels = np.zeros((len(pieces), length), dtype=np.int64)
    valid = np.zeros((len(pieces), length), dtype=bool)
    roots = np.zeros((len(pieces), length), dtype=np.int64)

    for index, piece in enumerate(pieces):
        source = getattr(piece, feature_field)
        target = getattr(piece, label_field)
        span = min(source.shape[0], length)
        features[index, :span] = source[:span]
        labels[index, :span] = target[:span]
        roots[index, :span] = piece.root[:span]
        valid[index, :span] = True
        # v1 labelled the padded region as chord 0 (the tonic) and took an
        # unmasked mean. `valid` still records the truth, so the honest metric
        # can be computed for this arm too.
        if truncate:
            labels[index, span:] = 0
            roots[index, span:] = 0

    return (
        torch.from_numpy(features), torch.from_numpy(labels),
        torch.from_numpy(valid), torch.from_numpy(roots),
    )


def label_to_root(labels: torch.Tensor, arm: Arm, modes: Sequence[str]) -> torch.Tensor:
    """Map any arm's class indices onto tonic-relative chord roots."""
    out = torch.zeros_like(labels)
    if arm.labels == "rich":
        return labels // len(RICH_QUALITIES)
    for index, mode in enumerate(modes):
        table = torch.tensor(V1_DEGREE_ROOT[mode], dtype=torch.long)
        out[index] = table[labels[index].clamp(0, 6)]
    return out


def run_arm(
    arm: Arm,
    train_pieces: Sequence[Piece],
    val_pieces: Sequence[Piece],
    test_pieces: Sequence[Piece],
    *,
    epochs: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_classes = V1_NUM_CHORD_TYPES if arm.labels == "v1_triads" else 12 * len(RICH_QUALITIES)
    model = V1ChordLSTM(n_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05, reduction="none")

    train_x, train_y, train_valid, _ = pack(train_pieces, arm, for_training=True)
    val_x, val_y, val_valid, _ = pack(val_pieces, arm, for_training=True)
    test_x, test_y, test_valid, test_roots = pack(test_pieces, arm, for_training=False)
    test_modes = [piece.mode for piece in test_pieces]

    def loss_for(logits, targets, valid):
        raw = criterion(logits.reshape(-1, n_classes), targets.reshape(-1)).reshape(targets.shape)
        if arm.sequence == "v1":
            return raw.mean()          # v1: no mask, padding counts
        mask = valid.float()
        return (raw * mask).sum() / mask.sum().clamp(min=1)

    def honest_root_accuracy(x, y, valid, roots, modes) -> float:
        model.eval()
        with torch.no_grad():
            predicted = model(x).argmax(dim=-1)
        model.train()
        predicted_roots = label_to_root(predicted, arm, modes)
        correct = (predicted_roots == roots) & valid
        return float(correct.sum()) / float(valid.sum().clamp(min=1))

    best_val, best_state, since = float("inf"), None, 0
    batch_size = 16
    for epoch in range(1, epochs + 1):
        order = torch.randperm(train_x.shape[0])
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_for(model(train_x[index]), train_y[index], train_valid[index])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_for(model(val_x), val_y, val_valid))
        model.train()
        if val_loss < best_val - 1e-4:
            best_val, since = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if since >= 20:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        predicted = model(test_x).argmax(dim=-1)
    honest_label_accuracy = float(((predicted == test_y) & test_valid).sum()) / float(test_valid.sum())
    unmasked_label_accuracy = float((predicted == test_y).sum()) / float(test_y.numel())
    root_accuracy = honest_root_accuracy(test_x, test_y, test_valid, test_roots, test_modes)

    # Accuracy restricted to the first 32 beats, i.e. the only region a v1-style
    # model ever trained on. The gap against the full-piece number is the cost
    # of the truncation.
    head = slice(0, V1_SEQUENCE_LENGTH)
    head_valid = test_valid[:, head]
    head_roots = label_to_root(predicted[:, head], arm, test_modes)
    head_accuracy = float(((head_roots == test_roots[:, head]) & head_valid).sum()) / float(head_valid.sum())

    return {
        "arm": arm.name,
        "representation": arm.representation,
        "labels": arm.labels,
        "sequence": arm.sequence,
        "classes": n_classes,
        "root_accuracy": round(root_accuracy, 4),
        "root_accuracy_first_32_beats": round(head_accuracy, 4),
        "label_accuracy_honest": round(honest_label_accuracy, 4),
        "label_accuracy_unmasked": round(unmasked_label_accuracy, 4),
        "epochs": epoch,
        "val_loss": round(best_val, 4),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(OUTPUT))
    args = parser.parse_args(argv)

    chorales = load_chorales()
    train, val, test = split_chorales(chorales)
    build = lambda pieces: [p for p in (build_piece(c) for c in pieces) if p is not None]  # noqa: E731
    train_pieces, val_pieces, test_pieces = build(train), build(val), build(test)
    print(f"{len(train_pieces)} train / {len(val_pieces)} val / {len(test_pieces)} test pieces\n")

    fidelity = label_space_fidelity(chorales)
    coverage = sequence_coverage(chorales)
    print("label-space fidelity (model-free):", json.dumps(fidelity))
    print("sequence coverage   (model-free):", json.dumps(coverage), "\n")

    results = []
    combinations = list(itertools.product(
        ("absolute", "tonic_relative"), ("v1_triads", "rich"), ("v1", "masked"),
    ))
    for representation, labels, padding in combinations:
        arm = Arm(representation, labels, padding)
        result = run_arm(arm, train_pieces, val_pieces, test_pieces, epochs=args.epochs, seed=args.seed)
        results.append(result)
        print(f"  {arm.name:42s} root_acc {result['root_accuracy']:.3f}  "
              f"(first 32 beats {result['root_accuracy_first_32_beats']:.3f})  "
              f"own-space acc {result['label_accuracy_honest']:.3f}")

    Path(args.out).write_text(json.dumps(
        {"label_space_fidelity": fidelity, "sequence_coverage": coverage, "arms": results}, indent=2,
    ) + "\n")
    print(f"\nwrote {args.out}")

    baseline = next(r for r in results if r["arm"] == "absolute/v1_triads/v1")
    fixed = next(r for r in results if r["arm"] == "tonic_relative/rich/masked")
    print(f"\nv1's configuration:      root accuracy {baseline['root_accuracy']:.3f}")
    print(f"all four failures fixed: root accuracy {fixed['root_accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
