"""Train the masked SATB model.

    python -m ml.training.train_neural                       # the real model
    python -m ml.training.train_neural --ablation            # + the v1 controls

The `--ablation` flag runs the experiment that actually tests the central claim
of the post-mortem. Same architecture, same data, same budget, three
representations:

  tonic_relative        every piece transposed so the tonic is C
  absolute              raw pitch, mode flag only — exactly v1's information
  absolute_augmented    raw pitch, plus 12-fold transposition augmentation

If tonic-relative beats absolute by a wide margin, "the representation deleted
the signal" stops being an assertion. If augmentation closes most of that gap,
that tells us the handicap was sample efficiency; if it does not, the handicap
was that the target was not a function of the input the model was given.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data.corpus import load_chorales, split_chorales  # noqa: E402
from ml.nn.encoding import (  # noqa: E402
    N_VOICES,
    Batch,
    Encoding,
    Example,
    Vocabulary,
    apply_mask,
    build_vocabulary,
    collate,
    encode_chorale,
    sample_mask,
    transposition_variants,
)
from ml.nn.model import MaskedSATBModel, ModelConfig, masked_cross_entropy  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "masked_satb.pt"


@dataclass
class TrainConfig:
    encoding: str = Encoding.TONIC_RELATIVE.value
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.3
    voice_embedding: int = 48
    batch_size: int = 16
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    max_epochs: int = 220
    patience: int = 25
    seed: int = 1234
    augment_semitones: int = 0
    soprano_visible_probability: float = 0.5
    max_length: int = 512


def to_tensors(batch: Batch) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.from_numpy(batch.tokens),
        "valid": torch.from_numpy(batch.valid),
        "metric": torch.from_numpy(batch.metric),
        "position": torch.from_numpy(batch.position),
        "phrase": torch.from_numpy(batch.phrase),
        "mode": torch.from_numpy(batch.mode),
    }


def clip_example(example: Example, max_length: int) -> Example:
    """Cap very long chorales so one 772-step outlier cannot dominate a batch.

    Unlike v1's fixed 32-step window this is a memory guard at 512 sixteenths
    (128 quarter notes), which is longer than every chorale but one, so in
    practice nothing is truncated at all.
    """
    span = example.tokens.shape[1]
    if span <= max_length:
        return example
    return Example(
        tokens=example.tokens[:, :max_length],
        valid=example.valid[:max_length],
        metric=example.metric[:max_length],
        position=example.position[:max_length],
        phrase=example.phrase[:max_length],
        mode=example.mode,
        shift=example.shift,
        piece_id=example.piece_id,
    )


def make_batches(examples: Sequence[Example], batch_size: int, rng: np.random.Generator, *, shuffle: bool) -> list[Batch]:
    order = rng.permutation(len(examples)) if shuffle else np.arange(len(examples))
    # Sort within a shuffled window so batches are length-homogeneous (less
    # padding) without becoming deterministic in content.
    window = batch_size * 8
    grouped: list[int] = []
    for start in range(0, len(order), window):
        chunk = sorted(order[start:start + window], key=lambda i: examples[i].tokens.shape[1])
        grouped.extend(chunk)
    return [collate([examples[i] for i in grouped[s:s + batch_size]]) for s in range(0, len(grouped), batch_size)]


def evaluate(
    model: MaskedSATBModel,
    examples: Sequence[Example],
    vocabulary: Vocabulary,
    *,
    seed: int,
    batch_size: int,
    repeats: int = 4,
    soprano_visible_probability: float = 1.0,
) -> tuple[float, float]:
    """Mean masked cross-entropy, plus the orderless-NADE NLL per token.

    Both are computed with the soprano fully visible by default, because that is
    the harmonization task: predict alto, tenor and bass given the melody.
    """
    model.eval()
    total_loss, total_count = 0.0, 0
    with torch.no_grad():
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat)
            for batch in make_batches(examples, batch_size, rng, shuffle=False):
                mask = sample_mask(
                    batch.valid, rng,
                    soprano_visible_probability=soprano_visible_probability,
                )
                masked_tokens = apply_mask(batch.tokens, mask, vocabulary)
                tensors = to_tensors(batch)
                logits = model(
                    torch.from_numpy(masked_tokens), tensors["metric"],
                    tensors["position"], tensors["phrase"], tensors["mode"],
                )
                loss, count = masked_cross_entropy(
                    logits, tensors["tokens"], torch.from_numpy(mask), tensors["valid"]
                )
                if count:
                    total_loss += float(loss) * count
                    total_count += count
    model.train()
    # Orderless-NADE estimator (Uria et al. 2014): with the number of hidden
    # sites drawn uniformly, D x (mean NLL over hidden sites) is an unbiased
    # estimate of the sequence NLL, so the mean itself is the per-token NLL.
    mean_loss = total_loss / max(1, total_count)
    return mean_loss, mean_loss


def train_one(config: TrainConfig, *, verbose: bool = True, tag: str = "") -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    chorales = load_chorales()
    train_pieces, val_pieces, test_pieces = split_chorales(chorales)
    encoding = Encoding(config.encoding)
    vocabulary = build_vocabulary(chorales, encoding)

    train_examples = [clip_example(encode_chorale(c, vocabulary), config.max_length) for c in train_pieces]
    val_examples = [clip_example(encode_chorale(c, vocabulary), config.max_length) for c in val_pieces]

    # Per-epoch augmentation: one transposition drawn per piece per epoch, so
    # every ablation arm takes the same number of gradient steps.
    variant_groups = transposition_variants(
        train_examples, vocabulary, half_range=config.augment_semitones
    )
    variants_per_piece = sum(len(g) for g in variant_groups) / max(1, len(variant_groups))

    model_config = ModelConfig(
        voice_sizes=tuple(vocabulary.sizes),
        voice_embedding=config.voice_embedding,
        hidden=config.hidden,
        layers=config.layers,
        dropout=config.dropout,
    )
    model = MaskedSATBModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)

    if verbose:
        print(f"[{tag or config.encoding}] {len(train_examples)} train / {len(val_examples)} val pieces, "
              f"{variants_per_piece:.1f} transpositions/piece, vocab {vocabulary.sizes}, "
              f"{model.parameter_count():,} params")

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    since_improvement = 0
    history: list[dict] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        rng = np.random.default_rng(config.seed * 1000 + epoch)
        epoch_examples = [group[int(rng.integers(len(group)))] for group in variant_groups]
        epoch_loss, epoch_count = 0.0, 0
        for batch in make_batches(epoch_examples, config.batch_size, rng, shuffle=True):
            mask = sample_mask(
                batch.valid, rng,
                soprano_visible_probability=config.soprano_visible_probability,
            )
            masked_tokens = apply_mask(batch.tokens, mask, vocabulary)
            tensors = to_tensors(batch)
            logits = model(
                torch.from_numpy(masked_tokens), tensors["metric"],
                tensors["position"], tensors["phrase"], tensors["mode"],
            )
            loss, count = masked_cross_entropy(
                logits, tensors["tokens"], torch.from_numpy(mask), tensors["valid"]
            )
            if count == 0:
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss) * count
            epoch_count += count

        train_loss = epoch_loss / max(1, epoch_count)
        val_loss, val_nll = evaluate(
            model, val_examples, vocabulary,
            seed=config.seed, batch_size=config.batch_size, repeats=2,
        )
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, since_improvement = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_improvement += 1

        if verbose and (epoch == 1 or epoch % 10 == 0 or since_improvement >= config.patience):
            print(f"  epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                  f"ppl {math.exp(val_loss):.3f}  best {best_loss:.4f}@{best_epoch}")
        if since_improvement >= config.patience:
            if verbose:
                print(f"  early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started

    final_val, final_nll = evaluate(
        model, val_examples, vocabulary, seed=config.seed + 99,
        batch_size=config.batch_size, repeats=6,
    )
    return {
        "config": asdict(config),
        "model_config": model_config.to_dict(),
        "vocabulary": vocabulary.to_dict(),
        "state_dict": model.state_dict(),
        "val_loss": final_val,
        "val_nll_per_token": final_nll,
        "val_perplexity": math.exp(final_val),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "seconds": elapsed,
        "params": model.parameter_count(),
        "train_examples": len(train_examples),
        "variants_per_piece": round(variants_per_piece, 2),
    }


def save(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": result["state_dict"],
        "model_config": result["model_config"],
        "vocabulary": result["vocabulary"],
        "train_config": result["config"],
        "val_loss": result["val_loss"],
        "val_nll_per_token": result["val_nll_per_token"],
        "val_perplexity": result["val_perplexity"],
        "best_epoch": result["best_epoch"],
    }, path)
    print(f"saved {path}  (val loss {result['val_loss']:.4f}, ppl {result['val_perplexity']:.3f})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoding", default=Encoding.TONIC_RELATIVE.value,
                        choices=[e.value for e in Encoding])
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--augment", type=int, default=0,
                        help="Transposition augmentation half-range in semitones (absolute encoding only).")
    parser.add_argument("--out", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--ablation", action="store_true",
                        help="Run the representation ablation instead of a single model.")
    args = parser.parse_args(argv)

    base = TrainConfig(
        encoding=args.encoding, hidden=args.hidden, layers=args.layers, dropout=args.dropout,
        max_epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        learning_rate=args.lr, seed=args.seed, augment_semitones=args.augment,
    )

    if not args.ablation:
        result = train_one(base)
        save(result, Path(args.out))
        return 0

    from dataclasses import replace

    runs = {
        "tonic_relative": replace(base, encoding=Encoding.TONIC_RELATIVE.value, augment_semitones=0),
        "absolute": replace(base, encoding=Encoding.ABSOLUTE.value, augment_semitones=0),
        "absolute_augmented": replace(base, encoding=Encoding.ABSOLUTE.value, augment_semitones=6),
        # NOTE: augmentation here is per-epoch resampling, so this arm sees the
        # same number of gradient steps as the others — only the data it sees
        # is more varied.
    }
    summary = {}
    for name, config in runs.items():
        print(f"\n=== {name} ===")
        result = train_one(config, tag=name)
        summary[name] = {
            "val_loss": result["val_loss"],
            "val_perplexity": result["val_perplexity"],
            "val_nll_per_token": result["val_nll_per_token"],
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
            "seconds": round(result["seconds"], 1),
            "params": result["params"],
            "train_examples": result["train_examples"],
            "variants_per_piece": result["variants_per_piece"],
        }
        save(result, MODEL_DIR / f"ablation_{name}.pt")

    path = MODEL_DIR / "ablation.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== representation ablation ===")
    for name, values in summary.items():
        print(f"  {name:20s} val_loss {values['val_loss']:.4f}  ppl {values['val_perplexity']:.3f}  "
              f"({values['train_examples']} pieces x {values['variants_per_piece']} transpositions, "
              f"{values['seconds']}s)")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
