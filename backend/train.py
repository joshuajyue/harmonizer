# backend/train.py
"""Trains the Bach-chorale chord model and saves models/chord_harmonizer.pt.

Usage:
    python train.py
"""
import os
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from data_processor import MeasureBasedChordProcessor, NUM_CHORD_TYPES
from model import create_model

MODEL_PATH = os.path.join("models", "chord_harmonizer.pt")
MAX_PIECES = 400
BATCH_SIZE = 16
MAX_EPOCHS = 250
PATIENCE = 30
LEARNING_RATE = 1e-3


class BachDataset(Dataset):
    """Wraps processed (melody features, chord labels) pairs for PyTorch's DataLoader."""

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return torch.FloatTensor(example["input"]), torch.FloatTensor(example["target"])


def compute_class_weights(dataset):
    """Mild class weights (sqrt-dampened inverse frequency, capped at 2x) so the loss
    doesn't ignore rare chords, without overcorrecting into ignoring the common ones."""
    counts = Counter()
    for _, target in dataset:
        counts.update(torch.argmax(target, dim=-1).tolist())

    total = sum(counts.values())
    weights = torch.ones(NUM_CHORD_TYPES)
    for chord_degree in range(NUM_CHORD_TYPES):
        count = counts.get(chord_degree, 1)
        weights[chord_degree] = min((total / (NUM_CHORD_TYPES * count)) ** 0.5, 2.0)
    return weights


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs)
            target_idx = torch.argmax(targets, dim=-1)
            loss = criterion(logits.reshape(-1, NUM_CHORD_TYPES), target_idx.reshape(-1))
            total_loss += loss.item() * inputs.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == target_idx).sum().item()
            total += target_idx.numel()
    return total_loss / len(loader.dataset), correct / total


def train_chord_model():
    print("Loading and featurizing Bach chorales...")
    processor = MeasureBasedChordProcessor()
    examples = processor.process_bach_chorales(max_pieces=MAX_PIECES)
    if len(examples) < 10:
        print("Not enough training data found; aborting.")
        return

    dataset = BachDataset(examples)
    val_size = max(1, int(len(dataset) * 0.15))
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    class_weights = compute_class_weights(train_set)
    print(f"Train/val split: {len(train_set)}/{len(val_set)} pieces")
    print(f"Class weights: {[round(w, 2) for w in class_weights.tolist()]}")

    model = create_model()
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            target_idx = torch.argmax(targets, dim=-1)
            loss = criterion(logits.reshape(-1, NUM_CHORD_TYPES), target_idx.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}")

        if epochs_without_improvement >= PATIENCE:
            print(f"No improvement for {PATIENCE} epochs, stopping early at epoch {epoch}.")
            break

    os.makedirs("models", exist_ok=True)
    torch.save(best_state, MODEL_PATH)
    print(f"Saved best model (val_loss={best_val_loss:.4f}) to {MODEL_PATH}")


if __name__ == "__main__":
    train_chord_model()
