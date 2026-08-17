# backend/model.py
"""Compact BiLSTM chord-tagging model.

Input:  (batch, 32, 14) melody features - one feature vector per quarter-note beat
Output: (batch, 32, 7)  chord-degree logits - one prediction per beat

A small 2-layer bidirectional LSTM has plenty of capacity for this task
(7-way per-beat classification, ~100 training pieces, 32-step sequences) and
trains in seconds on CPU.
"""
import torch
import torch.nn as nn

INPUT_DIM = 14
HIDDEN_DIM = 128
OUTPUT_DIM = 7
NUM_LAYERS = 2
DROPOUT = 0.3


class ChordLSTM(nn.Module):
    """BiLSTM sequence tagger: melody features in, per-beat chord logits out."""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        """x: (batch, seq_len, input_dim) -> logits: (batch, seq_len, output_dim)"""
        lstm_out, _ = self.lstm(x)
        return self.classifier(self.dropout(lstm_out))


def create_model():
    """Factory for a freshly-initialized ChordLSTM."""
    return ChordLSTM()


def load_model(model_path):
    """Load trained weights from disk and return the model in eval mode."""
    model = create_model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
