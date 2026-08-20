"""The learned harmonizer: a masked-token model over all four voices.

Design follows directly from the v1 post-mortem rather than from wanting a
bigger network.

**Model the voices, not chord labels.** v1 ran `chordify()` on the corpus and
predicted one of seven diatonic triads per beat, which throws away the
counterpoint that is the whole reason the corpus is valuable, and uses a label
space that cannot represent what Bach wrote. Here the target is the actual
pitch each voice sings; chord labels are derived from the result afterwards, so
sevenths, inversions, applied chords and chromaticism are representable because
nothing was ever projected onto a small alphabet.

**Model transitions explicitly.** v1 did independent per-beat softmax and
argmax, so it was handed no harmonic grammar and could not induce one from a
polluted objective. Here every prediction is conditioned on the entire rest of
the texture in both directions through a bidirectional recurrence, and decoding
is iterative: the model repeatedly revises its own output, so a choice at beat 3
can be reconsidered once beat 7 exists. That is the Coconet/DeepBach recipe, and
it is a well-validated fit for a corpus of ~400 pieces on a laptop CPU.

**Train the orderless objective.** A random subset of the texture is hidden and
predicted from the rest, with the hidden *fraction* drawn uniformly. That single
model is then valid at every stage of blocked-Gibbs decoding, from "nothing is
written yet" to "one note is being revised", and it yields a principled held-out
likelihood estimate rather than an accuracy against lossy labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .encoding import MAX_POSITION, N_VOICES


@dataclass
class ModelConfig:
    voice_sizes: tuple[int, ...]
    voice_embedding: int = 48
    context_embedding: int = 16
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.3

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["voice_sizes"] = list(self.voice_sizes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelConfig":
        payload = dict(payload)
        payload["voice_sizes"] = tuple(payload["voice_sizes"])
        return cls(**payload)


class MaskedSATBModel(nn.Module):
    """Predicts every hidden (voice, time) pitch from the whole visible texture.

    Input  : (B, 4, T) tokens where hidden entries carry each voice's MASK id,
             plus metric position, phrase marker and mode.
    Output : one logit vector per voice per timestep, over that voice's pitch
             alphabet.

    One shared trunk with a per-voice output head, rather than DeepBach's four
    separate networks: with ~260 training chorales the voices share far more
    statistics than they have data to learn separately.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.voice_embeddings = nn.ModuleList(
            nn.Embedding(size, config.voice_embedding) for size in config.voice_sizes
        )
        self.metric_embedding = nn.Embedding(4, config.context_embedding)
        self.position_embedding = nn.Embedding(MAX_POSITION, config.context_embedding)
        self.phrase_embedding = nn.Embedding(2, config.context_embedding)
        self.mode_embedding = nn.Embedding(2, config.context_embedding)

        input_dim = N_VOICES * config.voice_embedding + 4 * config.context_embedding
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, config.hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.rnn = nn.LSTM(
            config.hidden,
            config.hidden,
            num_layers=config.layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.heads = nn.ModuleList(nn.Linear(2 * config.hidden, size) for size in config.voice_sizes)

    def forward(
        self,
        tokens: torch.Tensor,     # (B, 4, T)
        metric: torch.Tensor,     # (B, T)
        position: torch.Tensor,   # (B, T)
        phrase: torch.Tensor,     # (B, T)
        mode: torch.Tensor,       # (B,)
    ) -> list[torch.Tensor]:
        batch, _, length = tokens.shape
        parts = [self.voice_embeddings[v](tokens[:, v, :]) for v in range(N_VOICES)]
        parts.append(self.metric_embedding(metric))
        parts.append(self.position_embedding(position))
        parts.append(self.phrase_embedding(phrase))
        parts.append(self.mode_embedding(mode).unsqueeze(1).expand(batch, length, -1))
        hidden = self.input_projection(torch.cat(parts, dim=-1))
        encoded, _ = self.rnn(hidden)
        encoded = self.dropout(encoded)
        return [head(encoded) for head in self.heads]

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_cross_entropy(
    logits: list[torch.Tensor],
    targets: torch.Tensor,      # (B, 4, T)
    hidden_mask: torch.Tensor,  # (B, 4, T) bool — True where prediction is required
    valid: torch.Tensor,        # (B, T) bool — False on padding
) -> tuple[torch.Tensor, int]:
    """Cross-entropy over hidden, non-padded positions only.

    The `valid` term is the direct fix for v1's largest scoring bug: it padded
    every piece to 32 steps, labelled the padding as tonic, and then took an
    unmasked mean, so the model was rewarded for predicting I on nothing and the
    reported validation accuracy counted those positions as correct.
    """
    total = torch.zeros((), dtype=torch.float32, device=targets.device)
    count = 0
    valid_expanded = valid.unsqueeze(1).expand_as(hidden_mask)
    score_mask = hidden_mask & valid_expanded
    for voice in range(N_VOICES):
        selected = score_mask[:, voice, :]
        if not bool(selected.any()):
            continue
        voice_logits = logits[voice][selected]
        voice_targets = targets[:, voice, :][selected]
        total = total + nn.functional.cross_entropy(voice_logits, voice_targets, reduction="sum")
        count += int(selected.sum())
    if count == 0:
        return total, 0
    return total / count, count
