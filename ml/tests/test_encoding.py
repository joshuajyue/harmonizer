"""Tests for the neural encoding layer.

The first two classes are direct regression tests for v1's two structural bugs:
padding that polluted the loss, and an input representation that did not contain
the information the target was expressed in. Both are the kind of mistake that
produces a plausible-looking training curve and a model that cannot work, so
they are pinned here rather than trusted.
"""

import numpy as np
import pytest
import torch

from ml.data.corpus import REST, Chorale
from ml.nn.encoding import (
    Encoding,
    Example,
    VoiceVocab,
    Vocabulary,
    apply_mask,
    build_vocabulary,
    collate,
    encode_chorale,
    sample_mask,
    transpose_example,
    transposition_variants,
)
from ml.nn.model import MaskedSATBModel, ModelConfig, masked_cross_entropy
from ml.theory.pitch import Key


def make_chorale(voices, key=Key(0, "major"), shift=0, length=None) -> Chorale:
    length = length or len(voices[0])
    return Chorale(
        id="test",
        key=key,
        voices=[list(v) for v in voices],
        onsets=[[True] * length for _ in voices],
        fermatas=[False] * length,
        beat_strength=[3 if t % 4 == 0 else 0 for t in range(length)],
        time_signature=(4, 4),
        shift=shift,
    )


SIMPLE_VOCAB = Vocabulary(
    voices=[VoiceVocab(list(range(50, 90))) for _ in range(4)],
    encoding=Encoding.TONIC_RELATIVE,
)


class TestPaddingIsExcluded:
    """v1 padded to 32 steps, labelled the padding tonic, and took an unmasked
    mean, so the loss rewarded predicting I on nothing and val_acc counted those
    positions as correct."""

    def test_collate_marks_padding_invalid(self):
        short = Example(
            tokens=np.zeros((4, 5), dtype=np.int64), valid=np.ones(5, dtype=bool),
            metric=np.zeros(5, dtype=np.int64), position=np.zeros(5, dtype=np.int64),
            phrase=np.zeros(5, dtype=np.int64), mode=0, shift=0, piece_id="short",
        )
        long = Example(
            tokens=np.zeros((4, 12), dtype=np.int64), valid=np.ones(12, dtype=bool),
            metric=np.zeros(12, dtype=np.int64), position=np.zeros(12, dtype=np.int64),
            phrase=np.zeros(12, dtype=np.int64), mode=0, shift=0, piece_id="long",
        )
        batch = collate([short, long])
        assert batch.valid.shape == (2, 12)
        assert batch.valid[0, :5].all() and not batch.valid[0, 5:].any()
        assert batch.valid[1].all()

    def test_loss_ignores_padded_positions(self):
        """The regression test for v1's largest scoring bug.

        Corrupt every target in the padded region. If padding leaked into the
        loss the number would move; it must not.
        """
        torch.manual_seed(0)
        model = MaskedSATBModel(ModelConfig(voice_sizes=(12, 12, 12, 12), hidden=16, layers=1, dropout=0.0))
        model.eval()
        length = 10
        tokens = torch.randint(0, 10, (1, 4, length))
        valid = torch.zeros(1, length, dtype=torch.bool)
        valid[0, :4] = True
        hidden = torch.ones(1, 4, length, dtype=torch.bool)
        context = {
            "metric": torch.zeros(1, length, dtype=torch.long),
            "position": torch.zeros(1, length, dtype=torch.long),
            "phrase": torch.zeros(1, length, dtype=torch.long),
            "mode": torch.zeros(1, dtype=torch.long),
        }
        logits = model(tokens, **context)

        clean, count = masked_cross_entropy(logits, tokens, hidden, valid)
        corrupted_tokens = tokens.clone()
        corrupted_tokens[:, :, 4:] = (corrupted_tokens[:, :, 4:] + 5) % 10
        dirty, dirty_count = masked_cross_entropy(logits, corrupted_tokens, hidden, valid)

        assert count == dirty_count == 4 * 4
        assert torch.allclose(clean, dirty)

    def test_loss_counts_only_hidden_positions(self):
        torch.manual_seed(0)
        model = MaskedSATBModel(ModelConfig(voice_sizes=(12, 12, 12, 12), hidden=16, layers=1, dropout=0.0))
        model.eval()
        tokens = torch.randint(0, 10, (1, 4, 8))
        valid = torch.ones(1, 8, dtype=torch.bool)
        hidden = torch.zeros(1, 4, 8, dtype=torch.bool)
        hidden[0, 1, 3] = True
        context = {
            "metric": torch.zeros(1, 8, dtype=torch.long),
            "position": torch.zeros(1, 8, dtype=torch.long),
            "phrase": torch.zeros(1, 8, dtype=torch.long),
            "mode": torch.zeros(1, dtype=torch.long),
        }
        _, count = masked_cross_entropy(model(tokens, **context), tokens, hidden, valid)
        assert count == 1

    def test_sample_mask_never_touches_padding(self):
        valid = np.zeros((3, 20), dtype=bool)
        valid[0, :20] = True
        valid[1, :7] = True
        valid[2, :1] = True
        rng = np.random.default_rng(0)
        for _ in range(30):
            mask = sample_mask(valid, rng)
            for row in range(3):
                span = int(valid[row].sum())
                assert not mask[row, :, span:].any()

    def test_sample_mask_always_hides_something(self):
        valid = np.ones((4, 16), dtype=bool)
        rng = np.random.default_rng(1)
        for _ in range(30):
            assert sample_mask(valid, rng).any(axis=(1, 2)).all()

    def test_soprano_visibility_probability_is_respected(self):
        valid = np.ones((200, 8), dtype=bool)
        rng = np.random.default_rng(2)
        mask = sample_mask(valid, rng, soprano_visible_probability=1.0)
        assert not mask[:, 0, :].any()
        mask = sample_mask(valid, rng, soprano_visible_probability=0.0)
        assert mask[:, 0, :].any()


class TestTonicRelativeRepresentation:
    """v1 fed absolute pitch classes and trained on tonic-relative targets."""

    def test_same_degrees_in_different_keys_encode_identically(self):
        # A I-V-I in C and the same progression in E must produce the same
        # tokens. This is the property that lets one model serve all 12 keys.
        c_major = make_chorale(
            [[72, 71, 72], [67, 67, 67], [64, 62, 64], [48, 55, 48]], key=Key(0, "major"), shift=0,
        )
        e_major = make_chorale(
            [[76, 75, 76], [71, 71, 71], [68, 66, 68], [52, 59, 52]], key=Key(4, "major"), shift=-4,
        )
        a = encode_chorale(c_major, SIMPLE_VOCAB)
        b = encode_chorale(e_major, SIMPLE_VOCAB)
        assert np.array_equal(a.tokens, b.tokens)

    def test_absolute_encoding_does_not_collapse_keys(self):
        absolute = Vocabulary(voices=[VoiceVocab(list(range(30, 95))) for _ in range(4)], encoding=Encoding.ABSOLUTE)
        c_major = make_chorale([[72, 71, 72], [67, 67, 67], [64, 62, 64], [48, 55, 48]], key=Key(0, "major"), shift=0)
        e_major = make_chorale([[76, 75, 76], [71, 71, 71], [68, 66, 68], [52, 59, 52]], key=Key(4, "major"), shift=-4)
        a = encode_chorale(c_major, absolute)
        b = encode_chorale(e_major, absolute)
        assert not np.array_equal(a.tokens, b.tokens)

    def test_rests_survive_encoding(self):
        chorale = make_chorale([[72, REST, 72], [67, REST, 67], [64, REST, 64], [48, REST, 48]])
        example = encode_chorale(chorale, SIMPLE_VOCAB)
        for voice in range(4):
            assert example.tokens[voice, 1] == SIMPLE_VOCAB.voices[voice].rest_id

    def test_mode_is_carried(self):
        major = encode_chorale(make_chorale([[72]] * 4, key=Key(0, "major")), SIMPLE_VOCAB)
        minor = encode_chorale(make_chorale([[72]] * 4, key=Key(0, "minor")), SIMPLE_VOCAB)
        assert (major.mode, minor.mode) == (0, 1)


class TestVocabulary:
    def test_roundtrip(self):
        vocab = VoiceVocab(list(range(60, 72)))
        for pitch in range(60, 72):
            assert vocab.decode(vocab.encode(pitch)) == pitch
        assert vocab.decode(vocab.encode(REST)) == REST

    def test_out_of_range_pitch_snaps_by_octave(self):
        vocab = VoiceVocab(list(range(60, 72)))
        assert vocab.decode(vocab.encode(48)) == 60   # two octaves below -> C4
        assert vocab.decode(vocab.encode(38)) == 62   # D1 -> D4, octave-equivalent
        assert vocab.decode(vocab.encode(84)) == 60   # C6 -> C4
        # Whatever it snaps to must be a legal member of the alphabet.
        for pitch in range(20, 110):
            assert vocab.decode(vocab.encode(pitch)) in range(60, 72)

    def test_mask_and_rest_are_distinct_and_outside_the_pitches(self):
        vocab = VoiceVocab(list(range(60, 72)))
        assert vocab.rest_id != vocab.mask_id
        assert vocab.size == 14
        assert vocab.decode(vocab.mask_id) == REST

    def test_serialization_roundtrip(self):
        payload = SIMPLE_VOCAB.to_dict()
        restored = Vocabulary.from_dict(payload)
        assert restored.sizes == SIMPLE_VOCAB.sizes
        assert restored.encoding is SIMPLE_VOCAB.encoding

    def test_built_vocabulary_covers_the_corpus(self):
        chorales = [
            make_chorale([[72, 74], [67, 69], [60, 62], [48, 50]]),
            make_chorale([[79, 81], [72, 74], [64, 66], [40, 43]]),
        ]
        vocab = build_vocabulary(chorales, Encoding.ABSOLUTE, margin=0)
        assert vocab.voices[0].pitches[0] == 72 and vocab.voices[0].pitches[-1] == 81
        assert vocab.voices[3].pitches[0] == 40 and vocab.voices[3].pitches[-1] == 50


class TestAugmentation:
    def test_transposition_shifts_every_pitched_token(self):
        absolute = Vocabulary(voices=[VoiceVocab(list(range(40, 90))) for _ in range(4)], encoding=Encoding.ABSOLUTE)
        chorale = make_chorale([[72, 74], [67, 69], [60, 62], [48, 50]])
        example = encode_chorale(chorale, absolute)
        moved = transpose_example(example, 3, absolute)
        assert moved is not None
        for voice in range(4):
            for original, shifted in zip(example.tokens[voice], moved.tokens[voice]):
                assert absolute.voices[voice].decode(shifted) == absolute.voices[voice].decode(original) + 3

    def test_transposition_rejects_out_of_range(self):
        absolute = Vocabulary(voices=[VoiceVocab(list(range(60, 65))) for _ in range(4)], encoding=Encoding.ABSOLUTE)
        chorale = make_chorale([[64], [64], [64], [64]])
        example = encode_chorale(chorale, absolute)
        assert transpose_example(example, 6, absolute) is None

    def test_transposition_preserves_rests(self):
        absolute = Vocabulary(voices=[VoiceVocab(list(range(40, 90))) for _ in range(4)], encoding=Encoding.ABSOLUTE)
        chorale = make_chorale([[72, REST], [67, REST], [60, REST], [48, REST]])
        example = encode_chorale(chorale, absolute)
        moved = transpose_example(example, 2, absolute)
        for voice in range(4):
            assert moved.tokens[voice, 1] == absolute.voices[voice].rest_id

    def test_augmentation_is_a_noop_under_tonic_relative(self):
        """The central claim about the representation, as an assertion.

        Under a tonic-relative encoding, transposing and then re-normalising
        returns the original piece, so transposition augmentation buys nothing.
        The twelve keys come free from the representation instead of from twelve
        times the compute.
        """
        chorale = make_chorale([[72, 74], [67, 69], [60, 62], [48, 50]])
        example = encode_chorale(chorale, SIMPLE_VOCAB)
        groups = transposition_variants([example], SIMPLE_VOCAB, half_range=6)
        assert [len(g) for g in groups] == [1]

    def test_augmentation_expands_under_absolute(self):
        absolute = Vocabulary(voices=[VoiceVocab(list(range(30, 100))) for _ in range(4)], encoding=Encoding.ABSOLUTE)
        chorale = make_chorale([[72, 74], [67, 69], [60, 62], [48, 50]])
        example = encode_chorale(chorale, absolute)
        groups = transposition_variants([example], absolute, half_range=3)
        assert len(groups[0]) == 7


class TestMaskApplication:
    def test_apply_mask_uses_each_voices_own_mask_symbol(self):
        vocab = Vocabulary(
            voices=[VoiceVocab(list(range(60, 60 + n))) for n in (4, 5, 6, 7)],
            encoding=Encoding.TONIC_RELATIVE,
        )
        tokens = np.zeros((1, 4, 3), dtype=np.int64)
        mask = np.zeros((1, 4, 3), dtype=bool)
        mask[0, :, 1] = True
        out = apply_mask(tokens, mask, vocab)
        for voice in range(4):
            assert out[0, voice, 1] == vocab.voices[voice].mask_id
            assert out[0, voice, 0] == 0

    def test_apply_mask_does_not_mutate_input(self):
        tokens = np.zeros((1, 4, 3), dtype=np.int64)
        mask = np.ones((1, 4, 3), dtype=bool)
        apply_mask(tokens, mask, SIMPLE_VOCAB)
        assert (tokens == 0).all()
