"""The learned engine: blocked-Gibbs harmonization with the masked SATB model.

Decoding is iterative on purpose. v1 took an independent per-beat softmax and an
argmax, so nothing it predicted at beat 3 could ever be revised once beat 7
existed, and it had no transition model to make that revision with. Here the
model conditions every prediction on the whole rest of the texture in both
directions, and decoding repeatedly erases part of its own output and rewrites
it — the Coconet/DeepBach recipe. The harmonic grammar is not hand-written and
not absent; it lives in a model that gets to see its own answer.

Import stays cheap: torch and the checkpoint load on first use, so the backend
can import this module whether or not a model has been trained.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from contracts.schema import Chord, KeySignature, Melody, Violation, Voice

from ..data.corpus import REST, STEP
from ..data.melody import (
    MelodyGrid,
    detect_melody_key,
    grid_to_voices,
    melody_to_grid,
    select_voices,
    voices_to_grid,
)
from ..theory.chords import analyze_chord
from ..theory.pitch import Key, normalization_shift
from ..theory.voicing import VOICE_NAMES, analyze_texture, texture_from_voices
from .base import Harmonization, HarmonyEngine, register

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_CHECKPOINT = MODEL_DIR / "masked_satb.pt"

N_VOICES = 4
FREE_VOICES = (1, 2, 3)

#: Only used if an internal draft is requested without the originating melody,
#: which the public path never does. Harmonization is tempo-invariant: every
#: engine works in quarter-note beats and none reads this value.
DEFAULT_DRAFT_TEMPO = 90.0

#: Blocked-Gibbs schedule, following Coconet. The mask rate anneals from almost
#: everything to almost nothing, so early sweeps rough out the harmony and later
#: ones polish individual notes.
DEFAULT_SWEEPS = 64
MASK_RATE_START = 0.9
MASK_RATE_END = 0.05
ANNEAL_FRACTION = 0.7


@dataclass
class _Loaded:
    model: object
    vocabulary: object
    encoding: object
    metadata: dict


class NeuralHarmonyEngine(HarmonyEngine):
    """Masked SATB model decoded by annealed blocked Gibbs."""

    id = "neural"
    name = "Masked SATB Model (learned)"
    description = (
        "A bidirectional masked model over all four voices, trained on tonic-relative "
        "Bach chorales with an orderless objective and decoded by annealed blocked Gibbs. "
        "Predicts the actual notes each voice sings, not a chord label."
    )
    learned = True

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        *,
        sweeps: int = DEFAULT_SWEEPS,
        mask_rate_start: float = MASK_RATE_START,
        polish_rounds: int = 0,
        rule_weight: float = 1.0,
        engine_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.sweeps = sweeps
        self.mask_rate_start = mask_rate_start
        self.polish_rounds = polish_rounds
        self.rule_weight = rule_weight
        if engine_id:
            self.id = engine_id
        if name:
            self.name = name
        self._loaded: _Loaded | None = None
        self._lock = threading.Lock()

    # -- loading -----------------------------------------------------------

    def is_available(self) -> bool:
        return self.checkpoint.exists()

    def _load(self) -> _Loaded:
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            import torch

            from ..nn.encoding import Encoding, Vocabulary
            from ..nn.model import MaskedSATBModel, ModelConfig

            payload = torch.load(self.checkpoint, map_location="cpu")
            model = MaskedSATBModel(ModelConfig.from_dict(payload["model_config"]))
            model.load_state_dict(payload["state_dict"])
            model.eval()
            vocabulary = Vocabulary.from_dict(payload["vocabulary"])
            self._loaded = _Loaded(
                model=model,
                vocabulary=vocabulary,
                encoding=Encoding(payload["vocabulary"]["encoding"]),
                metadata={k: v for k, v in payload.items() if k != "state_dict"},
            )
        return self._loaded

    # -- features ----------------------------------------------------------

    def _context(self, grid: MelodyGrid, key: Key) -> dict:
        """Metric, phrase and mode features, computed exactly as in training."""
        from ..nn.encoding import MAX_POSITION

        steps_per_measure = grid.steps_per_measure
        offset = (steps_per_measure - grid.pickup_steps) % steps_per_measure
        position = np.array(
            [min((t + offset) % steps_per_measure, MAX_POSITION - 1) for t in range(grid.length)],
            dtype=np.int64,
        )
        return {
            "metric": np.array(grid.beat_strength, dtype=np.int64)[None, :],
            "position": position[None, :],
            "phrase": np.array([1 if flag else 0 for flag in grid.phrase_end], dtype=np.int64)[None, :],
            "mode": np.array([1 if key.is_minor else 0], dtype=np.int64),
        }

    def _shift(self, key: Key, grid: MelodyGrid | None = None) -> int:
        from ..nn.encoding import Encoding, register_correction

        loaded = self._load()
        if loaded.encoding is not Encoding.TONIC_RELATIVE:
            return 0
        shift = normalization_shift(key)
        if grid is None:
            return shift
        sounding = [p for p in grid.pitches if p != REST]
        if not sounding:
            return shift
        median = float(np.median(np.array(sounding, dtype=np.float64))) + shift
        return shift + register_correction(median)

    # -- decoding ----------------------------------------------------------

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        grid = melody_to_grid(melody)
        if melody.key is not None:
            key, confidence = Key(melody.key.tonic, melody.key.mode), melody.key.confidence or 1.0
        else:
            key, confidence = detect_melody_key(grid)

        if grid.length == 0:
            empty, names = select_voices([[], [], [], []], voice_count)
            return Harmonization(
                key=KeySignature(tonic=key.tonic, mode=key.mode, confidence=confidence),
                voices=grid_to_voices(empty, names=names),
            )

        lines = self._decode(grid, key, temperature=temperature, seed=seed, melody=melody)
        chords = self._chords(lines, key, grid.steps_per_beat)
        violations = self._violations(lines, key, grid.steps_per_beat)
        selected, names = select_voices(lines, voice_count)
        return Harmonization(
            key=KeySignature(tonic=key.tonic, mode=key.mode, confidence=confidence),
            voices=grid_to_voices(selected, names=names),
            chords=chords,
            violations=violations,
        )

    def _decode(
        self,
        grid: MelodyGrid,
        key: Key,
        *,
        temperature: float,
        seed: int | None,
        initial: Sequence[Sequence[int]] | None = None,
        melody: Melody | None = None,
    ) -> list[list[int]]:
        import torch

        loaded = self._load()
        vocabulary = loaded.vocabulary
        shift = self._shift(key, grid)
        length = grid.length
        context = self._context(grid, key)
        rng = np.random.default_rng(0 if seed is None else seed)

        tokens = np.zeros((1, N_VOICES, length), dtype=np.int64)
        soprano = [REST if p == REST else p + shift for p in grid.pitches]
        tokens[0, 0] = [vocabulary.voices[0].encode(p) for p in soprano]
        for voice in FREE_VOICES:
            vocab = vocabulary.voices[voice]
            if initial is not None and voice < len(initial):
                seeded = [REST if p == REST else p + shift for p in initial[voice]]
                seeded = (seeded + [REST] * length)[:length]
                tokens[0, voice] = [vocab.encode(p) for p in seeded]
            else:
                tokens[0, voice] = vocab.mask_id

        # A rest in the melody means silence in every part; fix those sites so
        # Gibbs never has to spend sweeps rediscovering it.
        rest_steps = np.array([p == REST for p in grid.pitches], dtype=bool)
        for voice in FREE_VOICES:
            tokens[0, voice, rest_steps] = vocabulary.voices[voice].rest_id

        free_sites = [
            (voice, step)
            for voice in FREE_VOICES
            for step in range(length)
            if not rest_steps[step]
        ]
        if not free_sites:
            return [list(grid.pitches)] + [[REST] * length for _ in FREE_VOICES]

        tensors = {
            "metric": torch.from_numpy(context["metric"]),
            "position": torch.from_numpy(context["position"]),
            "phrase": torch.from_numpy(context["phrase"]),
            "mode": torch.from_numpy(context["mode"]),
        }
        base_temperature = temperature if temperature > 0 else 1.0
        total = len(free_sites)
        site_array = np.array(free_sites)

        with torch.no_grad():
            for sweep in range(self.sweeps):
                progress = min(1.0, sweep / max(1e-9, ANNEAL_FRACTION * self.sweeps))
                rate = 1.0 if sweep == 0 and initial is None else max(
                    MASK_RATE_END, self.mask_rate_start * (1.0 - progress)
                )
                hidden_count = max(1, int(round(rate * total)))
                hidden = site_array[rng.permutation(total)[:hidden_count]]

                masked = tokens.copy()
                for voice in FREE_VOICES:
                    rows = hidden[hidden[:, 0] == voice][:, 1]
                    masked[0, voice, rows] = vocabulary.voices[voice].mask_id

                logits = loaded.model(torch.from_numpy(masked), **tensors)
                # Anneal to argmax: the last sweeps are deterministic refinement,
                # which is what makes temperature=0 a fixed point rather than a
                # coin flip, while the early sampled sweeps avoid the degenerate
                # mode that pure argmax from an empty texture falls into.
                step_temperature = base_temperature * max(0.0, 1.0 - progress)
                for voice in FREE_VOICES:
                    rows = hidden[hidden[:, 0] == voice][:, 1]
                    if rows.size == 0:
                        continue
                    voice_logits = logits[voice][0, rows].clone()
                    # Never emit the MASK symbol as a note, and never drop a
                    # voice out where the melody is sounding: both are hard
                    # facts about the output, not preferences to be learned.
                    voice_logits[:, vocabulary.voices[voice].mask_id] = -1e9
                    voice_logits[:, vocabulary.voices[voice].rest_id] = -1e9
                    if step_temperature <= 1e-6:
                        picked = torch.argmax(voice_logits, dim=-1).numpy()
                    else:
                        probabilities = torch.softmax(voice_logits / step_temperature, dim=-1).numpy()
                        # Vectorised inverse-CDF sampling: a Python loop over a
                        # few hundred rows per sweep dominates the runtime.
                        thresholds = rng.random((probabilities.shape[0], 1))
                        picked = (probabilities.cumsum(axis=1) < thresholds).sum(axis=1)
                        picked = np.minimum(picked, probabilities.shape[1] - 1)
                    tokens[0, voice, rows] = picked

        if self.polish_rounds:
            tokens = self._polish(
                tokens, tensors, rest_steps, loaded, vocabulary, length,
            )

        lines = [list(grid.pitches)]
        for voice in FREE_VOICES:
            vocab = vocabulary.voices[voice]
            decoded = [vocab.decode(int(token)) for token in tokens[0, voice]]
            lines.append([REST if p == REST else p - shift for p in decoded])
        return lines

    def _polish(self, tokens, tensors, rest_steps, loaded, vocabulary, length):
        """Coordinate ascent: re-choose each voice's whole line by Viterbi.

        With one voice fully masked the model's logits for it are independent of
        its own current notes, so its line can be re-solved exactly against the
        other three under the voice-leading rulebook. The model keeps every
        harmonic decision; the rules only rule out illegal realisations of it.
        """
        import torch

        from ._polish import polish_voice

        pitch_tables = {
            voice: np.array(vocabulary.voices[voice].pitches, dtype=np.int64)
            for voice in FREE_VOICES
        }
        active = ~rest_steps

        with torch.no_grad():
            for _ in range(self.polish_rounds):
                for voice in FREE_VOICES:
                    masked = tokens.copy()
                    masked[0, voice, :] = vocabulary.voices[voice].mask_id
                    logits = loaded.model(torch.from_numpy(masked), **tensors)[voice][0]
                    n_pitches = len(vocabulary.voices[voice].pitches)
                    log_probs = torch.log_softmax(logits[:, :n_pitches], dim=-1).numpy().astype(np.float64)

                    fixed = np.full((4, length), -1, dtype=np.int64)
                    for other in range(4):
                        table = np.array(vocabulary.voices[other].pitches, dtype=np.int64)
                        row = tokens[0, other]
                        pitched = row < table.shape[0]
                        fixed[other, pitched] = table[row[pitched]]

                    chosen = polish_voice(
                        log_probs, pitch_tables[voice], fixed, voice,
                        active=active, rule_weight=self.rule_weight,
                    )
                    lookup = {int(p): i for i, p in enumerate(pitch_tables[voice])}
                    for t in range(length):
                        if not active[t]:
                            continue
                        tokens[0, voice, t] = lookup[int(chosen[t])]
        return tokens

    # -- likelihood --------------------------------------------------------

    def log_likelihood(
        self,
        melody: Melody,
        voices: list[Voice],
        *,
        repeats: int = 8,
        seed: int = 0,
    ) -> tuple[float, int] | None:
        """Orderless-NADE estimate of log p(alto, tenor, bass | soprano).

        Returns (total log-probability in nats, number of predicted tokens).
        With the number of hidden sites drawn uniformly, D x (mean NLL over
        hidden sites) is an unbiased estimate of the sequence NLL (Uria et al.
        2014), so the per-token figure the harness reports is exactly that mean.
        """
        if not self.is_available():
            return None
        import torch

        loaded = self._load()
        vocabulary = loaded.vocabulary
        grid = melody_to_grid(melody)
        if grid.length == 0:
            return None
        key = Key(melody.key.tonic, melody.key.mode) if melody.key else detect_melody_key(grid)[0]
        shift = self._shift(key, grid)
        lines = voices_to_grid(voices, length=grid.length)
        while len(lines) < N_VOICES:
            lines.append([REST] * grid.length)

        tokens = np.zeros((1, N_VOICES, grid.length), dtype=np.int64)
        for voice in range(N_VOICES):
            vocab = vocabulary.voices[voice]
            moved = [REST if p == REST else p + shift for p in lines[voice]]
            tokens[0, voice] = [vocab.encode(p) for p in moved]

        context = self._context(grid, key)
        tensors = {
            "metric": torch.from_numpy(context["metric"]),
            "position": torch.from_numpy(context["position"]),
            "phrase": torch.from_numpy(context["phrase"]),
            "mode": torch.from_numpy(context["mode"]),
        }
        rng = np.random.default_rng(seed)
        total = len(FREE_VOICES) * grid.length
        accumulated = 0.0

        with torch.no_grad():
            for _ in range(repeats):
                hidden_count = int(rng.integers(1, total + 1))
                chosen = rng.permutation(total)[:hidden_count]
                mask = np.zeros((N_VOICES, grid.length), dtype=bool)
                for site in chosen:
                    mask[FREE_VOICES[site // grid.length], site % grid.length] = True

                masked = tokens.copy()
                for voice in FREE_VOICES:
                    masked[0, voice, mask[voice]] = vocabulary.voices[voice].mask_id
                logits = loaded.model(torch.from_numpy(masked), **tensors)

                summed, count = 0.0, 0
                for voice in FREE_VOICES:
                    rows = np.nonzero(mask[voice])[0]
                    if rows.size == 0:
                        continue
                    summed += float(torch.nn.functional.cross_entropy(
                        logits[voice][0, rows], torch.from_numpy(tokens[0, voice, rows]), reduction="sum",
                    ))
                    count += int(rows.size)
                if count:
                    accumulated += -total * (summed / count)

        return accumulated / repeats, total

    # -- output ------------------------------------------------------------

    def _chords(self, lines: Sequence[Sequence[int]], key: Key, steps_per_beat: int) -> list[Chord]:
        out: list[Chord] = []
        length = len(lines[0])
        for start in range(0, length, steps_per_beat):
            pitches = [line[start] for line in lines if start < len(line) and line[start] != REST]
            label = analyze_chord(pitches, key) if pitches else None
            if label is None:
                continue
            roman = label.roman(key.mode)
            span = round(min(steps_per_beat, length - start) * STEP, 6)
            if out and out[-1].roman == roman and math.isclose(
                out[-1].start + out[-1].duration, round(start * STEP, 6), abs_tol=1e-6
            ):
                out[-1] = out[-1].model_copy(update={"duration": round(out[-1].duration + span, 6)})
                continue
            out.append(Chord(
                start=round(start * STEP, 6), duration=span, roman=roman,
                root=label.absolute_root(key), quality=label.contract_quality(),
                inversion=label.inversion,
                secondaryOf=None if label.applied_to is None else key.to_absolute(label.applied_to),
                # Response-side fields are always populated, never left to a
                # default, so the UI never has to null-check. These engines write
                # common-practice chorale harmony: no upper extensions, and no
                # reharmonization, so there is no substitution to explain.
                extensions=[],
                substitutionOf=None,
                substitutionKind=None,
            ))
        return out

    def _violations(self, lines: Sequence[Sequence[int]], key: Key, steps_per_beat: int) -> list[Violation]:
        from ..eval.metrics import step_chords

        texture = texture_from_voices([[None if p == REST else p for p in line] for line in lines], step=STEP)
        chords = step_chords(lines, key, steps_per_beat=steps_per_beat)
        return [
            Violation(
                kind=defect.kind, severity=defect.severity, start=defect.offset,
                voices=[VOICE_NAMES[v] for v in defect.voices if v < 4], message=defect.message,
            )
            for defect in analyze_texture(texture, key, chords)
            if defect.severity != "info"
        ]


class NeuralRefinementEngine(NeuralHarmonyEngine):
    """The learned model started from the rule engine's answer instead of nothing.

    Blocked Gibbs from an all-masked start has to invent the whole texture from
    the melody alone in its first sweep and can settle into a poor mode. Seeding
    it with the rule engine's voicing turns the same model into a *reviser*: the
    rules supply a valid draft with a coherent harmonic skeleton, and the model
    rewrites whatever it disagrees with.

    Whether this beats either parent is an empirical question the harness
    answers, and it is the most honest way to combine them — no scores are
    blended, one system simply edits the other's output.
    """

    id = "neural_refine"
    name = "Rule draft, learned revision"
    description = (
        "The rule engine's SATB draft, then annealed blocked Gibbs under the learned "
        "model. Combines a valid harmonic skeleton with learned voice leading."
    )
    learned = True

    #: A refiner must not demolish the draft it was given: the schedule starts
    #: by erasing about a third of the texture rather than nearly all of it.
    REFINE_MASK_RATE_START = 0.35

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        *,
        sweeps: int = 16,
        polish_rounds: int = 2,
        rule_weight: float = 1.0,
    ) -> None:
        super().__init__(
            checkpoint,
            sweeps=sweeps,
            mask_rate_start=self.REFINE_MASK_RATE_START,
            polish_rounds=polish_rounds,
            rule_weight=rule_weight,
        )
        self._rules = None

    def _rule_engine(self):
        if self._rules is None:
            from .rules import RuleHarmonyEngine

            self._rules = RuleHarmonyEngine()
        return self._rules

    def _decode(self, grid: MelodyGrid, key: Key, *, temperature: float, seed: int | None,
                initial=None, melody: Melody | None = None):
        if initial is None:
            initial = self._draft(grid, key, melody)
        return super()._decode(
            grid, key, temperature=temperature, seed=seed, initial=initial, melody=melody,
        )

    def _draft(self, grid: MelodyGrid, key: Key, melody: Melody | None = None) -> list[list[int]]:
        from contracts.schema import KeySignature as _KeySignature
        from contracts.schema import Melody as _Melody
        from contracts.schema import Note as _Note
        from contracts.schema import TimeSignature as _TimeSignature

        notes: list[_Note] = []
        t = 0
        while t < grid.length:
            pitch = grid.pitches[t]
            if pitch == REST:
                t += 1
                continue
            run = t + 1
            while run < grid.length and grid.pitches[run] == pitch and not grid.onsets[run]:
                run += 1
            notes.append(_Note(pitch=pitch, start=round(t * STEP, 6), duration=round((run - t) * STEP, 6)))
            t = run
        if not notes:
            return [[REST] * grid.length for _ in range(N_VOICES)]

        # A faithful re-expression of the caller's melody, quantized to the grid
        # — not a synthetic one. Tempo is carried through rather than defaulted:
        # no engine reads it today, but fabricating a constant here is exactly
        # how a silently-wrong tempo would enter the system later.
        draft = self._rule_engine().harmonize(
            _Melody(
                notes=notes,
                tempo=melody.tempo if melody is not None else DEFAULT_DRAFT_TEMPO,
                timeSignature=_TimeSignature(
                    numerator=grid.time_signature[0], denominator=grid.time_signature[1]
                ),
                key=_KeySignature(tonic=key.tonic, mode=key.mode),
            ),
            voice_count=4,
        )
        return voices_to_grid(draft.voices, length=grid.length)


class ConstrainedNeuralEngine(NeuralHarmonyEngine):
    """The learned model with the voice-leading rulebook as a veto.

    Same model and same blocked-Gibbs decode as `neural`, followed by
    constrained coordinate ascent (see `_polish.py`). This is the deployable
    engine: it keeps the model's harmonic vocabulary and removes the parallel
    fifths and octaves that blocked Gibbs cannot see, because independent
    resampling has no way to notice two voices about to move together.

    `neural` is kept registered alongside it, unpolished, as the control — the
    difference between the two rows in the report is exactly what the veto buys.
    """

    id = "neural_vl"
    name = "Learned harmony, enforced counterpoint"
    description = (
        "The learned model decoded by blocked Gibbs, then each voice re-solved by "
        "Viterbi under the voice-leading rules. The model makes every harmonic "
        "decision; the rules only veto illegal ways of realising it."
    )
    learned = True

    #: Chosen by sweeping against the harness (ml/experiments/defect_style_tradeoff.py),
    #: not guessed. The result was that there is essentially no trade-off: going
    #: from 0 to 0.15 removes every parallel fifth and octave while *improving*
    #: style divergence and chord variety, because the constraint acts on how a
    #: chord is realised, not on which chord is chosen. Anything in 0.15-1.0 is
    #: within noise; 0.5 has the best chord variety of those and leaves the most
    #: room for the model's own preference while still binding.
    DEFAULT_RULE_WEIGHT = 0.5

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        *,
        sweeps: int = 24,
        polish_rounds: int = 2,
        rule_weight: float | None = None,
    ) -> None:
        super().__init__(
            checkpoint,
            sweeps=sweeps,
            polish_rounds=polish_rounds,
            rule_weight=self.DEFAULT_RULE_WEIGHT if rule_weight is None else rule_weight,
        )


register(NeuralHarmonyEngine())
register(ConstrainedNeuralEngine())
register(NeuralRefinementEngine())
