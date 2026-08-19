"""The `jazz_reharm` engine: rules skeleton in, reharmonized jazz out.

Registered through the same seam as every other engine, so it appears in
`GET /api/v1/engines` and in the A/B comparison UI with no change to
`backend/` or `web/`. It implements the existing `HarmonyEngine` interface —
melody in, voiced parts out — and calls the rules engine internally for the
functional skeleton, so no API change is required either.

Two engines are registered, not one, because the comparison is the finding:

  * `jazz_reharm` — learned. Jazz chord model plus the substitution vocabulary,
    sampled at a temperature. Different every seed.
  * `jazz_reharm_rules` — the hand-written substitution vocabulary chosen by
    Viterbi argmax. Deterministic, and a genuinely strong baseline.

Putting both in the registry means anyone can hear the difference the report
argues about instead of taking its word for it.

Temperature is the contract's dial and it means what the contract says: 0 is
argmax and is exactly reproducible. It is also, per the whole premise of this
workstream, the least interesting setting — the argmax of a reharmonization
distribution is the safe reharmonization. The recommended setting is around 1.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from contracts.schema import Chord, KeySignature, Melody, Violation, Voice

from ..engines.base import Harmonization, HarmonyEngine, register
from .chords import JazzChord, classify_melody_note
from .data import ChordSpan
from .metrics import MelodyNote, note_weight
from .model import ChordNGram
from .search import (
    HybridScorer,
    ReharmConfig,
    RuleScorer,
    build_lattice,
    realize,
    sample,
    viterbi,
)
from .skeleton import Skeleton, melody_notes, skeleton_from_rules
from .voicing import DEFAULT_STYLE, VoicingStyle, build_voices


def _to_contract_chord(
    span: ChordSpan,
    tonic: int,
    mode: str,
    following: JazzChord | None,
) -> Chord:
    chord = span.chord
    secondary: int | None = None
    if chord.is_dominant and following is not None:
        interval = (chord.root - following.root) % 12
        if interval in (7, 1) and following.root != tonic:
            secondary = following.root
    return Chord(
        start=round(span.start, 6),
        duration=round(span.duration, 6),
        roman=chord.roman(tonic, mode),
        root=chord.root,
        quality=chord.quality,
        inversion=chord.inversion,
        secondaryOf=secondary,
        extensions=list(chord.extensions),
        substitutionOf=chord.substitution_of,
        substitutionKind=chord.substitution_kind,
    )


def _violations(spans: Sequence[ChordSpan], melody: Sequence[MelodyNote], tonic: int, mode: str) -> list[Violation]:
    """What the listener might object to, said out loud.

    Chorale violations do not apply — parallel fifths are an idiom here — so
    what is reported instead is the two things that actually go wrong in a
    reharmonization: a chord that cannot support the melody over it, and a
    substitution that fails to resolve the way its own logic promises.
    """
    out: list[Violation] = []
    for index, span in enumerate(spans):
        for start, pitch, duration in melody:
            overlap = min(start + duration, span.stop) - max(start, span.start)
            if overlap <= 1e-6:
                continue
            verdict = classify_melody_note(span.chord, pitch % 12)
            if verdict.verdict != "conflict":
                continue
            if note_weight(max(start, span.start), overlap) < 0.75:
                continue
            out.append(Violation(
                kind="melody_conflict",
                severity="warning",
                start=round(max(start, span.start), 6),
                voices=["soprano"],
                message=(
                    f"melody sounds {verdict.interval} semitones above the root of "
                    f"{span.chord.symbol()}, a semitone above its "
                    f"{'third' if verdict.against == span.chord.third_interval else 'chord tone'}"
                ),
            ))
        following = spans[index + 1].chord if index + 1 < len(spans) else None
        if (
            span.chord.substitution_kind == "tritone"
            and following is not None
            and (span.chord.root - following.root) % 12 != 1
        ):
            out.append(Violation(
                kind="unresolved_substitution",
                severity="info",
                start=round(span.start, 6),
                voices=["bass"],
                message=(
                    f"{span.chord.symbol()} is a tritone substitute but does not resolve "
                    f"down a semitone into {following.symbol()}"
                ),
            ))
    return out


class _JazzReharmBase(HarmonyEngine):
    """Shared pipeline: skeleton -> lattice -> path -> voicing."""

    config: ReharmConfig = ReharmConfig()
    style: VoicingStyle = DEFAULT_STYLE

    def _skeleton(self, melody: Melody) -> Skeleton:
        return skeleton_from_rules(melody)

    def _empty(self, melody: Melody, skeleton: Skeleton) -> Harmonization:
        return Harmonization(key=skeleton.key, voices=[Voice(name="soprano", notes=list(melody.notes))])

    def _finish(
        self,
        melody: Melody,
        skeleton: Skeleton,
        spans: Sequence[ChordSpan],
        voice_count: int,
    ) -> Harmonization:
        origin = min((note.start for note in melody.notes), default=0.0)
        voices = build_voices(
            spans,
            skeleton.melody,
            voice_count=voice_count,
            style=self.style,
            beats_per_bar=skeleton.meter[0],
            origin=origin,
        )
        chords = [
            _to_contract_chord(
                span,
                skeleton.tonic,
                skeleton.mode,
                spans[index + 1].chord if index + 1 < len(spans) else None,
            )
            for index, span in enumerate(spans)
        ]
        for chord in chords:
            chord.start = round(chord.start + origin, 6)
        return Harmonization(
            key=skeleton.key,
            voices=voices,
            chords=chords,
            violations=_violations(spans, skeleton.melody, skeleton.tonic, skeleton.mode),
        )


class JazzReharmEngine(_JazzReharmBase):
    """Learned jazz reharmonization: sampled, not searched."""

    id = "jazz_reharm"
    name = "Jazz Reharmonization (learned)"
    description = (
        "Runs the functional rules engine for a diatonic skeleton, then reharmonizes it with "
        "tritone substitutions, backdoor and secondary dominants, modal interchange, passing "
        "diminished and ii-V insertions, scored by a chord model trained on the Jazz Harmony "
        "Treebank and the Weimar Jazz Database. Sampled at a temperature, so every seed is a "
        "different valid reharmonization; melody compatibility is enforced during generation. "
        "Voiced with rootless, quartal and upper-structure jazz voicings."
    )
    learned = True

    def __init__(self, config: ReharmConfig | None = None) -> None:
        self.config = config or ReharmConfig()
        self._model: ChordNGram | None = None
        self._loaded = False

    def model(self) -> ChordNGram | None:
        if not self._loaded:
            self._model = ChordNGram.load()
            self._loaded = True
        return self._model

    def is_available(self) -> bool:
        return self.model() is not None

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        model = self.model()
        if model is None:
            raise RuntimeError("jazz_reharm model missing: run `python -m ml.reharm.model`")
        skeleton = self._skeleton(melody)
        if not skeleton.units:
            return self._empty(melody, skeleton)

        config = replace(self.config, temperature=max(0.0, temperature))
        lattice = build_lattice(skeleton, config)
        scorer = HybridScorer(lattice, model, config)
        if temperature <= 0.0:
            path = viterbi(lattice, scorer)
        else:
            path = sample(
                lattice,
                scorer,
                temperature=temperature,
                top_p=config.top_p,
                seed=0 if seed is None else seed,
            )
        result = realize(lattice, path, skeleton)
        return self._finish(melody, skeleton, result.spans, voice_count)


class JazzReharmRulesEngine(_JazzReharmBase):
    """Hand-written substitution vocabulary, chosen by search. The baseline."""

    id = "jazz_reharm_rules"
    name = "Jazz Reharmonization (rules)"
    description = (
        "The same substitution vocabulary and the same hard melody constraints as the learned "
        "engine, but chosen by Viterbi argmax over hand-written functional scores with no "
        "learned parameters. Deterministic: one tune, one answer. Built as an honest baseline "
        "for the learned engine, and it is not an easy one to beat."
    )
    learned = False

    def __init__(self, config: ReharmConfig | None = None) -> None:
        self.config = config or ReharmConfig()

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        skeleton = self._skeleton(melody)
        if not skeleton.units:
            return self._empty(melody, skeleton)
        lattice = build_lattice(skeleton, self.config)
        result = realize(lattice, viterbi(lattice, RuleScorer(lattice, self.config)), skeleton)
        return self._finish(melody, skeleton, result.spans, voice_count)


JAZZ_REHARM = register(JazzReharmEngine())
JAZZ_REHARM_RULES = register(JazzReharmRulesEngine())
