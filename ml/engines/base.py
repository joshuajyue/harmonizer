"""The HarmonyEngine interface — the seam between rule-based and learned harmonizers.

Every engine takes a melody and returns fully voiced parts. This is the central
correction to v1, where engines returned a chord label per beat and the actual
note choices were left to a fixed renderer. Voice leading *is* the problem, so it
has to live inside the engine where it can be optimized and evaluated.

Any engine registered here is automatically comparable in ml/eval — a rule engine
and a neural engine are scored by the identical harness on the identical metrics.
That head-to-head is the point of v2: v1 had no way to tell whether the model was
actually better than the rules, so "the rules win" was never measurable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from contracts.schema import Chord, KeySignature, Melody, Violation, Voice


@dataclass
class Harmonization:
    """An engine's full output: the voices are the product, chords are metadata."""

    key: KeySignature
    voices: list[Voice]
    chords: list[Chord] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)


class HarmonyEngine(ABC):
    """Base class for every harmonization strategy.

    Implementations MUST be deterministic when `seed` is fixed and `temperature`
    is 0, so eval runs are reproducible.
    """

    #: Stable identifier used by the API and the eval harness.
    id: str = "base"
    name: str = "Base Engine"
    description: str = ""
    #: False for rule systems, True for anything with learned parameters.
    learned: bool = False

    @abstractmethod
    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        """Generate accompanying voices for `melody`.

        The melody is conventionally retained as the soprano voice; the engine
        supplies the remaining `voice_count - 1` parts.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Whether this engine can currently run (e.g. its checkpoint exists)."""
        return True

    def log_likelihood(
        self, melody: Melody, voices: list[Voice]
    ) -> tuple[float, int] | None:
        """Total log-probability (nats) this engine assigns to `voices`, and the
        number of predicted tokens it covers.

        Optional. Probabilistic engines implement it so the eval harness can
        report held-out likelihood on real Bach — the one metric that measures
        the model rather than a proxy for it. Rule engines return None.
        """
        return None


_REGISTRY: dict[str, HarmonyEngine] = {}


def register(engine: HarmonyEngine) -> HarmonyEngine:
    """Register an engine instance so the API and eval harness can find it by id."""
    if engine.id in _REGISTRY:
        raise ValueError(f"Engine id {engine.id!r} is already registered")
    _REGISTRY[engine.id] = engine
    return engine


def get_engine(engine_id: str) -> HarmonyEngine:
    if engine_id not in _REGISTRY:
        raise KeyError(f"Unknown engine {engine_id!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[engine_id]


def all_engines() -> list[HarmonyEngine]:
    return list(_REGISTRY.values())
