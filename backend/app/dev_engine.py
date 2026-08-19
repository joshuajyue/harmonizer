"""A registry-compatible pass-through engine for local smoke tests only."""

from contracts.schema import KeySignature, Melody, Voice
from ml.engines.base import Harmonization, HarmonyEngine, all_engines, register


class DevelopmentStubEngine(HarmonyEngine):
    id = "dev-stub"
    name = "Development Stub"
    description = "Passes the melody through as soprano; not a harmonizer."
    learned = False

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        del voice_count, temperature, seed
        key = melody.key or KeySignature(tonic=0, mode="major", confidence=0.0)
        return Harmonization(
            key=key,
            voices=[Voice(name="soprano", notes=melody.notes)],
        )


def register_development_engine() -> None:
    if not any(engine.id == DevelopmentStubEngine.id for engine in all_engines()):
        register(DevelopmentStubEngine())
