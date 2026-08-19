"""A registry-compatible canonical-fixture engine for local smoke tests."""

from functools import lru_cache
from pathlib import Path

from contracts.schema import HarmonizeResponse, Melody
from ml.engines.base import Harmonization, HarmonyEngine, all_engines, register

_RESPONSE_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "contracts"
    / "examples"
    / "harmonize.response.json"
)


@lru_cache(maxsize=1)
def _canonical_response() -> HarmonizeResponse:
    return HarmonizeResponse.model_validate_json(_RESPONSE_FIXTURE.read_text())


class DevelopmentStubEngine(HarmonyEngine):
    id = "dev-stub"
    name = "Development Stub"
    description = "Returns the canonical SATB fixture; not a harmonizer."
    learned = False

    def harmonize(
        self,
        melody: Melody,
        *,
        voice_count: int = 4,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Harmonization:
        del melody, voice_count, temperature, seed
        fixture = _canonical_response().model_copy(deep=True)
        return Harmonization(
            key=fixture.key,
            chords=fixture.chords,
            voices=fixture.voices,
            violations=fixture.violations,
        )

    def is_available(self) -> bool:
        return _RESPONSE_FIXTURE.is_file()


def register_development_engine() -> None:
    if not any(engine.id == DevelopmentStubEngine.id for engine in all_engines()):
        register(DevelopmentStubEngine())
