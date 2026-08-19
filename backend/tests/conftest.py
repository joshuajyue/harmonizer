from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.engines import EngineService
from contracts.schema import Chord, KeySignature, Melody, Voice
from ml.engines.base import Harmonization, HarmonyEngine


class StubEngine(HarmonyEngine):
    id = "stub"
    name = "Stub Engine"
    description = "Deterministic test engine"
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
        return Harmonization(
            key=melody.key or KeySignature(tonic=0, mode="major", confidence=1.0),
            chords=[
                Chord(
                    start=0,
                    duration=1,
                    roman="I",
                    root=0,
                    quality="maj",
                )
            ],
            voices=[Voice(name="soprano", notes=melody.notes)],
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    app = create_app(
        settings=Settings(),
        engine_service=EngineService(
            engines=[StubEngine()],
            discover_modules=False,
        ),
    )
    with TestClient(app) as test_client:
        yield test_client
