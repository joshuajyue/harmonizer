from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.engines import EngineService
from contracts.schema import HarmonizeRequest, HarmonizeResponse, Melody
from ml.engines.base import Harmonization, HarmonyEngine

EXAMPLES = Path(__file__).resolve().parents[2] / "contracts" / "examples"
CANONICAL_REQUEST = HarmonizeRequest.model_validate_json(
    (EXAMPLES / "melody.request.json").read_text()
)
CANONICAL_RESPONSE = HarmonizeResponse.model_validate_json(
    (EXAMPLES / "harmonize.response.json").read_text()
)


class StubEngine(HarmonyEngine):
    id = "rules"
    name = "Canonical Fixture Stub"
    description = "Returns the shared schema-valid SATB fixture"
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
        fixture = CANONICAL_RESPONSE.model_copy(deep=True)
        return Harmonization(
            key=fixture.key,
            chords=fixture.chords,
            voices=fixture.voices,
            violations=fixture.violations,
        )


@pytest.fixture
def canonical_request_payload() -> dict:
    return CANONICAL_REQUEST.model_dump(mode="json")


@pytest.fixture
def canonical_response_payload() -> dict:
    return CANONICAL_RESPONSE.model_dump(mode="json")


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
