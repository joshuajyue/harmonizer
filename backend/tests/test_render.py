import sys
import wave
from io import BytesIO

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.engines import EngineService
from backend.tests.conftest import StubEngine


def render_request(synth: str = "sf2") -> dict:
    return {
        "voices": [
            {
                "name": "soprano",
                "notes": [
                    {
                        "pitch": 69,
                        "start": 0,
                        "duration": 0.25,
                        "velocity": 90,
                    }
                ],
            }
        ],
        "tempo": 120,
        "synth": synth,
    }


def test_sf2_render_always_returns_valid_wav(client: TestClient) -> None:
    response = client.post("/api/v1/render", json=render_request())

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-harmonaizer-synth-used"] == "sf2"
    with wave.open(BytesIO(response.content), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getnframes() > 0


def test_unconfigured_ddsp_falls_back_transparently(client: TestClient) -> None:
    response = client.post("/api/v1/render", json=render_request("ddsp"))

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-synth-requested"] == "ddsp"
    assert response.headers["x-harmonaizer-synth-used"] == "sf2"
    assert "x-harmonaizer-fallback" in response.headers


def test_synth_capabilities_expose_optional_neural_tier(client: TestClient) -> None:
    response = client.get("/api/v1/synths")

    assert response.status_code == 200
    synths = {item["id"]: item for item in response.json()["synths"]}
    assert synths["sf2"]["available"] is True
    assert synths["ddsp"]["available"] is False
    assert synths["ddsp"]["neural"] is True


def test_unknown_synth_is_rejected(client: TestClient) -> None:
    response = client.post("/api/v1/render", json=render_request("mystery"))

    assert response.status_code == 422


def test_render_requires_tempo(client: TestClient) -> None:
    payload = render_request()
    del payload["tempo"]

    response = client.post("/api/v1/render", json=payload)

    assert response.status_code == 422


def test_render_defaults_to_sf2_when_synth_is_omitted(client: TestClient) -> None:
    payload = render_request()
    del payload["synth"]

    response = client.post("/api/v1/render", json=payload)

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-synth-used"] == "sf2"


def test_ddsp_adapter_is_discovered_without_eager_import() -> None:
    module_name = "backend.tests.fake_ddsp_adapter"
    sys.modules.pop(module_name, None)
    app = create_app(
        settings=Settings(ddsp_adapter=f"{module_name}:adapter"),
        engine_service=EngineService(
            engines=[StubEngine()],
            discover_modules=False,
        ),
    )

    with TestClient(app) as client:
        capabilities = client.get("/api/v1/synths")
        assert capabilities.json()["synths"][1]["available"] is True
        assert module_name not in sys.modules

        response = client.post("/api/v1/render", json=render_request("ddsp"))

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-synth-used"] == "ddsp"
    assert response.headers["x-harmonaizer-renderer"] == "neural-adapter"
    assert module_name in sys.modules
