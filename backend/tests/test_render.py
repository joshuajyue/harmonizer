import sys
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.engines import EngineService
from backend.app.services.synthesis.base import BackendRender, SynthRender
from backend.app.services.synthesis.ddsp import DdspSynthBackend
from backend.app.services.synthesis.world import WorldSynthBackend
from backend.tests.conftest import StubEngine
from contracts.schema import RenderRequest


def render_request(
    canonical_request_payload: dict,
    canonical_response_payload: dict,
    synth: str = "sf2",
) -> dict:
    return {
        "voices": [
            {
                "name": voice["name"],
                "notes": voice["notes"][:1],
            }
            for voice in canonical_response_payload["voices"]
        ],
        "tempo": canonical_request_payload["melody"]["tempo"],
        "synth": synth,
    }


def test_sf2_render_always_returns_valid_wav(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    response = client.post(
        "/api/v1/render",
        json=render_request(canonical_request_payload, canonical_response_payload),
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-harmonaizer-synth-used"] == "sf2"
    with wave.open(BytesIO(response.content), "rb") as wav:
        assert wav.getnchannels() == 2
        assert wav.getnframes() > 0


def test_unconfigured_ddsp_falls_back_transparently(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    response = client.post(
        "/api/v1/render",
        json=render_request(
            canonical_request_payload,
            canonical_response_payload,
            "ddsp",
        ),
    )

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-synth-requested"] == "ddsp"
    assert response.headers["x-harmonaizer-synth-used"] == "sf2"
    assert "x-harmonaizer-fallback" in response.headers


def test_render_fallback_header_is_ascii_sanitized(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    class UnicodeDiagnosticService:
        def render(self, request: object) -> SynthRender:
            del request
            return SynthRender(
                audio=b"audio",
                requested="ddsp",
                used="sf2",
                renderer="wavetable",
                fallback_reason="bad Ж\r\nreason",
            )

    client.app.state.synth_service = UnicodeDiagnosticService()
    response = client.post(
        "/api/v1/render",
        json=render_request(
            canonical_request_payload,
            canonical_response_payload,
            "ddsp",
        ),
    )

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-fallback"] == "bad ???reason"


def test_world_timbre_ids_are_ascii_only() -> None:
    backend = WorldSynthBackend(
        timbre_dir=Path("backend/.runtime/timbres"),
        sample_rate=44_100,
        max_render_seconds=10,
    )

    with pytest.raises(ValueError, match="Invalid timbre id"):
        backend._resolve_timbre("Ж")


def test_fluidsynth_failure_is_reported(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    backend = client.app.state.synth_service._sf2
    monkeypatch.setattr(backend, "_find_soundfont", lambda: Path("soundfont.sf2"))
    monkeypatch.setattr(
        "backend.app.services.synthesis.sf2.shutil.which",
        lambda _: "fluidsynth",
    )

    def fail(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        raise RuntimeError("synth crashed")

    monkeypatch.setattr(backend, "_render_fluidsynth", fail)
    response = client.post(
        "/api/v1/render",
        json=render_request(canonical_request_payload, canonical_response_payload),
    )

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-renderer"] == "wavetable"
    assert "FluidSynth failed" in response.headers["x-harmonaizer-fallback"]


def test_synth_capabilities_expose_optional_neural_tier(client: TestClient) -> None:
    response = client.get("/api/v1/synths")

    assert response.status_code == 200
    synths = {item["id"]: item for item in response.json()["synths"]}
    assert synths["sf2"]["available"] is True
    assert synths["ddsp"]["available"] is False
    assert synths["ddsp"]["neural"] is True


def test_unknown_synth_is_rejected(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    response = client.post(
        "/api/v1/render",
        json=render_request(
            canonical_request_payload,
            canonical_response_payload,
            "mystery",
        ),
    )

    assert response.status_code == 422


def test_render_requires_tempo(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    payload = render_request(canonical_request_payload, canonical_response_payload)
    del payload["tempo"]

    response = client.post("/api/v1/render", json=payload)

    assert response.status_code == 422


def test_render_rejects_excessive_note_count(
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    app = create_app(
        settings=Settings(max_render_notes=2),
        engine_service=EngineService(
            engines=[StubEngine()],
            discover_modules=False,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/render",
            json=render_request(
                canonical_request_payload,
                canonical_response_payload,
            ),
        )

    assert response.status_code == 422
    assert "limit is 2" in response.json()["detail"]


def test_render_rejects_excessive_summed_note_duration(
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    app = create_app(
        settings=Settings(max_render_work_seconds=1),
        engine_service=EngineService(
            engines=[StubEngine()],
            discover_modules=False,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/render",
            json=render_request(
                canonical_request_payload,
                canonical_response_payload,
            ),
        )

    assert response.status_code == 422
    assert "note-seconds" in response.json()["detail"]


def test_json_request_body_size_is_limited() -> None:
    app = create_app(
        settings=Settings(max_json_body_bytes=64),
        engine_service=EngineService(
            engines=[StubEngine()],
            discover_modules=False,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/render",
            content=b"{" + b" " * 128 + b"}",
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large."


def test_render_defaults_to_sf2_when_synth_is_omitted(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    payload = render_request(canonical_request_payload, canonical_response_payload)
    del payload["synth"]

    response = client.post("/api/v1/render", json=payload)

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-synth-used"] == "sf2"


def test_ddsp_adapter_is_discovered_without_eager_import(
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
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

        response = client.post(
            "/api/v1/render",
            json=render_request(
                canonical_request_payload,
                canonical_response_payload,
                "ddsp",
            ),
        )

    assert response.status_code == 200
    assert response.headers["x-harmonaizer-synth-used"] == "ddsp"
    assert response.headers["x-harmonaizer-renderer"] == "neural-adapter"
    assert module_name in sys.modules


def test_ddsp_adapter_renders_are_serialized(
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    wav = BytesIO()
    with wave.open(wav, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 8)

    class GuideBackend:
        def render(self, *args: object, **kwargs: object) -> BackendRender:
            del args, kwargs
            return BackendRender(audio=wav.getvalue(), renderer="guide")

    class StatefulAdapter:
        def __init__(self) -> None:
            self._state_lock = threading.Lock()
            self._active = 0
            self.concurrent = False

        def render(self, **kwargs: object) -> bytes:
            del kwargs
            with self._state_lock:
                self._active += 1
                self.concurrent |= self._active > 1
            time.sleep(0.05)
            with self._state_lock:
                self._active -= 1
            return wav.getvalue()

    request = RenderRequest.model_validate(
        render_request(canonical_request_payload, canonical_response_payload)
    )
    adapter = StatefulAdapter()
    backend = DdspSynthBackend(
        adapter_spec=None,
        guide_backend=GuideBackend(),  # type: ignore[arg-type]
        sample_rate=8_000,
    )
    backend._adapter = adapter
    start = threading.Barrier(3)

    def render() -> None:
        start.wait()
        backend.render(
            request.voices,
            tempo=request.tempo,
            timbre=request.timbre,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(render) for _ in range(2)]
        start.wait()
        for future in futures:
            future.result(timeout=5)

    assert adapter.concurrent is False
