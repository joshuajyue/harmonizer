import pytest
from fastapi.testclient import TestClient


def melody_request(engine: str = "stub") -> dict:
    return {
        "engine": engine,
        "melody": {
            "notes": [{"pitch": 60, "start": 0, "duration": 1}],
            "tempo": 90,
            "timeSignature": {"numerator": 4, "denominator": 4},
            "key": {"tonic": 0, "mode": "major"},
        },
    }


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "2.0.0"}


def test_engines_are_reflected_from_registry_adapter(client: TestClient) -> None:
    response = client.get("/api/v1/engines")

    assert response.status_code == 200
    assert response.json() == {
        "engines": [
            {
                "id": "stub",
                "name": "Stub Engine",
                "description": "Deterministic test engine",
                "available": True,
                "learned": False,
            }
        ]
    }


def test_harmonize_maps_engine_result_to_contract(client: TestClient) -> None:
    response = client.post("/api/v1/harmonize", json=melody_request())

    assert response.status_code == 200
    payload = response.json()
    assert payload["engine"] == "stub"
    assert payload["key"]["tonic"] == 0
    assert "confidence" in payload["key"]
    assert payload["chords"][0]["inversion"] == 0
    assert payload["chords"][0]["secondaryOf"] is None
    assert payload["voices"][0]["notes"][0]["pitch"] == 60
    assert payload["voices"][0]["notes"][0]["velocity"] == 80
    assert payload["violations"] == []
    assert payload["latencyMs"] >= 0


def test_missing_engine_is_clean_503(client: TestClient) -> None:
    response = client.post(
        "/api/v1/harmonize",
        json=melody_request(engine="missing"),
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "engine_unavailable"
    assert response.json()["detail"]["engine"] == "missing"


def test_failed_engine_is_clean_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = client.app.state.engine_service.resolve("stub")

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("checkpoint exploded")

    monkeypatch.setattr(engine, "harmonize", fail)
    response = client.post("/api/v1/harmonize", json=melody_request())

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "engine_unavailable",
        "message": "Harmony engine 'stub' failed to generate a result.",
        "engine": "stub",
    }


def test_harmonize_requires_melody_tempo(client: TestClient) -> None:
    payload = melody_request()
    del payload["melody"]["tempo"]

    response = client.post("/api/v1/harmonize", json=payload)

    assert response.status_code == 422


def test_partial_time_signature_is_rejected(client: TestClient) -> None:
    payload = melody_request()
    payload["melody"]["timeSignature"] = {"numerator": 3}

    response = client.post("/api/v1/harmonize", json=payload)

    assert response.status_code == 422


def test_omitted_time_signature_gets_complete_default(client: TestClient) -> None:
    payload = melody_request()
    del payload["melody"]["timeSignature"]

    response = client.post("/api/v1/harmonize", json=payload)

    assert response.status_code == 200
