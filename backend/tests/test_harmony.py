import pytest
from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "2.0.0"}


def test_engines_are_reflected_from_registry_adapter(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    response = client.get("/api/v1/engines")

    assert response.status_code == 200
    assert response.json() == {
        "engines": [
            {
                "id": canonical_request_payload["engine"],
                "name": "Canonical Fixture Stub",
                "description": "Returns the shared schema-valid SATB fixture",
                "available": True,
                "learned": False,
            }
        ]
    }


def test_harmonize_maps_engine_result_to_contract(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    response = client.post(
        "/api/v1/harmonize",
        json=canonical_request_payload,
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == set(canonical_response_payload)
    assert set(payload["key"]) == set(canonical_response_payload["key"])
    assert len(payload["chords"]) == len(canonical_response_payload["chords"])
    assert len(payload["voices"]) == len(canonical_response_payload["voices"])
    assert len(payload["violations"]) == len(canonical_response_payload["violations"])
    assert all(
        set(chord) == set(canonical_response_payload["chords"][0])
        for chord in payload["chords"]
    )
    assert all(
        set(voice) == set(canonical_response_payload["voices"][0])
        for voice in payload["voices"]
    )
    assert all(
        set(note) == set(canonical_response_payload["voices"][0]["notes"][0])
        for voice in payload["voices"]
        for note in voice["notes"]
    )
    assert all(
        set(violation) == set(canonical_response_payload["violations"][0])
        for violation in payload["violations"]
    )
    assert payload["engine"] == canonical_response_payload["engine"]
    assert payload["chords"][0]["inversion"] == 0
    assert payload["chords"][0]["secondaryOf"] is None
    assert payload["voices"][0]["notes"][0] == canonical_response_payload["voices"][
        0
    ]["notes"][0]
    assert payload["latencyMs"] >= 0


def test_missing_engine_is_clean_503(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    canonical_request_payload["engine"] = "missing"
    response = client.post(
        "/api/v1/harmonize",
        json=canonical_request_payload,
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "engine_unavailable"
    assert response.json()["detail"]["engine"] == "missing"


def test_failed_engine_is_clean_503(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    canonical_request_payload: dict,
) -> None:
    engine_id = canonical_request_payload["engine"]
    engine = client.app.state.engine_service.resolve(engine_id)

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("checkpoint exploded")

    monkeypatch.setattr(engine, "harmonize", fail)
    response = client.post(
        "/api/v1/harmonize",
        json=canonical_request_payload,
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "engine_unavailable",
        "message": f"Harmony engine {engine_id!r} failed to generate a result.",
        "engine": engine_id,
    }


def test_harmonize_requires_melody_tempo(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    del canonical_request_payload["melody"]["tempo"]

    response = client.post(
        "/api/v1/harmonize",
        json=canonical_request_payload,
    )

    assert response.status_code == 422


def test_partial_time_signature_is_rejected(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    canonical_signature = canonical_request_payload["melody"]["timeSignature"]
    canonical_request_payload["melody"]["timeSignature"] = {
        "numerator": canonical_signature["numerator"]
    }

    response = client.post(
        "/api/v1/harmonize",
        json=canonical_request_payload,
    )

    assert response.status_code == 422


def test_omitted_time_signature_gets_complete_default(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    del canonical_request_payload["melody"]["timeSignature"]

    response = client.post(
        "/api/v1/harmonize",
        json=canonical_request_payload,
    )

    assert response.status_code == 200
