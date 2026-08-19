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
    assert payload["voices"][0]["notes"][0]["pitch"] == 60
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
