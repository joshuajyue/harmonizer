from io import BytesIO

import mido
from fastapi.testclient import TestClient


def harmonization_payload() -> dict:
    return {
        "key": {"tonic": 0, "mode": "major", "confidence": 1.0},
        "chords": [],
        "voices": [
            {
                "name": "soprano",
                "notes": [
                    {
                        "pitch": 60,
                        "start": 0,
                        "duration": 1,
                        "velocity": 80,
                    }
                ],
            },
            {
                "name": "alto",
                "notes": [
                    {
                        "pitch": 55,
                        "start": 0,
                        "duration": 1,
                        "velocity": 72,
                    }
                ],
            },
        ],
        "violations": [],
        "engine": "stub",
        "latencyMs": 1.0,
    }


def test_midi_export_and_import_round_trip(client: TestClient) -> None:
    exported = client.post(
        "/api/v1/midi/export?tempo=96",
        json=harmonization_payload(),
    )

    assert exported.status_code == 200
    assert exported.content.startswith(b"MThd")
    imported = client.post(
        "/api/v1/midi/import",
        files={"file": ("harmony.mid", exported.content, "audio/midi")},
    )

    assert imported.status_code == 200
    payload = imported.json()
    assert payload["tempo"] == 96
    assert {note["pitch"] for note in payload["notes"]} == {55, 60}
    assert payload["key"]["tonic"] == 0


def test_invalid_midi_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("bad.mid", b"not-midi", "audio/midi")},
    )

    assert response.status_code == 422


def test_midi_without_declared_tempo_is_rejected(client: TestClient) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    output = BytesIO()
    midi.save(file=output)

    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("no-tempo.mid", output.getvalue(), "audio/midi")},
    )

    assert response.status_code == 422
    assert "does not declare a tempo" in response.json()["detail"]


def test_midi_export_requires_tempo(client: TestClient) -> None:
    response = client.post(
        "/api/v1/midi/export",
        json=harmonization_payload(),
    )

    assert response.status_code == 422
