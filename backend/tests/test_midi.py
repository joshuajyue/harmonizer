from io import BytesIO

import mido
import pytest
from fastapi.testclient import TestClient


def test_midi_export_and_import_round_trip(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    tempo = canonical_request_payload["melody"]["tempo"]
    exported = client.post(
        f"/api/v1/midi/export?tempo={tempo}",
        json=canonical_response_payload,
    )

    assert exported.status_code == 200
    assert exported.content.startswith(b"MThd")
    imported = client.post(
        "/api/v1/midi/import",
        files={"file": ("harmony.mid", exported.content, "audio/midi")},
    )

    assert imported.status_code == 200
    payload = imported.json()
    expected_notes = [
        note
        for voice in canonical_response_payload["voices"]
        for note in voice["notes"]
    ]
    assert payload["tempo"] == pytest.approx(tempo, abs=0.001)
    assert len(payload["notes"]) == len(expected_notes)
    assert {note["pitch"] for note in payload["notes"]} == {
        note["pitch"] for note in expected_notes
    }
    assert payload["timeSignature"] == canonical_request_payload["melody"][
        "timeSignature"
    ]
    assert payload["key"]["tonic"] == canonical_response_payload["key"]["tonic"]


def test_invalid_midi_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("bad.mid", b"not-midi", "audio/midi")},
    )

    assert response.status_code == 422


def test_midi_without_declared_tempo_is_rejected(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    fixture_note = canonical_request_payload["melody"]["notes"][0]
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.Message(
            "note_on",
            note=fixture_note["pitch"],
            velocity=fixture_note["velocity"],
            time=0,
        )
    )
    track.append(
        mido.Message(
            "note_off",
            note=fixture_note["pitch"],
            velocity=0,
            time=round(fixture_note["duration"] * midi.ticks_per_beat),
        )
    )
    output = BytesIO()
    midi.save(file=output)

    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("no-tempo.mid", output.getvalue(), "audio/midi")},
    )

    assert response.status_code == 422
    assert "does not declare a tempo" in response.json()["detail"]


def test_midi_export_requires_tempo(
    client: TestClient,
    canonical_response_payload: dict,
) -> None:
    response = client.post(
        "/api/v1/midi/export",
        json=canonical_response_payload,
    )

    assert response.status_code == 422
