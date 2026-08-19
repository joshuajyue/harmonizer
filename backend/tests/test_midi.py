from io import BytesIO

import mido
import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.services.engines import EngineService
from backend.app.services.midi import MidiConversionError, MidiService
from backend.tests.conftest import StubEngine
from contracts.schema import TimeSignature, Voice


def test_midi_export_and_import_round_trip(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    tempo = canonical_request_payload["melody"]["tempo"]
    signature = canonical_request_payload["melody"]["timeSignature"]
    exported = client.post(
        f"/api/v1/midi/export?tempo={tempo}"
        f"&numerator={signature['numerator']}"
        f"&denominator={signature['denominator']}",
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
    assert "confidence" not in payload["key"]


def test_invalid_midi_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("bad.mid", b"not-midi", "audio/midi")},
    )

    assert response.status_code == 422


def test_multipart_request_body_size_is_limited() -> None:
    app = create_app(
        settings=Settings(max_upload_bytes=32),
        engine_service=EngineService(
            engines=[StubEngine()],
            discover_modules=False,
        ),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/midi/import",
            files={"file": ("large.mid", b"x" * 70_000, "audio/midi")},
        )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large."


@pytest.mark.parametrize("division", [0, 0xE728])
def test_unsupported_midi_time_division_is_rejected(
    client: TestClient,
    division: int,
) -> None:
    track = b"\x00\xff\x2f\x00"
    midi = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + division.to_bytes(2, "big")
        + b"MTrk"
        + len(track).to_bytes(4, "big")
        + track
    )

    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("bad-division.mid", midi, "audio/midi")},
    )

    assert response.status_code == 422
    assert "time division is unsupported" in response.json()["detail"]


def test_midi_import_caps_note_count_before_full_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(88), time=0))
    for pitch in (60, 62, 64):
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=120))
    output = BytesIO()
    midi.save(file=output)
    service = MidiService(max_upload_bytes=100_000, max_notes=2)

    def parse_should_not_run(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("MIDI parser should not run after preflight rejection")

    monkeypatch.setattr(mido, "MidiFile", parse_should_not_run)
    with pytest.raises(MidiConversionError, match="too many notes"):
        service.import_melody(output.getvalue())


def test_midi_conversion_validation_errors_are_rejected(
    client: TestClient,
) -> None:
    track = (
        b"\x00\xff\x51\x03\x07\xa1\x20"
        b"\x00\xff\x58\x04\x00\x02\x18\x08"
        b"\x00\xff\x2f\x00"
    )
    midi = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (480).to_bytes(2, "big")
        + b"MTrk"
        + len(track).to_bytes(4, "big")
        + track
    )

    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("bad-meter.mid", midi, "audio/midi")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not valid MIDI."


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
        "/api/v1/midi/export?numerator=4&denominator=4",
        json=canonical_response_payload,
    )

    assert response.status_code == 422


def test_midi_export_requires_time_signature(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    response = client.post(
        f"/api/v1/midi/export?tempo={canonical_request_payload['melody']['tempo']}",
        json=canonical_response_payload,
    )

    assert response.status_code == 422


def test_midi_export_rejects_unrepresentable_tempo(
    client: TestClient,
    canonical_response_payload: dict,
) -> None:
    response = client.post(
        "/api/v1/midi/export?tempo=0.0001&numerator=4&denominator=4",
        json=canonical_response_payload,
    )

    assert response.status_code == 422


def test_midi_service_rejects_unrepresentable_tempo(
    canonical_response_payload: dict,
) -> None:
    service = MidiService(max_upload_bytes=100_000)
    voices = [
        Voice.model_validate(voice)
        for voice in canonical_response_payload["voices"]
    ]

    with pytest.raises(MidiConversionError, match="between 4 and 400"):
        service.voices_to_midi(
            voices,
            tempo=0.0001,
            time_signature=TimeSignature(numerator=4, denominator=4),
        )


def test_midi_export_rejects_non_power_of_two_meter(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    tempo = canonical_request_payload["melody"]["tempo"]
    response = client.post(
        f"/api/v1/midi/export?tempo={tempo}&numerator=4&denominator=3",
        json=canonical_response_payload,
    )

    assert response.status_code == 422
    assert "power-of-two denominator" in response.json()["detail"]


def test_midi_export_preserves_explicit_meter(
    client: TestClient,
    canonical_request_payload: dict,
    canonical_response_payload: dict,
) -> None:
    tempo = canonical_request_payload["melody"]["tempo"]
    exported = client.post(
        f"/api/v1/midi/export?tempo={tempo}&numerator=3&denominator=4",
        json=canonical_response_payload,
    )
    imported = client.post(
        "/api/v1/midi/import",
        files={"file": ("three-four.mid", exported.content, "audio/midi")},
    )

    assert exported.status_code == 200
    assert imported.status_code == 200
    assert imported.json()["timeSignature"] == {
        "numerator": 3,
        "denominator": 4,
    }


def test_midi_import_omits_absent_key(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    note = canonical_request_payload["melody"]["notes"][0]
    midi = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(88), time=0))
    track.append(
        mido.Message(
            "note_on",
            note=note["pitch"],
            velocity=note["velocity"],
            time=0,
        )
    )
    track.append(
        mido.Message(
            "note_off",
            note=note["pitch"],
            velocity=0,
            time=480,
        )
    )
    output = BytesIO()
    midi.save(file=output)

    response = client.post(
        "/api/v1/midi/import",
        files={"file": ("no-key.mid", output.getvalue(), "audio/midi")},
    )

    assert response.status_code == 200
    assert "key" not in response.json()
