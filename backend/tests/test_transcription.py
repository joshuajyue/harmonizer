import wave
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.services.transcription import AudioDecodeError, TranscriptionService
from contracts.schema import Melody, TimeSignature


class FakeTranscriptionService:
    def __init__(self, melody: Melody) -> None:
        self._melody = melody

    def transcribe(
        self,
        data: bytes,
        *,
        tempo: float | None,
        time_signature: TimeSignature,
    ) -> Melody:
        assert data == b"audio"
        assert tempo is not None
        return self._melody.model_copy(
            update={
                "tempo": tempo,
                "timeSignature": time_signature,
            },
            deep=True,
        )


def test_transcribe_upload_uses_contract_response(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    fixture_melody = Melody.model_validate(canonical_request_payload["melody"])
    signature = fixture_melody.timeSignature
    client.app.state.transcription_service = FakeTranscriptionService(fixture_melody)

    response = client.post(
        f"/api/v1/transcribe?tempo={fixture_melody.tempo}"
        f"&numerator={signature.numerator}&denominator={signature.denominator}",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["notes"] == canonical_request_payload["melody"]["notes"]
    assert response.json()["tempo"] == fixture_melody.tempo
    assert response.json()["timeSignature"] == canonical_request_payload["melody"][
        "timeSignature"
    ]


def test_transcribe_rejects_partial_time_signature(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    melody = canonical_request_payload["melody"]
    response = client.post(
        f"/api/v1/transcribe?tempo={melody['tempo']}"
        f"&numerator={melody['timeSignature']['numerator']}",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 422
    assert "both time-signature" in response.json()["detail"]


def test_transcribe_uses_complete_default_time_signature(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    fixture_melody = Melody.model_validate(canonical_request_payload["melody"])
    client.app.state.transcription_service = FakeTranscriptionService(fixture_melody)

    response = client.post(
        f"/api/v1/transcribe?tempo={fixture_melody.tempo}",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["timeSignature"] == canonical_request_payload["melody"][
        "timeSignature"
    ]


def test_pyin_tracks_a_clean_monophonic_tone(
    canonical_request_payload: dict,
) -> None:
    fixture_melody = Melody.model_validate(canonical_request_payload["melody"])
    fixture_note = fixture_melody.notes[0]
    sample_rate = 22_050
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    frequency = 440.0 * 2.0 ** ((fixture_note.pitch - 69) / 12.0)
    samples = 0.3 * np.sin(2 * np.pi * frequency * time)
    samples[:256] *= np.linspace(0, 1, 256)
    samples[-256:] *= np.linspace(1, 0, 256)
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.asarray(samples * 32767, dtype="<i2").tobytes())

    service = TranscriptionService(
        max_upload_bytes=1_000_000,
        max_audio_seconds=5,
    )
    melody = service.transcribe(
        output.getvalue(),
        tempo=fixture_melody.tempo,
        time_signature=fixture_melody.timeSignature,
    )

    assert melody.notes
    assert any(abs(note.pitch - fixture_note.pitch) <= 1 for note in melody.notes)


def test_decode_rejects_audio_over_duration_limit(
    canonical_request_payload: dict,
) -> None:
    fixture_melody = Melody.model_validate(canonical_request_payload["melody"])
    sample_rate = 8_000
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(np.zeros(sample_rate * 2, dtype="<i2").tobytes())

    service = TranscriptionService(
        max_upload_bytes=100_000,
        max_audio_seconds=1,
    )

    with pytest.raises(AudioDecodeError, match="longer than 1 seconds"):
        service.transcribe(
            output.getvalue(),
            tempo=fixture_melody.tempo,
            time_signature=fixture_melody.timeSignature,
        )


def test_transcribe_detects_tempo_from_note_onsets(
    canonical_request_payload: dict,
) -> None:
    fixture_melody = Melody.model_validate(canonical_request_payload["melody"])
    sample_rate = 22_050
    segments: list[np.ndarray] = []
    seconds_per_beat = 60.0 / fixture_melody.tempo
    for fixture_note in fixture_melody.notes[:6]:
        sample_count = round(sample_rate * seconds_per_beat)
        time = np.arange(sample_count, dtype=np.float32) / sample_rate
        frequency = 440.0 * 2.0 ** ((fixture_note.pitch - 69) / 12.0)
        samples = 0.3 * np.sin(2 * np.pi * frequency * time)
        fade = round(sample_rate * 0.025)
        samples[:fade] *= np.linspace(0, 1, fade)
        samples[-fade:] *= np.linspace(1, 0, fade)
        segments.append(samples)
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(
            np.asarray(np.concatenate(segments) * 32767, dtype="<i2").tobytes()
        )

    service = TranscriptionService(
        max_upload_bytes=1_000_000,
        max_audio_seconds=10,
    )
    melody = service.transcribe(
        output.getvalue(),
        tempo=None,
        time_signature=fixture_melody.timeSignature,
    )

    assert abs(melody.tempo - fixture_melody.tempo) <= 5
    assert len(melody.notes) >= 4
