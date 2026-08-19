import wave
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.services.transcription import AudioDecodeError, TranscriptionService
from contracts.schema import Melody, Note, TimeSignature


class FakeTranscriptionService:
    def transcribe(
        self,
        data: bytes,
        *,
        tempo: float,
        time_signature: TimeSignature,
    ) -> Melody:
        assert data == b"audio"
        return Melody(
            notes=[Note(pitch=64, start=0, duration=1)],
            tempo=tempo,
            timeSignature=time_signature,
        )


def test_transcribe_upload_uses_contract_response(client: TestClient) -> None:
    client.app.state.transcription_service = FakeTranscriptionService()

    response = client.post(
        "/api/v1/transcribe?tempo=100&numerator=3&denominator=4",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["notes"][0]["pitch"] == 64
    assert response.json()["tempo"] == 100
    assert response.json()["timeSignature"] == {"numerator": 3, "denominator": 4}


def test_pyin_tracks_a_clean_monophonic_tone() -> None:
    sample_rate = 22_050
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    samples = 0.3 * np.sin(2 * np.pi * 440.0 * time)
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
        tempo=120,
        time_signature=TimeSignature(numerator=4, denominator=4),
    )

    assert melody.notes
    assert any(abs(note.pitch - 69) <= 1 for note in melody.notes)


def test_decode_rejects_audio_over_duration_limit() -> None:
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
            tempo=90,
            time_signature=TimeSignature(numerator=4, denominator=4),
        )
