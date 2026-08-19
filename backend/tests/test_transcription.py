import subprocess
import wave
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from imageio_ffmpeg import get_ffmpeg_exe

import backend.app.services.transcription as transcription_module
from backend.app.services.transcription import (
    AudioDecodeError,
    TranscriptionResult,
    TranscriptionService,
    normalize_melody_octaves,
)
from contracts.schema import Melody, Note, TimeSignature


class FakeTranscriptionService:
    def __init__(self, melody: Melody) -> None:
        self._melody = melody

    def transcribe(
        self,
        data: bytes,
        *,
        tempo: float | None,
        time_signature: TimeSignature,
        normalize_octave: bool = True,
        octave_shift: int | None = None,
    ) -> TranscriptionResult:
        assert data == b"audio"
        assert tempo is not None
        normalization = normalize_melody_octaves(
            self._melody.notes,
            enabled=normalize_octave,
            forced_octave_shift=octave_shift,
        )
        return TranscriptionResult(
            melody=self._melody.model_copy(
                update={
                    "notes": normalization.notes,
                    "tempo": tempo,
                    "timeSignature": time_signature,
                },
                deep=True,
            ),
            octave_shift=normalization.octave_shift,
            detected_median_pitch=normalization.detected_median_pitch,
        )


def notes_at(pitches: list[int]) -> list[Note]:
    return [
        Note(pitch=pitch, start=float(index), duration=1.0, velocity=80)
        for index, pitch in enumerate(pitches)
    ]


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
    assert response.headers["x-harmonaizer-octave-shift"] == "0"
    assert float(
        response.headers["x-harmonaizer-detected-median-pitch"]
    ) == pytest.approx(
        np.median([note.pitch for note in fixture_melody.notes])
    )


def test_bass_register_is_shifted_up_by_whole_octaves() -> None:
    original = notes_at([43, 45, 47, 50])

    normalized = normalize_melody_octaves(original)

    assert normalized.octave_shift == 2
    assert normalized.detected_median_pitch == 46
    assert [note.pitch for note in normalized.notes] == [67, 69, 71, 74]


def test_melody_already_in_working_range_is_not_shifted() -> None:
    original = notes_at([60, 62, 64, 67])

    normalized = normalize_melody_octaves(original)

    assert normalized.octave_shift == 0
    assert [note.pitch for note in normalized.notes] == [60, 62, 64, 67]


def test_wide_range_is_shifted_globally_without_clamping() -> None:
    original_pitches = [43, 55, 67, 79]

    normalized = normalize_melody_octaves(notes_at(original_pitches))
    normalized_pitches = [note.pitch for note in normalized.notes]

    assert normalized.octave_shift == 1
    assert normalized_pitches == [55, 67, 79, 91]
    assert [
        right - left for left, right in zip(normalized_pitches, normalized_pitches[1:])
    ] == [
        right - left for left, right in zip(original_pitches, original_pitches[1:])
    ]
    assert [pitch % 12 for pitch in normalized_pitches] == [
        pitch % 12 for pitch in original_pitches
    ]


def test_transcribe_can_disable_octave_normalization(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    fixture = Melody.model_validate(canonical_request_payload["melody"]).model_copy(
        update={"notes": notes_at([43, 45, 47, 50])},
        deep=True,
    )
    client.app.state.transcription_service = FakeTranscriptionService(fixture)

    response = client.post(
        f"/api/v1/transcribe?tempo={fixture.tempo}"
        "&normalizeOctave=false",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert [note["pitch"] for note in response.json()["notes"]] == [43, 45, 47, 50]
    assert response.headers["x-harmonaizer-octave-shift"] == "0"
    assert response.headers["x-harmonaizer-detected-median-pitch"] == "46"


def test_transcribe_can_force_octave_shift(
    client: TestClient,
    canonical_request_payload: dict,
) -> None:
    fixture = Melody.model_validate(canonical_request_payload["melody"]).model_copy(
        update={"notes": notes_at([43, 45, 47, 50])},
        deep=True,
    )
    client.app.state.transcription_service = FakeTranscriptionService(fixture)

    response = client.post(
        f"/api/v1/transcribe?tempo={fixture.tempo}"
        "&normalizeOctave=false&octaveShift=1",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 200
    assert [note["pitch"] for note in response.json()["notes"]] == [55, 57, 59, 62]
    assert response.headers["x-harmonaizer-octave-shift"] == "+1"


def test_forced_octave_shift_rejects_midi_overflow() -> None:
    with pytest.raises(
        transcription_module.TranscriptionError,
        match="outside MIDI range",
    ):
        normalize_melody_octaves(
            notes_at([120, 124]),
            forced_octave_shift=1,
        )


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
    result = service.transcribe(
        output.getvalue(),
        tempo=fixture_melody.tempo,
        time_signature=fixture_melody.timeSignature,
    )

    assert result.melody.notes
    assert any(
        abs(note.pitch - fixture_note.pitch) <= 1 for note in result.melody.notes
    )


def test_webm_upload_uses_bundled_ffmpeg(
    client: TestClient,
    canonical_request_payload: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_melody = Melody.model_validate(canonical_request_payload["melody"])
    fixture_note = fixture_melody.notes[0]
    sample_rate = 22_050
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    frequency = 440.0 * 2.0 ** ((fixture_note.pitch - 69) / 12.0)
    samples = 0.3 * np.sin(2 * np.pi * frequency * time)
    pcm = np.asarray(samples * 32767, dtype="<i2").tobytes()
    encoded = subprocess.run(
        [
            get_ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "64k",
            "-f",
            "webm",
            "pipe:1",
        ],
        input=pcm,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=30,
    ).stdout
    monkeypatch.setattr(transcription_module.shutil, "which", lambda _: None)

    response = client.post(
        f"/api/v1/transcribe?tempo={fixture_melody.tempo}"
        f"&numerator={fixture_melody.timeSignature.numerator}"
        f"&denominator={fixture_melody.timeSignature.denominator}",
        files={"audio": ("recording.webm", encoded, "audio/webm")},
    )

    assert response.status_code == 200
    assert response.json()["tempo"] == fixture_melody.tempo
    assert any(
        abs(note["pitch"] - fixture_note.pitch) <= 1
        for note in response.json()["notes"]
    )


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
    result = service.transcribe(
        output.getvalue(),
        tempo=None,
        time_signature=fixture_melody.timeSignature,
    )

    assert abs(result.melody.tempo - fixture_melody.tempo) <= 5
    assert len(result.melody.notes) >= 4
