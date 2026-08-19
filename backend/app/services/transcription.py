from __future__ import annotations

import logging
import shutil
import subprocess
from io import BytesIO

import numpy as np
import soundfile as sf

from contracts.schema import Melody, Note, TimeSignature

logger = logging.getLogger(__name__)


class AudioDecodeError(ValueError):
    pass


class TranscriptionError(ValueError):
    pass


class TranscriptionService:
    def __init__(
        self,
        *,
        max_upload_bytes: int,
        max_audio_seconds: float,
        analysis_sample_rate: int = 22_050,
    ) -> None:
        self._max_upload_bytes = max_upload_bytes
        self._max_audio_seconds = max_audio_seconds
        self._analysis_sample_rate = analysis_sample_rate

    def transcribe(
        self,
        data: bytes,
        *,
        tempo: float,
        time_signature: TimeSignature,
    ) -> Melody:
        samples = self._decode(data)
        notes = self._track_pitch(samples, tempo)
        if not notes:
            raise TranscriptionError("No stable monophonic pitch was detected.")
        return Melody(
            notes=notes,
            tempo=tempo,
            timeSignature=time_signature,
        )

    def _decode(self, data: bytes) -> np.ndarray:
        if not data:
            raise AudioDecodeError("The uploaded audio file is empty.")
        if len(data) > self._max_upload_bytes:
            raise AudioDecodeError("The uploaded audio file is too large.")

        samples: np.ndarray
        sample_rate: int
        try:
            with sf.SoundFile(BytesIO(data)) as audio:
                sample_rate = audio.samplerate
                max_frames = round(self._max_audio_seconds * sample_rate)
                if len(audio) > max_frames:
                    raise AudioDecodeError(
                        f"Audio may not be longer than {self._max_audio_seconds:g} seconds."
                    )
                decoded = audio.read(
                    frames=max_frames + 1,
                    dtype="float32",
                    always_2d=True,
                )
            samples = np.mean(decoded, axis=1, dtype=np.float32)
        except AudioDecodeError:
            raise
        except Exception:
            samples, sample_rate = self._decode_with_ffmpeg(data)

        if sample_rate <= 0 or samples.size == 0:
            raise AudioDecodeError("The uploaded audio file contains no samples.")
        if samples.size / sample_rate > self._max_audio_seconds:
            raise AudioDecodeError(
                f"Audio may not be longer than {self._max_audio_seconds:g} seconds."
            )

        samples = np.nan_to_num(samples, copy=False)
        samples -= np.mean(samples, dtype=np.float64)
        if sample_rate != self._analysis_sample_rate:
            import librosa

            samples = librosa.resample(
                samples,
                orig_sr=sample_rate,
                target_sr=self._analysis_sample_rate,
            ).astype(np.float32, copy=False)
        return np.ascontiguousarray(samples, dtype=np.float32)

    def _decode_with_ffmpeg(self, data: bytes) -> tuple[np.ndarray, int]:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise AudioDecodeError(
                "This audio encoding is unsupported; install ffmpeg for MP3/WebM uploads."
            )
        command = [
            ffmpeg,
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-t",
            str(self._max_audio_seconds + 1),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(self._analysis_sample_rate),
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=max(30.0, self._max_audio_seconds * 2),
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.info("ffmpeg could not decode uploaded audio: %s", exc)
            raise AudioDecodeError(
                "The uploaded file is not valid WAV, MP3, or WebM audio."
            ) from exc
        samples = np.frombuffer(completed.stdout, dtype="<f4").copy()
        return samples, self._analysis_sample_rate

    def _track_pitch(self, samples: np.ndarray, tempo: float) -> list[Note]:
        import librosa

        frame_length = 2048
        hop_length = 256
        if samples.size < frame_length:
            samples = np.pad(samples, (0, frame_length - samples.size))

        f0, voiced, voiced_probability = librosa.pyin(
            samples,
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C7")),
            sr=self._analysis_sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            fill_na=np.nan,
        )
        rms = librosa.feature.rms(
            y=samples,
            frame_length=frame_length,
            hop_length=hop_length,
        )[0]
        frame_count = min(len(f0), len(rms))
        if frame_count == 0:
            return []
        return _frames_to_notes(
            f0=f0[:frame_count],
            voiced=voiced[:frame_count],
            voiced_probability=voiced_probability[:frame_count],
            rms=rms[:frame_count],
            sample_rate=self._analysis_sample_rate,
            hop_length=hop_length,
            tempo=tempo,
        )


def _frames_to_notes(
    *,
    f0: np.ndarray,
    voiced: np.ndarray,
    voiced_probability: np.ndarray,
    rms: np.ndarray,
    sample_rate: int,
    hop_length: int,
    tempo: float,
) -> list[Note]:
    peak_rms = float(np.max(rms, initial=0.0))
    if peak_rms <= 1e-6:
        return []
    rms_floor = max(1e-4, peak_rms * 0.025)
    pitches: list[int | None] = []
    for frequency, is_voiced, probability, frame_rms in zip(
        f0,
        voiced,
        voiced_probability,
        rms,
        strict=True,
    ):
        if (
            not is_voiced
            or not np.isfinite(frequency)
            or probability < 0.5
            or frame_rms < rms_floor
        ):
            pitches.append(None)
            continue
        midi_pitch = int(np.clip(np.rint(69 + 12 * np.log2(frequency / 440.0)), 0, 127))
        pitches.append(midi_pitch)

    for index in range(1, len(pitches) - 1):
        before, current, after = pitches[index - 1 : index + 2]
        if before == after and current != before:
            pitches[index] = before

    frame_seconds = hop_length / sample_rate
    minimum_frames = max(2, round(0.08 / frame_seconds))
    raw_segments: list[tuple[int, int, int]] = []
    segment_pitch: int | None = None
    segment_start = 0
    for index, pitch in enumerate([*pitches, None]):
        if pitch == segment_pitch:
            continue
        if segment_pitch is not None and index - segment_start >= minimum_frames:
            raw_segments.append((segment_start, index, segment_pitch))
        segment_pitch = pitch
        segment_start = index

    merged: list[tuple[int, int, int]] = []
    max_merge_gap = max(1, round(0.06 / frame_seconds))
    for start, end, pitch in raw_segments:
        if merged and merged[-1][2] == pitch and start - merged[-1][1] <= max_merge_gap:
            previous_start, _, _ = merged[-1]
            merged[-1] = (previous_start, end, pitch)
        else:
            merged.append((start, end, pitch))

    seconds_to_beats = tempo / 60.0
    notes: list[Note] = []
    for start, end, pitch in merged:
        local_rms = float(np.mean(rms[start:end]))
        loudness = np.sqrt(max(0.0, local_rms / peak_rms))
        velocity = int(np.clip(round(35 + 85 * loudness), 1, 127))
        notes.append(
            Note(
                pitch=pitch,
                start=start * frame_seconds * seconds_to_beats,
                duration=(end - start) * frame_seconds * seconds_to_beats,
                velocity=velocity,
            )
        )
    return notes
