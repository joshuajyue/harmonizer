from __future__ import annotations

import importlib.util
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from backend.app.services.synthesis.audio import encode_pcm16_wav
from backend.app.services.synthesis.base import BackendRender, SynthAvailability
from backend.app.services.synthesis.sf2 import _render_duration
from contracts.schema import Voice


class WorldSynthBackend:
    """Optional CPU fallback that preserves a local reference voice's formants."""

    id = "world"
    name = "WORLD Voice Resynthesis"
    description = "Classic formant-preserving vocoder fallback using an authorized voice sample."
    neural = False

    def __init__(
        self,
        *,
        timbre_dir: Path,
        sample_rate: int,
        max_render_seconds: float,
    ) -> None:
        self._timbre_dir = timbre_dir
        self._sample_rate = sample_rate
        self._max_render_seconds = min(max_render_seconds, 60.0)

    def list_timbres(self) -> list[str]:
        if not self._timbre_dir.is_dir():
            return []
        return sorted(
            path.stem
            for path in self._timbre_dir.glob("*.wav")
            if path.is_file() and not path.name.startswith(".")
        )

    def availability(self, timbre: str | None = None) -> SynthAvailability:
        if importlib.util.find_spec("pyworld") is None:
            return SynthAvailability(
                False,
                "Install backend/requirements-voice.txt to enable WORLD resynthesis.",
            )
        try:
            self._resolve_timbre(timbre)
        except ValueError as exc:
            return SynthAvailability(False, str(exc))
        return SynthAvailability(True)

    def render(
        self,
        voices: list[Voice],
        *,
        tempo: float,
        timbre: str | None,
    ) -> BackendRender:
        import pyworld

        duration = _render_duration(voices, tempo)
        if duration > self._max_render_seconds:
            raise ValueError(
                f"WORLD renders are limited to {self._max_render_seconds:g} seconds."
            )
        reference_path = self._resolve_timbre(timbre)
        reference, reference_rate = sf.read(
            reference_path,
            dtype="float64",
            always_2d=True,
        )
        reference = np.mean(reference, axis=1)
        if reference_rate != self._sample_rate:
            divisor = gcd(reference_rate, self._sample_rate)
            reference = resample_poly(
                reference,
                self._sample_rate // divisor,
                reference_rate // divisor,
            )
        if reference.size < self._sample_rate // 10:
            raise ValueError("The WORLD timbre sample must be at least 100ms long.")
        reference = np.ascontiguousarray(reference, dtype=np.float64)

        frame_period_ms = 5.0
        source_f0, spectral_envelope, aperiodicity = pyworld.wav2world(
            reference,
            self._sample_rate,
            frame_period=frame_period_ms,
        )
        voiced_frames = np.flatnonzero(source_f0 > 0)
        if voiced_frames.size == 0:
            raise ValueError("The WORLD timbre sample contains no voiced pitch.")

        output_samples = max(
            1,
            round(max(0.5, duration + 0.1) * self._sample_rate),
        )
        target_frames = max(1, round(output_samples * 1000 / self._sample_rate / frame_period_ms))
        mix = np.zeros((output_samples, 2), dtype=np.float64)
        voice_count = max(1, len(voices))

        for voice_index, voice in enumerate(voices):
            source_indices = np.resize(
                np.roll(voiced_frames, voice_index * 7),
                target_frames,
            )
            target_f0 = np.zeros(target_frames, dtype=np.float64)
            frame_gain = np.zeros(target_frames, dtype=np.float64)
            for note in voice.notes:
                start_seconds = max(0.0, note.start * 60.0 / tempo)
                end_seconds = max(
                    start_seconds,
                    (note.start + note.duration) * 60.0 / tempo,
                )
                start = max(0, round(start_seconds * 1000 / frame_period_ms))
                end = min(target_frames, round(end_seconds * 1000 / frame_period_ms))
                if start >= end:
                    continue
                target_f0[start:end] = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
                gain = max(
                    frame_gain[start:end].max(initial=0),
                    note.velocity / 127.0,
                )
                frame_gain[start:end] = gain

            synthesized = pyworld.synthesize(
                target_f0,
                spectral_envelope[source_indices],
                aperiodicity[source_indices],
                self._sample_rate,
                frame_period=frame_period_ms,
            )
            synthesized = synthesized[:output_samples]
            gain = np.interp(
                np.arange(synthesized.size) / self._sample_rate,
                np.arange(target_frames) * frame_period_ms / 1000.0,
                frame_gain,
            )
            synthesized *= gain
            pan = 0.0 if voice_count == 1 else -0.65 + 1.3 * voice_index / (voice_count - 1)
            mix[: synthesized.size, 0] += synthesized * np.sqrt((1.0 - pan) / 2.0)
            mix[: synthesized.size, 1] += synthesized * np.sqrt((1.0 + pan) / 2.0)

        peak = float(np.max(np.abs(mix), initial=0.0))
        if peak > 0:
            mix *= min(1.0, 0.95 / peak)
        return BackendRender(
            audio=encode_pcm16_wav(mix.astype(np.float32), self._sample_rate),
            renderer="world",
        )

    def _resolve_timbre(self, timbre: str | None) -> Path:
        available = self.list_timbres()
        if timbre is None:
            if len(available) == 1:
                timbre = available[0]
            else:
                raise ValueError("Choose a configured WORLD timbre.")
        if Path(timbre).name != timbre or not timbre.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid timbre id.")
        candidate = self._timbre_dir / f"{timbre}.wav"
        if not candidate.is_file():
            raise ValueError(f"WORLD timbre {timbre!r} is not installed.")
        return candidate
