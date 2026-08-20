from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

import numpy as np

from backend.app.services.midi import MidiService
from backend.app.services.synthesis.audio import encode_pcm16_wav, is_wav
from backend.app.services.synthesis.base import (
    BackendRender,
    SynthAvailability,
)
from contracts.schema import TimeSignature, Voice

logger = logging.getLogger(__name__)

_SYSTEM_SOUNDFONTS = (
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/TimGM6mb.sf2"),
    Path("/usr/share/soundfonts/default.sf2"),
    Path("/usr/local/share/soundfonts/default.sf2"),
)


class SoundFontSynthBackend:
    id = "sf2"
    name = "SoundFont Preview"
    description = (
        "FluidSynth SoundFont rendering with an in-process wavetable fallback."
    )
    neural = False

    def __init__(
        self,
        *,
        sample_rate: int,
        max_render_seconds: float,
        runtime_dir: Path,
        soundfont_path: Path | None,
        midi_service: MidiService,
    ) -> None:
        self._sample_rate = sample_rate
        self._max_render_seconds = max_render_seconds
        self._runtime_dir = runtime_dir
        self._configured_soundfont = soundfont_path
        self._midi_service = midi_service

    def availability(self, timbre: str | None = None) -> SynthAvailability:
        del timbre
        soundfont = self._find_soundfont()
        if shutil.which("fluidsynth") and soundfont:
            return SynthAvailability(
                True,
                f"FluidSynth using {soundfont.name}; in-process fallback is also ready.",
            )
        return SynthAvailability(
            True,
            "FluidSynth or a SoundFont was not found; using the in-process wavetable renderer.",
        )

    def render(
        self,
        voices: list[Voice],
        *,
        tempo: float,
        timbre: str | None = None,
    ) -> BackendRender:
        del timbre
        duration = _render_duration(voices, tempo)
        if duration > self._max_render_seconds:
            raise ValueError(
                f"Render duration {duration:.1f}s exceeds the "
                f"{self._max_render_seconds:g}s limit."
            )
        soundfont = self._find_soundfont()
        executable = shutil.which("fluidsynth")
        fallback_reason: str | None = None
        if executable and soundfont:
            try:
                return BackendRender(
                    audio=self._render_fluidsynth(
                        executable,
                        soundfont,
                        voices,
                        tempo,
                        duration,
                    ),
                    renderer="fluidsynth",
                )
            except Exception:
                logger.exception("FluidSynth failed; using the in-process renderer")
                fallback_reason = (
                    "FluidSynth failed; used the in-process wavetable renderer."
                )
        else:
            fallback_reason = (
                "FluidSynth or a SoundFont is unavailable; "
                "used the in-process wavetable renderer."
            )
        return BackendRender(
            audio=self._render_wavetable(voices, tempo, duration),
            renderer="wavetable",
            fallback_reason=fallback_reason,
        )

    def _find_soundfont(self) -> Path | None:
        candidates = (
            (self._configured_soundfont,) if self._configured_soundfont else ()
        ) + _SYSTEM_SOUNDFONTS
        return next((path for path in candidates if path.is_file()), None)

    def _render_fluidsynth(
        self,
        executable: str,
        soundfont: Path,
        voices: list[Voice],
        tempo: float,
        duration: float,
    ) -> bytes:
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        token = uuid4().hex
        midi_path = self._runtime_dir / f"{token}.mid"
        wav_path = self._runtime_dir / f"{token}.wav"
        try:
            midi_path.write_bytes(
                self._midi_service.voices_to_midi(
                    voices,
                    tempo=tempo,
                    time_signature=TimeSignature(numerator=4, denominator=4),
                )
            )
            completed = subprocess.run(
                [
                    executable,
                    "-ni",
                    "-g",
                    "0.8",
                    "-r",
                    str(self._sample_rate),
                    "-F",
                    str(wav_path),
                    str(soundfont),
                    str(midi_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=max(30.0, duration * 2 + 10),
            )
            audio = wav_path.read_bytes()
            if not is_wav(audio):
                error = completed.stderr.decode(errors="replace").strip()
                raise RuntimeError(f"FluidSynth did not produce WAV audio: {error}")
            return audio
        finally:
            midi_path.unlink(missing_ok=True)
            wav_path.unlink(missing_ok=True)

    def _render_wavetable(
        self,
        voices: list[Voice],
        tempo: float,
        duration: float,
    ) -> bytes:
        sample_count = max(1, int(np.ceil(max(0.5, duration + 0.1) * self._sample_rate)))
        mix = np.zeros((sample_count, 2), dtype=np.float32)
        voice_count = max(1, len(voices))

        for voice_index, voice in enumerate(voices):
            pan = 0.0 if voice_count == 1 else -0.65 + 1.3 * voice_index / (voice_count - 1)
            left_gain = np.sqrt((1.0 - pan) / 2.0)
            right_gain = np.sqrt((1.0 + pan) / 2.0)
            brightness = 0.8 + 0.1 * voice_index
            for note in voice.notes:
                start_seconds = max(0.0, note.start * 60.0 / tempo)
                end_seconds = max(
                    start_seconds,
                    (note.start + note.duration) * 60.0 / tempo,
                )
                note_seconds = end_seconds - start_seconds
                if note_seconds <= 0:
                    continue
                start = round(start_seconds * self._sample_rate)
                length = max(1, round(note_seconds * self._sample_rate))
                end = min(sample_count, start + length)
                if start >= end:
                    continue

                time = np.arange(end - start, dtype=np.float32) / self._sample_rate
                frequency = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
                signal = np.zeros_like(time)
                partials = (1.0, 0.35 * brightness, 0.16 * brightness, 0.07)
                for harmonic, weight in enumerate(partials, start=1):
                    if frequency * harmonic >= self._sample_rate / 2:
                        break
                    signal += weight * np.sin(
                        2 * np.pi * frequency * harmonic * time + voice_index * 0.23
                    )
                signal /= sum(partials)

                envelope = np.ones_like(signal)
                attack = min(len(envelope), max(1, round(0.018 * self._sample_rate)))
                release = min(len(envelope), max(1, round(0.06 * self._sample_rate)))
                envelope[:attack] *= np.linspace(0.0, 1.0, attack, dtype=np.float32)
                envelope[-release:] *= np.linspace(1.0, 0.0, release, dtype=np.float32)
                amplitude = 0.22 * (note.velocity / 127.0)
                signal *= envelope * amplitude
                mix[start:end, 0] += signal * left_gain
                mix[start:end, 1] += signal * right_gain

        peak = float(np.max(np.abs(mix), initial=0.0))
        if peak > 0.95:
            mix *= 0.95 / peak
        return encode_pcm16_wav(mix, self._sample_rate)


def _render_duration(voices: list[Voice], tempo: float) -> float:
    last_beat = max(
        (
            max(0.0, note.start + note.duration)
            for voice in voices
            for note in voice.notes
        ),
        default=0.0,
    )
    return last_beat * 60.0 / tempo
