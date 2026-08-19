from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings
from backend.app.routers import harmony, health, midi, render, transcription
from backend.app.services.engines import EngineService
from backend.app.services.midi import MidiService
from backend.app.services.synthesis import SynthService
from backend.app.services.synthesis.ddsp import DdspSynthBackend
from backend.app.services.synthesis.sf2 import SoundFontSynthBackend
from backend.app.services.synthesis.world import WorldSynthBackend
from backend.app.services.transcription import TranscriptionService


def create_app(
    *,
    settings: Settings | None = None,
    engine_service: EngineService | None = None,
    midi_service: MidiService | None = None,
    synth_service: SynthService | None = None,
    transcription_service: TranscriptionService | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    resolved_midi_service = midi_service or MidiService(
        max_upload_bytes=resolved_settings.max_upload_bytes
    )
    sf2_backend = SoundFontSynthBackend(
        sample_rate=resolved_settings.sample_rate,
        max_render_seconds=resolved_settings.max_render_seconds,
        runtime_dir=resolved_settings.runtime_dir,
        soundfont_path=resolved_settings.soundfont_path,
        midi_service=resolved_midi_service,
    )
    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
    )
    application.state.settings = resolved_settings
    application.state.engine_service = engine_service or EngineService(
        enable_dev_engine=resolved_settings.enable_dev_engine
    )
    application.state.midi_service = resolved_midi_service
    application.state.synth_service = synth_service or SynthService(
        sf2=sf2_backend,
        ddsp=DdspSynthBackend(
            adapter_spec=resolved_settings.ddsp_adapter,
            guide_backend=sf2_backend,
            sample_rate=resolved_settings.sample_rate,
        ),
        world=WorldSynthBackend(
            timbre_dir=resolved_settings.timbre_dir,
            sample_rate=resolved_settings.sample_rate,
            max_render_seconds=resolved_settings.max_render_seconds,
        ),
    )
    application.state.transcription_service = (
        transcription_service
        or TranscriptionService(
            max_upload_bytes=resolved_settings.max_upload_bytes,
            max_audio_seconds=resolved_settings.max_render_seconds,
        )
    )

    if resolved_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[
                "X-HarmonAIzer-Synth-Requested",
                "X-HarmonAIzer-Synth-Used",
                "X-HarmonAIzer-Renderer",
                "X-HarmonAIzer-Fallback",
                "X-HarmonAIzer-Octave-Shift",
                "X-HarmonAIzer-Detected-Median-Pitch",
            ],
        )

    prefix = "/api/v1"
    application.include_router(health.router, prefix=prefix)
    application.include_router(harmony.router, prefix=prefix)
    application.include_router(render.router, prefix=prefix)
    application.include_router(transcription.router, prefix=prefix)
    application.include_router(midi.router, prefix=prefix)
    return application


logging.basicConfig(level=logging.INFO)
app = create_app()
