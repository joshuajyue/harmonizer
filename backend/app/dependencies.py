from typing import cast

from fastapi import Request

from backend.app.config import Settings
from backend.app.services.engines import EngineService
from backend.app.services.midi import MidiService
from backend.app.services.synthesis import SynthService
from backend.app.services.transcription import TranscriptionService


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_engine_service(request: Request) -> EngineService:
    return cast(EngineService, request.app.state.engine_service)


def get_midi_service(request: Request) -> MidiService:
    return cast(MidiService, request.app.state.midi_service)


def get_synth_service(request: Request) -> SynthService:
    return cast(SynthService, request.app.state.synth_service)


def get_transcription_service(request: Request) -> TranscriptionService:
    return cast(TranscriptionService, request.app.state.transcription_service)
