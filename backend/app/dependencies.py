from typing import cast

from fastapi import Request

from backend.app.config import Settings
from backend.app.services.engines import EngineService


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_engine_service(request: Request) -> EngineService:
    return cast(EngineService, request.app.state.engine_service)
