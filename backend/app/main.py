from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings
from backend.app.routers import harmony, health
from backend.app.services.engines import EngineService


def create_app(
    *,
    settings: Settings | None = None,
    engine_service: EngineService | None = None,
) -> FastAPI:
    resolved_settings = settings or Settings.from_env()
    application = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
    )
    application.state.settings = resolved_settings
    application.state.engine_service = engine_service or EngineService(
        enable_dev_engine=resolved_settings.enable_dev_engine
    )

    if resolved_settings.cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=list(resolved_settings.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    prefix = "/api/v1"
    application.include_router(health.router, prefix=prefix)
    application.include_router(harmony.router, prefix=prefix)
    return application


logging.basicConfig(level=logging.INFO)
app = create_app()
