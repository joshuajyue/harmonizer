from fastapi import APIRouter, Depends

from backend.app.config import Settings
from backend.app.dependencies import get_settings
from backend.app.models import HealthResponse

router = APIRouter(tags=["service"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=settings.app_version)
