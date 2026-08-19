from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool

from backend.app.dependencies import get_engine_service
from backend.app.models import ErrorResponse
from backend.app.services.engines import EngineService, EngineUnavailableError
from contracts.schema import EnginesResponse, HarmonizeRequest, HarmonizeResponse

router = APIRouter(tags=["harmony"])


@router.get("/engines", response_model=EnginesResponse)
def list_engines(
    service: EngineService = Depends(get_engine_service),
) -> EnginesResponse:
    return EnginesResponse(engines=service.list_engines())


@router.post(
    "/harmonize",
    response_model=HarmonizeResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}},
)
async def harmonize(
    request: HarmonizeRequest,
    service: EngineService = Depends(get_engine_service),
) -> HarmonizeResponse:
    try:
        return await run_in_threadpool(service.harmonize, request)
    except EngineUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "engine_unavailable",
                "message": str(exc),
                "engine": exc.engine_id,
            },
        ) from exc
