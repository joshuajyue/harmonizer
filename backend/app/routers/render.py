from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from backend.app.dependencies import get_synth_service
from backend.app.models import SynthsResponse
from backend.app.services.synthesis.service import SynthService, UnknownSynthError
from contracts.schema import RenderRequest

router = APIRouter(tags=["audio"])


@router.get("/synths", response_model=SynthsResponse)
def list_synths(
    service: SynthService = Depends(get_synth_service),
) -> SynthsResponse:
    return SynthsResponse(synths=service.capabilities())


@router.post(
    "/render",
    response_class=Response,
    responses={
        200: {
            "content": {
                "audio/wav": {
                    "schema": {"type": "string", "format": "binary"},
                }
            },
            "description": "Rendered WAV audio",
        }
    },
)
async def render(
    request: RenderRequest,
    service: SynthService = Depends(get_synth_service),
) -> Response:
    try:
        rendered = await run_in_threadpool(service.render, request)
    except (UnknownSynthError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    headers = {
        "Content-Disposition": 'inline; filename="harmonized.wav"',
        "X-HarmonAIzer-Synth-Requested": rendered.requested,
        "X-HarmonAIzer-Synth-Used": rendered.used,
        "X-HarmonAIzer-Renderer": rendered.renderer,
    }
    if rendered.fallback_reason:
        headers["X-HarmonAIzer-Fallback"] = _safe_header_value(
            rendered.fallback_reason
        )
    return Response(
        content=rendered.audio,
        media_type="audio/wav",
        headers=headers,
    )


def _safe_header_value(value: str, limit: int = 512) -> str:
    return "".join(
        character if 0x20 <= ord(character) <= 0x7E else "?"
        for character in value
    )[:limit]
