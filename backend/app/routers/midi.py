from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from backend.app.config import Settings
from backend.app.dependencies import get_midi_service, get_settings
from backend.app.services.midi import MidiConversionError, MidiService
from contracts.schema import HarmonizeResponse, Melody, TimeSignature

router = APIRouter(prefix="/midi", tags=["midi"])


@router.post("/import", response_model=Melody, response_model_exclude_none=True)
async def import_midi(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: MidiService = Depends(get_midi_service),
) -> Melody:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The uploaded MIDI file is too large.")
    try:
        return await run_in_threadpool(service.import_melody, data)
    except MidiConversionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/export",
    response_class=Response,
    responses={
        200: {
            "content": {
                "audio/midi": {
                    "schema": {"type": "string", "format": "binary"},
                }
            },
            "description": "Exported Standard MIDI File",
        }
    },
)
async def export_midi(
    harmonization: HarmonizeResponse,
    tempo: float = Query(ge=4, le=400),
    numerator: int = Query(gt=0, le=255),
    denominator: int = Query(gt=0, le=128),
    service: MidiService = Depends(get_midi_service),
) -> Response:
    try:
        data = await run_in_threadpool(
            service.export_harmonization,
            harmonization,
            tempo=tempo,
            time_signature=TimeSignature(
                numerator=numerator,
                denominator=denominator,
            ),
        )
    except MidiConversionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return Response(
        content=data,
        media_type="audio/midi",
        headers={"Content-Disposition": 'attachment; filename="harmonized.mid"'},
    )
