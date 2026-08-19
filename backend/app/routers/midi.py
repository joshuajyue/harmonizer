from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from backend.app.dependencies import get_midi_service, get_settings
from backend.app.config import Settings
from backend.app.services.midi import MidiConversionError, MidiService
from contracts.schema import HarmonizeResponse, Melody

router = APIRouter(prefix="/midi", tags=["midi"])


@router.post("/import", response_model=Melody)
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


@router.post("/export", response_class=Response)
async def export_midi(
    harmonization: HarmonizeResponse,
    tempo: float = Query(default=90.0, gt=0),
    service: MidiService = Depends(get_midi_service),
) -> Response:
    data = await run_in_threadpool(
        service.export_harmonization,
        harmonization,
        tempo=tempo,
    )
    return Response(
        content=data,
        media_type="audio/midi",
        headers={"Content-Disposition": 'attachment; filename="harmonized.mid"'},
    )
