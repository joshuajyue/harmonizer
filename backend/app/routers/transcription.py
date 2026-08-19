from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.app.config import Settings
from backend.app.dependencies import (
    get_settings,
    get_transcription_service,
)
from backend.app.services.transcription import (
    AudioDecodeError,
    TranscriptionError,
    TranscriptionService,
)
from contracts.schema import Melody, TimeSignature

router = APIRouter(tags=["audio"])


@router.post("/transcribe", response_model=Melody)
async def transcribe(
    audio: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
    tempo: float | None = Query(default=None, gt=0),
    numerator: int | None = Query(default=None, gt=0),
    denominator: int | None = Query(default=None, gt=0),
    settings: Settings = Depends(get_settings),
    service: TranscriptionService = Depends(get_transcription_service),
) -> Melody:
    upload = audio or file
    if upload is None:
        raise HTTPException(status_code=422, detail="Upload an audio file.")
    if (numerator is None) != (denominator is None):
        raise HTTPException(
            status_code=422,
            detail="Provide both time-signature numerator and denominator.",
        )
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="The uploaded audio file is too large.")
    try:
        return await run_in_threadpool(
            service.transcribe,
            data,
            tempo=tempo,
            time_signature=TimeSignature(
                numerator=numerator if numerator is not None else 4,
                denominator=denominator if denominator is not None else 4,
            ),
        )
    except (AudioDecodeError, TranscriptionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
