from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
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

OCTAVE_SHIFT_HEADER = "X-HarmonAIzer-Octave-Shift"
DETECTED_MEDIAN_HEADER = "X-HarmonAIzer-Detected-Median-Pitch"


@router.post(
    "/transcribe",
    response_model=Melody,
    responses={
        200: {
            "headers": {
                OCTAVE_SHIFT_HEADER: {
                    "description": "Global whole-octave shift applied to every note.",
                    "schema": {"type": "string", "example": "+2"},
                },
                DETECTED_MEDIAN_HEADER: {
                    "description": "Median MIDI pitch before octave normalization.",
                    "schema": {"type": "string", "example": "43.5"},
                },
            }
        }
    },
)
async def transcribe(
    response: Response,
    audio: UploadFile | None = File(default=None),
    file: UploadFile | None = File(default=None),
    tempo: float | None = Query(default=None, gt=0),
    numerator: int | None = Query(default=None, gt=0),
    denominator: int | None = Query(default=None, gt=0),
    normalize_octave: bool = Query(
        default=True,
        alias="normalizeOctave",
        description="Shift the whole melody by octaves toward MIDI range 60-79.",
    ),
    octave_shift: int | None = Query(
        default=None,
        alias="octaveShift",
        description="Force a signed whole-octave shift; overrides normalizeOctave.",
    ),
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
        result = await run_in_threadpool(
            service.transcribe,
            data,
            tempo=tempo,
            time_signature=TimeSignature(
                numerator=numerator if numerator is not None else 4,
                denominator=denominator if denominator is not None else 4,
            ),
            normalize_octave=normalize_octave,
            octave_shift=octave_shift,
        )
    except (AudioDecodeError, TranscriptionError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response.headers[OCTAVE_SHIFT_HEADER] = (
        f"{result.octave_shift:+d}" if result.octave_shift else "0"
    )
    response.headers[DETECTED_MEDIAN_HEADER] = f"{result.detected_median_pitch:g}"
    return result.melody
