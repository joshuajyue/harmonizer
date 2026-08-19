"""Shared API contract for HarmonAIzer v2 (Pydantic mirror of contracts/types.ts).

SOURCE OF TRUTH is contracts/types.ts — change both together.

Core v2 change vs v1: an engine returns *voiced parts* (actual notes, with voice
leading), not just a chord label per beat. Chord labels are metadata; the voices
are the product.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["major", "minor"]
VoiceName = Literal["soprano", "alto", "tenor", "bass"]
Severity = Literal["info", "warning", "error"]


class Note(BaseModel):
    pitch: int = Field(ge=0, le=127, description="Absolute MIDI pitch. Middle C = 60.")
    start: float = Field(description="Offset in quarter-note beats from the start.")
    duration: float = Field(gt=0, description="Length in quarter-note beats.")
    velocity: int = Field(default=80, ge=0, le=127)


class KeySignature(BaseModel):
    tonic: int = Field(ge=0, le=11, description="Pitch class of the tonic. C = 0.")
    mode: Mode
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class TimeSignature(BaseModel):
    # No defaults: a partially-specified time signature is meaningless, and silently
    # filling in 4/4 would misplace every downbeat.
    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)


class Melody(BaseModel):
    notes: list[Note]
    # Required: defaulting the tempo silently renders the result at the wrong speed
    # with no error anywhere, which is worse than rejecting the request.
    tempo: float = Field(gt=0)
    timeSignature: TimeSignature = Field(
        default_factory=lambda: TimeSignature(numerator=4, denominator=4)
    )
    key: Optional[KeySignature] = Field(
        default=None, description="If omitted, the backend detects it and returns it."
    )


class Chord(BaseModel):
    start: float
    duration: float
    roman: str = Field(description='Display form: "V65", "bII", "V/V". The only text the UI renders.')
    root: int = Field(ge=0, le=11)
    quality: str = Field(description='"maj" | "min" | "dim" | "aug" | "dom7" | "maj7" | "min7" | "halfdim7" | "dim7"')
    inversion: int = Field(default=0, ge=0)
    secondaryOf: Optional[int] = Field(
        default=None, description="Set when the chord tonicizes another degree (e.g. V/V)."
    )


class Voice(BaseModel):
    name: VoiceName
    notes: list[Note]


class Violation(BaseModel):
    """A voice-leading or style rule the result breaks. Surfaced in the UI, never hidden."""

    kind: str = Field(description='"parallel_fifths" | "voice_crossing" | "unresolved_leading_tone" | "spacing" | ...')
    severity: Severity
    start: float
    voices: list[VoiceName]
    message: str


class HarmonizeOptions(BaseModel):
    voiceCount: int = Field(default=4, ge=2, le=8)
    temperature: float = Field(default=0.0, ge=0.0, description="0 = deterministic/argmax.")
    seed: Optional[int] = Field(
        default=None, description="Same seed + same input must reproduce the same output."
    )


class HarmonizeRequest(BaseModel):
    melody: Melody
    engine: str
    options: HarmonizeOptions = Field(default_factory=HarmonizeOptions)


class HarmonizeResponse(BaseModel):
    key: KeySignature
    chords: list[Chord]
    voices: list[Voice]
    violations: list[Violation] = Field(default_factory=list)
    engine: str
    latencyMs: float


class EngineInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    learned: bool = Field(description="False for the rule engine, True for learned models.")


class EnginesResponse(BaseModel):
    engines: list[EngineInfo]


class RenderRequest(BaseModel):
    """POST /api/v1/render -> audio/wav."""

    voices: list[Voice]
    tempo: float = Field(gt=0)
    synth: str = Field(default="sf2", description='"sf2" (fast preview) or "ddsp" (neural voice).')
    timbre: Optional[str] = None
