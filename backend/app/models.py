from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    engine: str | None = None


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class SynthInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool
    neural: bool
    fallback: str | None = None
    reason: str | None = None
    timbres: list[str] = Field(default_factory=list)


class SynthsResponse(BaseModel):
    synths: list[SynthInfo]
