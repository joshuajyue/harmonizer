from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from contracts.schema import Voice


@dataclass(frozen=True)
class SynthAvailability:
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class BackendRender:
    audio: bytes
    renderer: str


@dataclass(frozen=True)
class SynthRender:
    audio: bytes
    requested: str
    used: str
    renderer: str
    fallback_reason: str | None = None


class SynthBackend(Protocol):
    id: str
    name: str
    description: str
    neural: bool

    def availability(self, timbre: str | None = None) -> SynthAvailability: ...

    def render(
        self,
        voices: list[Voice],
        *,
        tempo: float,
        timbre: str | None,
    ) -> BackendRender: ...
