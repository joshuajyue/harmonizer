from __future__ import annotations

import logging
import math

from backend.app.models import SynthInfo
from backend.app.services.synthesis.base import SynthRender
from backend.app.services.synthesis.ddsp import DdspSynthBackend
from backend.app.services.synthesis.sf2 import SoundFontSynthBackend
from backend.app.services.synthesis.world import WorldSynthBackend
from contracts.schema import RenderRequest

logger = logging.getLogger(__name__)


class UnknownSynthError(ValueError):
    pass


class SynthService:
    def __init__(
        self,
        *,
        sf2: SoundFontSynthBackend,
        ddsp: DdspSynthBackend,
        world: WorldSynthBackend,
        max_notes: int,
        max_work_seconds: float,
    ) -> None:
        self._sf2 = sf2
        self._ddsp = ddsp
        self._world = world
        self._max_notes = max_notes
        self._max_work_seconds = max_work_seconds

    def capabilities(self) -> list[SynthInfo]:
        sf2_status = self._sf2.availability()
        ddsp_status = self._ddsp.availability()
        return [
            SynthInfo(
                id="sf2",
                name=self._sf2.name,
                description=self._sf2.description,
                available=sf2_status.available,
                neural=False,
                reason=sf2_status.reason,
            ),
            SynthInfo(
                id="ddsp",
                name=self._ddsp.name,
                description=self._ddsp.description,
                available=ddsp_status.available,
                neural=True,
                fallback="WORLD with a configured timbre, then sf2",
                reason=ddsp_status.reason,
                timbres=self._world.list_timbres(),
            ),
        ]

    def render(self, request: RenderRequest) -> SynthRender:
        self._validate_work(request)
        requested = request.synth.lower()
        if requested == "sf2":
            rendered = self._sf2.render(
                request.voices,
                tempo=request.tempo,
                timbre=request.timbre,
            )
            return SynthRender(
                audio=rendered.audio,
                requested="sf2",
                used="sf2",
                renderer=rendered.renderer,
                fallback_reason=rendered.fallback_reason,
            )
        if requested != "ddsp":
            raise UnknownSynthError(
                f"Unknown synth {request.synth!r}; expected 'sf2' or 'ddsp'."
            )

        fallback_reasons: list[str] = []
        neural_status = self._ddsp.availability(request.timbre)
        if neural_status.available:
            try:
                rendered = self._ddsp.render(
                    request.voices,
                    tempo=request.tempo,
                    timbre=request.timbre,
                )
                return SynthRender(
                    audio=rendered.audio,
                    requested="ddsp",
                    used="ddsp",
                    renderer=rendered.renderer,
                )
            except Exception:
                logger.exception("Neural render failed; trying explicit fallbacks")
                fallback_reasons.append("the configured neural adapter failed")
        elif neural_status.reason:
            fallback_reasons.append(neural_status.reason)

        world_status = self._world.availability(request.timbre)
        if world_status.available:
            try:
                rendered = self._world.render(
                    request.voices,
                    tempo=request.tempo,
                    timbre=request.timbre,
                )
                return SynthRender(
                    audio=rendered.audio,
                    requested="ddsp",
                    used="world",
                    renderer=rendered.renderer,
                    fallback_reason="; ".join(fallback_reasons) or "neural backend unavailable",
                )
            except Exception:
                logger.exception("WORLD render failed; using sf2")
                fallback_reasons.append("WORLD resynthesis failed")
        elif world_status.reason:
            fallback_reasons.append(world_status.reason)

        rendered = self._sf2.render(
            request.voices,
            tempo=request.tempo,
            timbre=None,
        )
        if rendered.fallback_reason:
            fallback_reasons.append(rendered.fallback_reason)
        return SynthRender(
            audio=rendered.audio,
            requested="ddsp",
            used="sf2",
            renderer=rendered.renderer,
            fallback_reason="; ".join(fallback_reasons) or "neural backend unavailable",
        )

    def _validate_work(self, request: RenderRequest) -> None:
        notes = [note for voice in request.voices for note in voice.notes]
        if len(notes) > self._max_notes:
            raise ValueError(
                f"Render contains {len(notes)} notes; the limit is {self._max_notes}."
            )
        work_seconds = sum(note.duration * 60.0 / request.tempo for note in notes)
        if not math.isfinite(work_seconds) or work_seconds > self._max_work_seconds:
            raise ValueError(
                f"Render synthesis work is {work_seconds:.1f} note-seconds; "
                f"the limit is {self._max_work_seconds:g}."
            )
