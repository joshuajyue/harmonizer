from __future__ import annotations

import importlib
import importlib.util
import inspect
import threading

from backend.app.services.synthesis.audio import is_wav
from backend.app.services.synthesis.base import BackendRender, SynthAvailability
from backend.app.services.synthesis.sf2 import SoundFontSynthBackend
from contracts.schema import Voice


class DdspSynthBackend:
    """Lazy bridge to a separately installed DDSP-SVC or RVC adapter."""

    id = "ddsp"
    name = "Neural Voice"
    description = (
        "Optional lazy-loaded DDSP-SVC/RVC adapter; model and hardware dependencies "
        "are kept out of the base service."
    )
    neural = True

    def __init__(
        self,
        *,
        adapter_spec: str | None,
        guide_backend: SoundFontSynthBackend,
        sample_rate: int,
    ) -> None:
        self._adapter_spec = adapter_spec
        self._guide_backend = guide_backend
        self._sample_rate = sample_rate
        self._adapter: object | None = None
        self._load_lock = threading.Lock()
        self._render_lock = threading.Lock()

    def availability(self, timbre: str | None = None) -> SynthAvailability:
        del timbre
        if not self._adapter_spec:
            return SynthAvailability(
                False,
                "Set HARMONIZER_DDSP_ADAPTER=module:object and install its model dependencies.",
            )
        try:
            module_name, _ = self._split_spec()
            if importlib.util.find_spec(module_name) is None:
                return SynthAvailability(
                    False,
                    f"Configured adapter module {module_name!r} is not installed.",
                )
        except (ImportError, ModuleNotFoundError, ValueError) as exc:
            return SynthAvailability(False, str(exc))
        return SynthAvailability(
            True,
            "Configured; model loading and hardware initialization are deferred until render.",
        )

    def render(
        self,
        voices: list[Voice],
        *,
        tempo: float,
        timbre: str | None,
    ) -> BackendRender:
        adapter = self._load_adapter()
        guide = self._guide_backend.render(
            voices,
            tempo=tempo,
            timbre=None,
        ).audio
        with self._render_lock:
            availability = getattr(adapter, "is_available", None)
            if callable(availability) and not availability():
                raise RuntimeError(
                    "The configured neural adapter reports that it is unavailable."
                )
            renderer = getattr(adapter, "render", adapter)
            if not callable(renderer):
                raise TypeError("The configured neural adapter is not callable.")
            audio = renderer(
                voices=voices,
                tempo=tempo,
                timbre=timbre,
                guide_audio=guide,
                sample_rate=self._sample_rate,
            )
        if not isinstance(audio, (bytes, bytearray)) or not is_wav(bytes(audio)):
            raise TypeError("The configured neural adapter did not return WAV bytes.")
        return BackendRender(audio=bytes(audio), renderer="neural-adapter")

    def _split_spec(self) -> tuple[str, str]:
        if not self._adapter_spec or ":" not in self._adapter_spec:
            raise ValueError(
                "HARMONIZER_DDSP_ADAPTER must use the form 'module:object'."
            )
        module_name, object_name = self._adapter_spec.rsplit(":", 1)
        if not module_name or not object_name:
            raise ValueError(
                "HARMONIZER_DDSP_ADAPTER must use the form 'module:object'."
            )
        return module_name, object_name

    def _load_adapter(self) -> object:
        if self._adapter is not None:
            return self._adapter
        with self._load_lock:
            if self._adapter is not None:
                return self._adapter
            module_name, object_name = self._split_spec()
            module = importlib.import_module(module_name)
            adapter = getattr(module, object_name)
            if inspect.isclass(adapter):
                adapter = adapter()
            self._adapter = adapter
            return adapter
