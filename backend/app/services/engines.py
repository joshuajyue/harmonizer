from __future__ import annotations

import importlib
import logging
import pkgutil
import threading
from collections.abc import Callable
from time import perf_counter

from contracts.schema import EngineInfo, HarmonizeRequest, HarmonizeResponse
from ml.engines.base import (
    Harmonization,
    HarmonyEngine,
    all_engines,
    get_engine,
)

logger = logging.getLogger(__name__)

DEFAULT_ENGINE_PACKAGES = ("ml.engines", "ml.reharm")


class EngineUnavailableError(RuntimeError):
    def __init__(self, engine_id: str, message: str) -> None:
        super().__init__(message)
        self.engine_id = engine_id


class EngineService:
    """Lazy adapter around the shared ML engine registry."""

    def __init__(
        self,
        *,
        enable_dev_engine: bool = False,
        discover_modules: bool = True,
        engines: list[HarmonyEngine] | None = None,
        engine_provider: Callable[[], list[HarmonyEngine]] = all_engines,
        engine_lookup: Callable[[str], HarmonyEngine] = get_engine,
        engine_packages: tuple[str, ...] = DEFAULT_ENGINE_PACKAGES,
    ) -> None:
        self._enable_dev_engine = enable_dev_engine
        self._discover_modules = discover_modules
        self._fixed_engines = (
            {engine.id: engine for engine in engines}
            if engines is not None
            else None
        )
        self._engine_provider = engine_provider
        self._engine_lookup = engine_lookup
        self._engine_packages = engine_packages
        self._discovered = False
        self._discovery_lock = threading.Lock()

    def _discover(self) -> None:
        if self._fixed_engines is not None or self._discovered:
            return
        with self._discovery_lock:
            if self._discovered:
                return
            if self._discover_modules:
                importlib.invalidate_caches()
                for package_name in self._engine_packages:
                    try:
                        package = importlib.import_module(package_name)
                    except ModuleNotFoundError as exc:
                        if package_name == exc.name or package_name.startswith(
                            f"{exc.name}."
                        ):
                            logger.debug(
                                "Optional harmony engine package %s is not installed",
                                package_name,
                            )
                        else:
                            logger.exception(
                                "Could not import harmony engine package %s",
                                package_name,
                            )
                        continue
                    except Exception:
                        logger.exception(
                            "Could not import harmony engine package %s",
                            package_name,
                        )
                        continue

                    prefix = f"{package.__name__}."
                    for module in pkgutil.iter_modules(package.__path__, prefix):
                        short_name = module.name.rsplit(".", 1)[-1]
                        if module.name.endswith(".base") or short_name.startswith("_"):
                            continue
                        try:
                            importlib.import_module(module.name)
                        except Exception:
                            logger.exception(
                                "Could not import harmony engine module %s",
                                module.name,
                            )
            if self._enable_dev_engine:
                from backend.app.dev_engine import register_development_engine

                register_development_engine()
            self._discovered = True

    def _engines(self) -> list[HarmonyEngine]:
        self._discover()
        if self._fixed_engines is not None:
            return list(self._fixed_engines.values())
        return self._engine_provider()

    def list_engines(self) -> list[EngineInfo]:
        infos: list[EngineInfo] = []
        for engine in sorted(self._engines(), key=lambda item: item.id):
            try:
                available = bool(engine.is_available())
            except Exception:
                logger.exception("Availability check failed for engine %s", engine.id)
                available = False
            infos.append(
                EngineInfo(
                    id=engine.id,
                    name=engine.name,
                    description=engine.description,
                    available=available,
                    learned=engine.learned,
                )
            )
        return infos

    def resolve(self, engine_id: str) -> HarmonyEngine:
        self._discover()
        try:
            if self._fixed_engines is not None:
                engine = self._fixed_engines[engine_id]
            else:
                engine = self._engine_lookup(engine_id)
        except KeyError as exc:
            raise EngineUnavailableError(
                engine_id,
                f"Harmony engine {engine_id!r} is not registered.",
            ) from exc

        try:
            available = engine.is_available()
        except Exception as exc:
            logger.exception("Availability check failed for engine %s", engine_id)
            raise EngineUnavailableError(
                engine_id,
                f"Harmony engine {engine_id!r} could not be initialized.",
            ) from exc
        if not available:
            raise EngineUnavailableError(
                engine_id,
                f"Harmony engine {engine_id!r} is currently unavailable.",
            )
        return engine

    def harmonize(self, request: HarmonizeRequest) -> HarmonizeResponse:
        engine = self.resolve(request.engine)
        started = perf_counter()
        try:
            result = engine.harmonize(
                request.melody,
                voice_count=request.options.voiceCount,
                temperature=request.options.temperature,
                seed=request.options.seed,
            )
        except Exception as exc:
            logger.exception("Harmony engine %s failed", request.engine)
            raise EngineUnavailableError(
                request.engine,
                f"Harmony engine {request.engine!r} failed to generate a result.",
            ) from exc
        latency_ms = (perf_counter() - started) * 1000
        if not isinstance(result, Harmonization):
            raise EngineUnavailableError(
                request.engine,
                f"Harmony engine {request.engine!r} returned an invalid result.",
            )
        try:
            return HarmonizeResponse(
                key=result.key,
                chords=result.chords,
                voices=result.voices,
                violations=result.violations,
                engine=engine.id,
                latencyMs=latency_ms,
            )
        except Exception as exc:
            logger.exception("Harmony engine %s returned contract-invalid data", request.engine)
            raise EngineUnavailableError(
                request.engine,
                f"Harmony engine {request.engine!r} returned an invalid result.",
            ) from exc
