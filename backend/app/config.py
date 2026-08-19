from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "HarmonAIzer API"
    app_version: str = "2.0.0"
    enable_dev_engine: bool = False
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    max_upload_bytes: int = 25 * 1024 * 1024
    max_render_seconds: float = 180.0
    sample_rate: int = 44_100
    runtime_dir: Path = PROJECT_ROOT / "backend" / ".runtime"
    soundfont_path: Path | None = None
    ddsp_adapter: str | None = None
    timbre_dir: Path = PROJECT_ROOT / "backend" / "timbres"

    @classmethod
    def from_env(cls) -> Settings:
        origins = tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "").split(",")
            if origin.strip()
        )
        soundfont = os.getenv("HARMONIZER_SOUNDFONT")
        return cls(
            enable_dev_engine=_env_bool("HARMONIZER_ENABLE_DEV_ENGINE"),
            cors_origins=origins,
            max_upload_bytes=int(os.getenv("HARMONIZER_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)),
            max_render_seconds=float(os.getenv("HARMONIZER_MAX_RENDER_SECONDS", 180.0)),
            sample_rate=int(os.getenv("HARMONIZER_SAMPLE_RATE", 44_100)),
            runtime_dir=Path(
                os.getenv(
                    "HARMONIZER_RUNTIME_DIR",
                    str(PROJECT_ROOT / "backend" / ".runtime"),
                )
            ),
            soundfont_path=Path(soundfont) if soundfont else None,
            ddsp_adapter=os.getenv("HARMONIZER_DDSP_ADAPTER"),
            timbre_dir=Path(
                os.getenv(
                    "HARMONIZER_TIMBRE_DIR",
                    str(PROJECT_ROOT / "backend" / "timbres"),
                )
            ),
        )
