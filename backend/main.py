"""Uvicorn entry point."""

try:
    from backend.app.main import app
except ModuleNotFoundError as exc:
    if exc.name != "backend":
        raise
    raise RuntimeError(
        "HarmonAIzer was launched from inside backend/ as 'main'. "
        "Run it from the repository root instead:\n"
        "  cd ..\n"
        "  uvicorn backend.main:app"
    ) from exc

__all__ = ["app"]
