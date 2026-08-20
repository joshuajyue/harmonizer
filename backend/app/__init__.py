"""FastAPI application package."""

if __name__ == "app":
    raise RuntimeError(
        "HarmonAIzer was launched from backend/ as 'app.main'. "
        "Run it from the repository root instead:\n"
        "  cd ..\n"
        "  uvicorn backend.main:app"
    )

from backend.app.main import create_app

__all__ = ["create_app"]
