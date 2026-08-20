import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_wrong_working_directory_reports_correct_launch_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import app.main"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "launched from backend/ as 'app.main'" in completed.stderr
    assert "cd .." in completed.stderr
    assert "uvicorn backend.main:app" in completed.stderr


def test_top_level_main_reports_correct_launch_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import main"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert "launched from inside backend/ as 'main'" in completed.stderr
    assert "cd .." in completed.stderr
    assert "uvicorn backend.main:app" in completed.stderr
