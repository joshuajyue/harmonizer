"""Harmonization engines. Importing a module here registers its engine.

The backend discovers engines by importing every module in this package, so each
module must self-register at import time and must stay cheap to import — a
neural engine loads its checkpoint lazily, never at import.
"""

from .base import Harmonization, HarmonyEngine, all_engines, get_engine, register

__all__ = ["Harmonization", "HarmonyEngine", "all_engines", "get_engine", "register"]
