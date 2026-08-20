"""Which engines this suite is entitled to make assertions about.

Every test module here used to parametrise over `all_engines()`. That reads the
*global* registry, and the registry is process-wide: any module imported earlier
in the same pytest run contributes to it. Once `ml/reharm/` arrived with its own
engines, collection order (`reharm` sorts before `tests`) silently swept them
into 31 of this suite's parametrisations.

The failures were real bugs in one case and category errors in the rest, but
both outcomes are wrong for this suite to be reporting. These tests encode
*chorale* invariants — no chord extensions, no substitution provenance,
common-practice structural closure, defect rates calibrated against Bach. A
reharmonizer is supposed to violate all four; that is what it is for. Policing
another package's engines against this package's aesthetic is a category error,
and it puts a red suite in front of whoever runs the tests next for a reason
that has nothing to do with them.

So the owned set is explicit rather than discovered. `test_owned_set_is_current`
below keeps it honest in the only direction that can rot: a new engine added to
`ml/engines/` and not listed here fails loudly rather than going untested.
"""

from __future__ import annotations

import ml.engines.baselines  # noqa: F401
import ml.engines.neural  # noqa: F401
import ml.engines.rules  # noqa: F401
from ml.engines.base import HarmonyEngine, all_engines

#: Engines defined in `ml/engines/`. Anything else in the registry belongs to
#: another package and is deliberately out of scope for this suite.
OWNED_ENGINE_IDS = frozenset(
    {"rules", "neural", "neural_vl", "neural_refine", "fixed_thirds"}
)


def chorale_engines(*, exclude: frozenset[str] | set[str] = frozenset()) -> list[HarmonyEngine]:
    """The available engines this suite owns, in registry order."""
    return [
        engine
        for engine in all_engines()
        if engine.id in OWNED_ENGINE_IDS
        and engine.id not in exclude
        and engine.is_available()
    ]


def engine_ids(engines: list[HarmonyEngine]) -> list[str]:
    return [engine.id for engine in engines]
