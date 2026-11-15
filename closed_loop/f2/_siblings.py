"""Lazy access to the three sibling repositories and pandas (F2 only).

Nothing here imports a sibling at module level, so ``import closed_loop.f2`` works without them and
every public F2 function fails with ``F2Unavailable`` (docs/adr/0002, section 3.13) when one is
missing. The siblings are reached through ``importlib`` and typed as ``Any``: the repository's own
code is type-checked without them installed, and the F2 tests are what catches a misuse.

Dependency confusion: the siblings are never named bare in an install hint. Direct URLs or local
paths only (docs/adr/0001, section 14).
"""

from __future__ import annotations

import importlib
from typing import Any

SIBLING_MODULES = ("asset_health", "maintenance_agent", "plant_copilot")

_HINT = (
    "F2 needs the sibling repositories installed from a local path or a direct URL "
    "(never a bare package name: those are unclaimed on PyPI). See the README, section on F2."
)


class F2Unavailable(ImportError):
    """A sibling repository (or pandas) needed for F2 is not installed."""


def load(module: str) -> Any:
    """Import ``module`` (for example ``"asset_health.pipeline"``) or raise ``F2Unavailable``."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise F2Unavailable(f"cannot import {module}: {exc}. {_HINT}") from exc
