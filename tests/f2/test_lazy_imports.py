"""F2 is an extra: importing it costs nothing and pulls in no sibling, no pandas, no scikit-learn.

These tests carry no ``f2`` marker on purpose: they must pass on a machine without the siblings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN = ("asset_health", "maintenance_agent", "plant_copilot", "pandas", "sklearn")
MODULES = ("render", "splice", "alert_source", "cmms", "human", "agent")


def modules_loaded_after(statement: str) -> list[str]:
    code = (
        f"{statement}\nimport json, sys\n"
        f"print(json.dumps(sorted(m for m in {FORBIDDEN!r} if m in sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert isinstance(loaded, list)
    return [str(m) for m in loaded]


def test_closed_loop_itself_is_free_of_siblings() -> None:
    assert modules_loaded_after("import closed_loop") == []


def test_every_f2_module_imports_without_loading_a_sibling_or_pandas() -> None:
    statement = "\n".join(f"import closed_loop.f2.{m}" for m in MODULES)
    assert modules_loaded_after(statement) == []


def test_the_f2_package_exports_the_unavailable_error() -> None:
    from closed_loop.f2 import SIBLING_MODULES, F2Unavailable

    assert issubclass(F2Unavailable, ImportError)
    assert set(SIBLING_MODULES) == {"asset_health", "maintenance_agent", "plant_copilot"}
