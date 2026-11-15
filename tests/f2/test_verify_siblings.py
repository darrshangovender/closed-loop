"""P11c: ``scripts/verify_siblings.py`` compares the installed siblings with the lock."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCK = {
    "siblings": {
        "asset-health": {"version": "0.1.1", "tree_sha256": "aa"},
        "maintenance-agent": {"version": "0.1.1", "tree_sha256": "bb"},
    }
}


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "verify_siblings", ROOT / "scripts/verify_siblings.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_siblings"] = module  # a dataclass looks its module up here
    spec.loader.exec_module(module)
    return module


def info(version: str | None, tree: str | None) -> dict[str, object]:
    return {"version": version, "files": 3, "tree_sha256": tree}


class TestVerify:
    def test_equal_version_and_hash_match(self, tool: ModuleType) -> None:
        rows = tool.verify(LOCK, {"asset-health": info("0.1.1", "aa")}, {"asset-health": "aa"})
        assert [(r.status, r.source) for r in rows] == [("match", "identical")]
        assert tool.exit_code(rows) == 0

    def test_a_changed_file_is_a_difference(self, tool: ModuleType) -> None:
        rows = tool.verify(LOCK, {"asset-health": info("0.1.1", "zz")}, {"asset-health": "aa"})
        assert (rows[0].status, rows[0].source) == ("differs", "differs")
        assert tool.exit_code(rows) == 1

    def test_a_changed_version_alone_is_a_difference(self, tool: ModuleType) -> None:
        rows = tool.verify(LOCK, {"asset-health": info("0.2.0", "aa")}, {"asset-health": None})
        assert (rows[0].status, rows[0].source) == ("differs", "absent")

    def test_a_sibling_that_is_not_installed_is_missing_and_outranks_a_difference(
        self, tool: ModuleType
    ) -> None:
        rows = tool.verify(
            LOCK,
            {"asset-health": info("0.1.1", "zz"), "maintenance-agent": info(None, None)},
            {"asset-health": None, "maintenance-agent": None},
        )
        assert [r.status for r in rows] == ["differs", "missing"]
        assert tool.exit_code(rows) == 2

    def test_a_sibling_the_lock_does_not_know_is_not_locked(self, tool: ModuleType) -> None:
        rows = tool.verify(LOCK, {"plant-copilot": info("0.1.1", "cc")}, {"plant-copilot": "cc"})
        assert rows[0].status == "not locked" and tool.exit_code(rows) == 2


@pytest.mark.f2
def test_the_installed_siblings_match_the_lock() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_siblings.py"), "--json"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    rows = json.loads(out.stdout)
    assert out.returncode == 0, out.stdout
    assert [r["sibling"] for r in rows] == ["asset-health", "maintenance-agent", "plant-copilot"]
    assert all(r["status"] == "match" for r in rows)
