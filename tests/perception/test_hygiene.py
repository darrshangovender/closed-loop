"""P4: what the perception modules may import, and the held-out seeds they must not touch."""

import ast
import sys
from pathlib import Path

import pytest

PERCEPTION = Path(__file__).resolve().parents[2] / "closed_loop" / "perception"
MODULES = sorted(p for p in PERCEPTION.glob("*.py"))
FORBIDDEN_TOP = {
    "pandas",
    "sklearn",
    "asset_health",
    "maintenance_agent",
    "plant_copilot",
    "requests",
    "httpx",
    "socket",
}
HELDOUT_NAMES = {"heldout_seed", "f2_seed", "HeldoutGuard", "Ledger", "prereg_sha256"}


def imports_of(path: Path) -> list[tuple[str, list[str]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, [a.name for a in node.names]))
    return out


def test_the_perception_modules_exist():
    names = {p.name for p in MODULES}
    assert {"estimator.py", "labels.py", "coverage.py", "conformal.py"} <= names
    assert {"contract.py", "monitor.py"} <= names


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_sibling_pandas_sklearn_or_network_import(path):
    for module, _ in imports_of(path):
        assert module.split(".")[0] not in FORBIDDEN_TOP, f"{path.name} imports {module}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_only_the_open_seed_roles_are_imported_from_audit(path):
    """Perception reads the fixed seed roles (dev, calib). It never reaches the guard, the ledger
    or the hash-derived seeds, so a held-out seed cannot reach a quantile from here."""
    for module, names in imports_of(path):
        if module.startswith("closed_loop.audit"):
            assert module == "closed_loop.audit.seeds", f"{path.name} imports {module}"
            assert set(names) <= {"DEV_SEEDS", "CALIB_SEEDS"}, f"{path.name} imports {names}"


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_held_out_machinery_is_named(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not (used & HELDOUT_NAMES), f"{path.name} touches {used & HELDOUT_NAMES}"


def test_no_module_draws_from_the_global_numpy_or_python_random():
    for path in MODULES:
        text = path.read_text(encoding="utf-8")
        assert "np.random." not in text.replace("np.random.Generator", ""), path.name
        assert "import random" not in text, path.name
        assert "time.time(" not in text and "datetime" not in text, path.name


def test_the_package_imports_and_keeps_the_forbidden_modules_out():
    import subprocess

    code = (
        "import sys, time\n"
        "t = time.perf_counter()\n"
        "import closed_loop.perception.estimator, closed_loop.perception.labels\n"
        "import closed_loop.perception.coverage, closed_loop.perception.conformal\n"
        "import closed_loop.perception.contract, closed_loop.perception.monitor\n"
        "dt = time.perf_counter() - t\n"
        "bad = [m for m in ('pandas','sklearn','asset_health','maintenance_agent','plant_copilot')"
        " if m in sys.modules]\n"
        "print(dt, bad)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PERCEPTION.parents[1],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert done.returncode == 0, done.stderr
    seconds, bad = done.stdout.split(" ", 1)
    assert bad.strip() == "[]"
    # numpy alone is 0.3 to 1.5 s on this laptop; the guard is against a 40 s sibling import
    assert float(seconds) < 10.0
