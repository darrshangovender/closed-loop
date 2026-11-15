"""Repository-shape guards: dependency confusion, import hygiene and the scaffold's own files."""

import ast
import json
import os
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

import closed_loop

REPO = Path(__file__).resolve().parents[2]
SIBLINGS = ("asset-health", "maintenance-agent", "plant-copilot")
# Names that PyPI would resolve to a stranger's project (0001, section 14).
GUARDED = {canonicalize_name(n) for n in (*SIBLINGS, "closed-loop")}
AREAS = (
    "core", "stats", "spec", "plant", "perception", "shield", "proposers", "causal",
    "planning", "search", "audit", "eval", "f2",
)  # fmt: skip
FORBIDDEN_AT_IMPORT = ("asset_health", "maintenance_agent", "plant_copilot", "pandas", "sklearn")
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".ruff_cache"}


def is_local_path(spec: str) -> bool:
    return spec.startswith((".", "/", "~", "file:")) or bool(re.match(r"^[A-Za-z]:[\\/]", spec))


def bare_sibling_refs(specs: Iterable[str]) -> list[str]:
    """Requirement strings that name a guarded project without a direct URL or a local path."""
    bad = []
    for raw in specs:
        spec = raw.strip()
        if not spec or spec.startswith("#"):
            continue
        spec = re.sub(r"^(-e|--editable)\s+", "", spec)
        if spec.startswith("-"):  # other pip options carry no requirement
            continue
        if is_local_path(spec):
            continue
        try:
            req = Requirement(spec)
        except InvalidRequirement:
            bad.append(raw)  # cannot be parsed, so cannot be shown safe
            continue
        if canonicalize_name(req.name) not in GUARDED:
            continue
        if req.url is None or not (req.url.startswith("git+") or is_local_path(req.url)):
            bad.append(raw)
    return bad


INSTALL_LINE = re.compile(r"\b(?:pip|uv)\b.*\b(?:install|add)\b(?P<rest>.*)")
BARE_TOKEN = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?(?:[<>=!~].*)?$")


def bare_names_on_install_line(line: str) -> list[str]:
    """Tokens after ``pip install`` / ``uv add`` that name a guarded project with no URL."""
    stripped = line.strip()
    if stripped.startswith("#"):
        return []
    match = INSTALL_LINE.search(stripped)
    if not match:
        return []
    tokens = [t.strip("\"',") for t in match.group("rest").split()]
    bad = []
    for i, token in enumerate(tokens):
        found = BARE_TOKEN.match(token)
        if not found or canonicalize_name(found.group("name")) not in GUARDED:
            continue
        if i + 1 < len(tokens) and tokens[i + 1] == "@":
            continue  # "name @ url" split on the spaces
        bad.append(token)
    return bad


def pyproject_specs() -> list[str]:
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    specs = list(data["build-system"]["requires"])
    specs += data["project"].get("dependencies", [])
    for extra in data["project"].get("optional-dependencies", {}).values():
        specs += extra
    for group in data.get("dependency-groups", {}).values():
        specs += [s for s in group if isinstance(s, str)]
    return specs


def tracked_text_files(patterns: tuple[str, ...]) -> list[Path]:
    found = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        found += [Path(root) / f for f in files if any(Path(f).match(p) for p in patterns)]
    return found


class TestNoBareSiblingNames:
    def test_pyproject(self):
        assert bare_sibling_refs(pyproject_specs()) == []

    def test_requirements_files(self):
        for path in tracked_text_files(("requirements*.txt",)):
            lines = path.read_text(encoding="utf-8").splitlines()
            assert bare_sibling_refs(lines) == [], path

    def test_ci_and_makefile_do_not_install_a_bare_name(self):
        files = [REPO / "Makefile", *tracked_text_files(("*.yml", "*.yaml"))]
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                assert bare_names_on_install_line(line) == [], (path, line)

    @pytest.mark.parametrize(
        "bad",
        [
            "pip install asset-health",
            "run: pip install -e . maintenance_agent>=0.1",
            "uv pip install 'plant-copilot[x]'",
            "\tpip install closed-loop",
            "uv add asset-health",
        ],
    )
    def test_install_line_checker_rejects_bare_names(self, bad):
        assert bare_names_on_install_line(bad) != []

    @pytest.mark.parametrize(
        "good",
        [
            'pip install -e ".[dev]"',
            "pip install ../asset-health ../maintenance-agent",
            "pip install 'asset-health @ git+https://github.com/darrshangovender/asset-health'",
            "# pip install asset-health is what an attacker wants you to type",
            "pip install numpy",
            "echo asset-health",
        ],
    )
    def test_install_line_checker_accepts_urls_paths_and_comments(self, good):
        assert bare_names_on_install_line(good) == []

    def test_the_integration_extra_names_all_three_siblings_by_url(self):
        data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        extra = data["project"]["optional-dependencies"]["integration"]
        names = {canonicalize_name(Requirement(s).name) for s in extra}
        assert names == {canonicalize_name(s) for s in SIBLINGS}
        assert all(Requirement(s).url.startswith("git+https://") for s in extra)

    # The checker needs teeth: it must reject what it is there to reject.
    @pytest.mark.parametrize(
        "bad",
        [
            "asset-health",
            "asset_health>=0.1",
            "Asset.Health==0.1.1",
            "maintenance-agent[mcp]>=1",
            "plant-copilot",
            "closed-loop[integration]",
            "plant-copilot @ https://example.invalid/plant_copilot.tar.gz",
            "plant-copilot ; python_version >= '3.11'",
            "this is not a requirement !!",
        ],
    )
    def test_checker_rejects_bare_references(self, bad):
        assert bare_sibling_refs([bad]) == [bad]

    @pytest.mark.parametrize(
        "good",
        [
            "asset-health @ git+https://github.com/darrshangovender/asset-health",
            "maintenance-agent @ git+https://github.com/darrshangovender/maintenance-agent@v0.1.1",
            "plant_copilot @ git+ssh://git@github.com/darrshangovender/plant-copilot",
            "-e ../asset-health",
            "../maintenance-agent",
            "./plant-copilot",
            "C:\\work\\asset-health",
            "numpy>=1.26",
            "scipy",
            "# asset-health",
            "",
            "--index-url https://pypi.org/simple",
        ],
    )
    def test_checker_accepts_urls_paths_and_other_packages(self, good):
        assert bare_sibling_refs([good]) == []


def run_child(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=180,
    )


class TestImportHygiene:
    def test_import_is_fast_and_pulls_no_sibling_pandas_or_sklearn(self):
        code = (
            "import json, sys, time\n"
            "t0 = time.perf_counter()\n"
            "import closed_loop\n"
            "dt = time.perf_counter() - t0\n"
            f"bad = [m for m in {FORBIDDEN_AT_IMPORT!r} if m in sys.modules]\n"
            "print(json.dumps({'seconds': dt, 'bad': bad}))\n"
        )
        done = run_child(code)
        assert done.returncode == 0, done.stderr
        out = json.loads(done.stdout)
        assert out["bad"] == []
        assert out["seconds"] < 3.0

    def test_every_module_outside_f2_imports_cleanly_without_siblings_or_pandas(self):
        code = (
            "import importlib, json, pkgutil, sys\n"
            "import closed_loop\n"
            "names = [m.name for m in pkgutil.walk_packages(closed_loop.__path__, 'closed_loop.')\n"
            "         if not m.name.startswith('closed_loop.f2')]\n"
            "for n in names:\n"
            "    importlib.import_module(n)\n"
            f"bad = [m for m in {FORBIDDEN_AT_IMPORT!r} if m in sys.modules]\n"
            "print(json.dumps({'n': len(names), 'bad': bad}))\n"
        )
        done = run_child(code)
        assert done.returncode == 0, done.stderr
        out = json.loads(done.stdout)
        assert out["bad"] == []
        assert out["n"] >= 12  # the areas plus core's modules

    def test_the_forbidden_list_is_really_importable_here_or_the_test_proves_nothing(self):
        # If pandas were not installed, "pandas not in sys.modules" would hold trivially.
        # This is informational: it records which forbidden modules the check could catch.
        code = (
            "import importlib.util, json\n"
            f"print(json.dumps({{m: importlib.util.find_spec(m) is not None "
            f"for m in {FORBIDDEN_AT_IMPORT!r}}}))\n"
        )
        done = run_child(code)
        assert done.returncode == 0, done.stderr
        present = json.loads(done.stdout)
        if not all(present.values()):
            pytest.skip(f"cannot be caught on this machine, not installed: {present}")

    def test_no_test_module_imports_the_forbidden_list_at_the_top(self):
        # A clean install of ".[dev]" has none of them, and a top-level import fails collection of
        # the whole suite instead of skipping one module (the clean-install check of 3 Nov 2026).
        bad = []
        for path in sorted((REPO / "tests").rglob("*.py")):
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    if name.split(".")[0] in FORBIDDEN_AT_IMPORT:
                        bad.append(f"{path.relative_to(REPO).as_posix()}:{node.lineno} {name}")
        assert bad == []


def test_area_packages_exist_and_import():
    for area in AREAS:
        init = REPO / "closed_loop" / area / "__init__.py"
        assert init.is_file(), init
        __import__(f"closed_loop.{area}")


def test_layout_is_flat_not_src():
    assert (REPO / "closed_loop" / "__init__.py").is_file()
    assert not (REPO / "src").exists()


def test_version_is_in_step_with_pyproject():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert closed_loop.__version__ == data["project"]["version"]


def test_init_defines_nothing_but_the_version():
    # submodules appear as attributes once imported; nothing else may be public
    public = {n for n in vars(closed_loop) if not n.startswith("_")}
    assert public <= {"core", "stats", "spec", "plant", "perception", "shield", "proposers",
                      "causal", "planning", "search", "audit", "eval", "f2"}  # fmt: skip


def test_distribution_and_marker_metadata():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    assert project["name"] == "closed-loop"
    assert project["authors"][0]["email"] == "darrshangovender@gmail.com"
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["closed_loop"]
    pytest_cfg = data["tool"]["pytest"]["ini_options"]
    assert pytest_cfg["pythonpath"] == ["."] and pytest_cfg["addopts"] == "-ra -q"
    assert any(m.startswith("f2:") for m in pytest_cfg["markers"])
    assert {"dev", "integration"} <= set(project["optional-dependencies"])
    core = {canonicalize_name(Requirement(s).name) for s in project["dependencies"]}
    assert core == {"numpy", "scipy", "pydantic"}


def test_gitignore_hides_internal_notes_and_the_result_cache():
    lines = (REPO / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "docs/adr/progress/" in lines and "benchmarks/.cache/" in lines
    # the lock stub must stay trackable
    assert "*.lock" not in lines


def test_gitattributes_forces_lf():
    text = (REPO / ".gitattributes").read_text(encoding="utf-8")
    assert "* text=auto eol=lf" in text.splitlines()


def test_siblings_lock_stub_is_valid_json_naming_all_three():
    data = json.loads((REPO / "integration" / "siblings.lock").read_text(encoding="utf-8"))
    assert set(data["siblings"]) == set(SIBLINGS)
    for entry in data["siblings"].values():
        assert entry["source"].startswith("git+https://")


def test_makefile_has_the_promised_targets():
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    for target in ("install", "test", "lint", "bench", "quick", "clean"):
        assert re.search(rf"^{target}:", text, re.MULTILINE), target
    assert "\t" in text  # recipes are tab-indented


def test_ci_workflow_runs_lint_and_tests_on_both_pythons():
    text = (REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert '["3.11", "3.12"]' in text
    assert "ruff check ." in text and "pytest tests/" in text
    assert "continue-on-error" not in text
