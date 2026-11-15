"""Check the installed sibling repositories against ``integration/siblings.lock``.

    python scripts/verify_siblings.py            a table; exit 0 if all match, 1 if one differs,
                                                 2 if one is not installed
    python scripts/verify_siblings.py --json     the same as JSON on standard output

For each of asset-health, maintenance-agent and plant-copilot it prints the installed version and
the tree hash of the installed package, the version and hash the lock records, and whether the
installed copy is identical to the repository source that sits beside this repository, by the hash
of its Python files (a missing source folder is reported, not an error). The siblings have no tags
and no git history, so the lock is a record of what was tested; a mismatch means the F2 findings and
episodes were measured on other code than the one installed now, and ``docs/FINDINGS.md`` has to be
re-read.

The tree hash is the one of ``scripts/hash_siblings.py`` (the same function, loaded from that file),
which reads the files as bytes and never imports a sibling. Nothing here installs anything; the
siblings are installed from a local path or a direct URL, never from a bare package name.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = ROOT.parent  # the repositories sit side by side


def hashing() -> ModuleType:
    """``scripts/hash_siblings.py`` as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "hash_siblings", ROOT / "scripts" / "hash_siblings.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("hash_siblings", module)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class Row:
    sibling: str
    status: str  # "match" | "differs" | "missing" | "not locked"
    installed_version: str | None
    locked_version: str | None
    installed_tree: str | None
    locked_tree: str | None
    source: str  # "identical" | "differs" | "absent"


def verify(
    lock: dict[str, Any],
    installed: dict[str, dict[str, Any]],
    source_trees: dict[str, str | None],
    installed_python: dict[str, str | None] | None = None,
) -> list[Row]:
    """Compare ``installed`` (version and tree hash per sibling) with ``lock``.

    ``source_trees`` holds the hash of the Python files of the repository source beside this one
    (None when that folder does not exist) and ``installed_python`` the same hash for the installed
    package; without it the full tree hash of the install is compared, which is only right for a
    package whose wheel and repository folder hold the same files.
    """
    rows: list[Row] = []
    locked = lock["siblings"]
    for key, info in installed.items():
        entry = locked.get(key)
        tree = info.get("tree_sha256")
        src = source_trees.get(key)
        mine = tree if installed_python is None else installed_python.get(key)
        if tree is None:
            status = "missing"
        elif entry is None:
            status = "not locked"
        elif (info.get("version"), tree) == (entry.get("version"), entry.get("tree_sha256")):
            status = "match"
        else:
            status = "differs"
        rows.append(
            Row(
                key,
                status,
                info.get("version"),
                None if entry is None else entry.get("version"),
                tree,
                None if entry is None else entry.get("tree_sha256"),
                "absent" if src is None else ("identical" if src == mine else "differs"),
            )
        )
    return rows


def exit_code(rows: list[Row]) -> int:
    if any(r.status in ("missing", "not locked") for r in rows):
        return 2
    return 0 if all(r.status == "match" for r in rows) else 1


def python_tree(tool: ModuleType, folder: Path) -> str:
    """The tree hash of the Python files under ``folder`` (a wheel may carry data files that the
    repository keeps elsewhere, as plant-copilot does with its corpus)."""
    lines = [line for line in tool.manifest(folder) if line.endswith(".py")]
    manifest = "\n".join(lines) + "\n"
    return str(tool.hashlib.sha256(manifest.encode("utf-8")).hexdigest())


def source_trees(tool: ModuleType) -> dict[str, str | None]:
    """Per sibling, the Python-file hash of the repository source beside this repository."""
    out: dict[str, str | None] = {}
    for key, (_dist, module) in tool.SIBLINGS.items():
        folder = REPOSITORIES / key / module
        out[key] = python_tree(tool, folder) if folder.is_dir() else None
    return out


def installed_python_trees(tool: ModuleType) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, (_dist, module) in tool.SIBLINGS.items():
        folder = tool.package_dir(module)
        out[key] = None if folder is None else python_tree(tool, folder)
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)
    tool = hashing()
    lock = json.loads(tool.LOCK.read_text(encoding="utf-8"))
    rows = verify(lock, tool.describe(), source_trees(tool), installed_python_trees(tool))
    if args.json:
        print(json.dumps([asdict(r) for r in rows], indent=2))
    else:
        for r in rows:
            print(
                f"{r.sibling:18s} {r.status:10s} installed {r.installed_version} "
                f"{(r.installed_tree or '-')[:12]}  lock {r.locked_version} "
                f"{(r.locked_tree or '-')[:12]}  Python sources vs repository: {r.source}"
            )
        code = exit_code(rows)
        if code == 2:
            print("a sibling is not installed (or not in the lock): F2 tests are skipped")
        elif code == 1:
            print("an installed sibling differs from the lock: re-read docs/FINDINGS.md")
    return exit_code(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
