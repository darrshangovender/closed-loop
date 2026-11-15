"""Hash the installed sibling repositories and keep ``integration/siblings.lock`` accurate.

    python scripts/hash_siblings.py            print the installed versions and tree hashes
    python scripts/hash_siblings.py --check    compare them with integration/siblings.lock (exit 1)
    python scripts/hash_siblings.py --write    rewrite the hash fields of the lock from the install

The siblings have no tags and no git history (nothing to pin), so the lock records what was tested:
the installed version and a tree hash of the installed package. The tree hash is the SHA-256 of a
manifest with one line per file, ``<sha256 of the file with CRLF turned into LF>  <posix path>``,
sorted by path, over every file under the package folder except ``__pycache__`` and ``*.pyc``. The
paths are relative to the folder that holds the package, so ``asset_health/pipeline.py`` and so on.
Packet P11c's ``scripts/verify_siblings.py`` uses the same function.

Sibling code is never imported here: the version comes from the installed distribution's metadata
and the files are read as bytes. The sibling names appear only as lock keys and distribution names
read from metadata, never as an install instruction (dependency confusion, 0001 section 14).
"""

from __future__ import annotations

import hashlib
import json
import sys
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "integration" / "siblings.lock"
# lock key -> (distribution name, import name)
SIBLINGS: dict[str, tuple[str, str]] = {
    "asset-health": ("asset-health", "asset_health"),
    "maintenance-agent": ("maintenance-agent", "maintenance_agent"),
    "plant-copilot": ("plant-copilot", "plant_copilot"),
}
HASH_METHOD = (
    "sha256 of a sorted manifest of '<sha256 of file, CRLF as LF>  <posix path relative to the "
    "folder holding the package>' lines, over all files under the installed package folder except "
    "__pycache__ and *.pyc"
)


def package_dir(import_name: str) -> Path | None:
    """The folder of an installed package, found without importing it."""
    spec = find_spec(import_name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def manifest(root: Path) -> list[str]:
    """One ``<digest>  <path>`` line per file under ``root``, sorted by path."""
    base = root.parent
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        lines.append(f"{file_digest(path)}  {path.relative_to(base).as_posix()}")
    return sorted(lines, key=lambda line: line.split("  ", 1)[1])


def tree_sha256(root: Path) -> str:
    return hashlib.sha256(("\n".join(manifest(root)) + "\n").encode("utf-8")).hexdigest()


def describe() -> dict[str, dict[str, object]]:
    """Installed version, file count and tree hash of each sibling (None where not installed)."""
    out: dict[str, dict[str, object]] = {}
    for key, (dist, module) in SIBLINGS.items():
        root = package_dir(module)
        if root is None:
            out[key] = {"version": None, "files": 0, "tree_sha256": None}
            continue
        out[key] = {
            "version": metadata.version(dist),
            "files": len(manifest(root)),
            "tree_sha256": tree_sha256(root),
        }
    return out


def _load() -> dict[str, object]:
    return json.loads(LOCK.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def main(argv: list[str]) -> int:
    installed = describe()
    if not argv:
        for key, info in installed.items():
            print(f"{key}: version {info['version']}, {info['files']} files, {info['tree_sha256']}")
        return 0
    if argv == ["--write"]:
        lock = _load()
        lock["status"] = "pinned by tree hash (no tags or commits exist)"
        lock["hash_method"] = HASH_METHOD
        siblings = lock["siblings"]
        assert isinstance(siblings, dict)
        for key, info in installed.items():
            siblings[key].update(info)
        LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {LOCK}")
        return 0
    if argv == ["--check"]:
        siblings = _load()["siblings"]
        assert isinstance(siblings, dict)
        bad = [
            f"{key}: installed {info['version']} {info['tree_sha256']}, "
            f"lock {siblings[key].get('version')} {siblings[key].get('tree_sha256')}"
            for key, info in installed.items()
            if (info["version"], info["tree_sha256"])
            != (siblings[key].get("version"), siblings[key].get("tree_sha256"))
        ]
        print("\n".join(bad) if bad else "installed siblings match integration/siblings.lock")
        return 1 if bad else 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
