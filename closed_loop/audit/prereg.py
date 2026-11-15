"""Pre-registration hash and code hash.

Interface text: docs/adr/0002-interfaces.md, section 3.12; method: docs/adr/0001-architecture.md,
section 12.

The hash is over **normalised bytes**: CRLF becomes LF and spaces and tabs at the end of a line
are removed, so a checkout with Windows line endings or an editor that trims whitespace gives the
same hash as the committed file. Nothing else is normalised: a missing final newline, a
byte-order mark, a lone CR or a changed character all change the hash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "CODE_SUFFIXES",
    "code_sha256",
    "file_sha256",
    "normalise_bytes",
    "normalised_bytes",
    "prereg_sha256",
]

CODE_SUFFIXES: tuple[str, ...] = (".py",)
_SKIP_DIRS = frozenset(
    {"__pycache__", ".git", ".venv", "venv", "node_modules", ".cache", ".hypothesis"}
    | {".pytest_cache", ".mypy_cache", ".ruff_cache"}
)


def normalise_bytes(data: bytes) -> bytes:
    """CRLF to LF, then strip spaces and tabs from the end of every line."""
    lines = data.replace(b"\r\n", b"\n").split(b"\n")
    return b"\n".join(line.rstrip(b" \t") for line in lines)


def normalised_bytes(path: Path) -> bytes:
    """The bytes of ``path`` with CRLF turned into LF and trailing spaces stripped."""
    return normalise_bytes(Path(path).read_bytes())


def prereg_sha256(path: Path) -> str:
    """SHA-256 (lower-case hex) of the normalised bytes of the pre-registration."""
    return hashlib.sha256(normalised_bytes(path)).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of the normalised bytes of any text file (used for ``tuned.json`` and similar)."""
    return prereg_sha256(path)


def _files_under(root: Path, suffixes: tuple[str, ...]) -> list[tuple[str, Path]]:
    if root.is_file():
        return [(root.name, root)]
    if not root.is_dir():
        raise FileNotFoundError(f"code root does not exist: {root}")
    found = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in rel.parts[:-1]):
            continue
        if path.is_file() and path.suffix in suffixes:
            found.append((f"{root.name}/{rel.as_posix()}", path))
    return found


def code_sha256(roots: Sequence[Path], suffixes: tuple[str, ...] = CODE_SUFFIXES) -> str:
    """One hash over every source file under ``roots`` (a directory or a single file).

    Each file contributes its path relative to its root's parent (``closed_loop/core/rng.py``) and
    its normalised bytes, so the hash does not depend on where the repository sits on disk or on
    line endings. Files are taken in sorted order. Caches and virtual environments are skipped.
    Raises ``FileNotFoundError`` for a missing root and ``ValueError`` if no file is found or two
    roots would contribute the same relative path.
    """
    entries: dict[str, Path] = {}
    for root in roots:
        for rel, path in _files_under(Path(root), tuple(suffixes)):
            if rel in entries:
                raise ValueError(f"two code roots contribute the same path {rel!r}")
            entries[rel] = path
    if not entries:
        raise ValueError(f"no {', '.join(suffixes)} files found under {[str(r) for r in roots]}")
    digest = hashlib.sha256()
    for rel in sorted(entries):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(normalised_bytes(entries[rel])).digest())
    return digest.hexdigest()
