"""Paths as a run writes them: relative to the repository, never this machine's (P12g, G5).

The first full run wrote the absolute path of ``benchmarks/tuned.json`` 1,913 times into
``results.json``, because ``TunedValues.source`` was absolute and reached every episode's
``tuning`` string and every record's ``extra``. It also reached the key of every cached cell (the
tuned values are part of a task's fingerprint), so a clean checkout at another path would not
have found the cells of this one. And an error caught in a worker carried the absolute path of
each frame of its traceback into a verdict and the ledger.

``relative_path`` gives the path relative to the repository root, or the base name when the file is
outside it; ``scrub`` does the same to the paths inside a piece of text, and turns the home folder
into ``~``. Neither looks at anything but the text, so the result is the same on every machine.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

__all__ = ["ROOT", "relative_path", "scrub"]

ROOT: Final[Path] = Path(__file__).resolve().parents[2]

_FRAME: Final[re.Pattern[str]] = re.compile(r'File "([^"]+)"')


def relative_path(path: str | os.PathLike[str]) -> str:
    """``path`` relative to the repository root with forward slashes; its base name if outside."""
    target = Path(path)
    try:
        return target.resolve().relative_to(ROOT).as_posix()
    except (ValueError, OSError):
        return target.name


def _spellings(path: Path) -> list[str]:
    """The ways a folder is written in text: native, forward slashes, doubled backslashes."""
    native = str(path)
    return sorted({native, path.as_posix(), native.replace("\\", "\\\\")}, key=len, reverse=True)


def scrub(text: str) -> str:
    """``text`` with every path it holds made repository-relative (or ``~`` for the home folder).

    A ``File "..."`` frame of a traceback becomes the relative path, or the base name if the file
    is outside the repository. Any other occurrence of the repository's folder is removed, so that
    ``<root>/closed_loop/x.py`` reads ``closed_loop/x.py``, and the home folder becomes ``~``.
    """
    text = _FRAME.sub(lambda m: f'File "{relative_path(m.group(1))}"', text)
    for spelling in _spellings(ROOT):
        text = re.sub(re.escape(spelling) + r"[\\/]{1,2}", "", text, flags=re.IGNORECASE)
        text = re.sub(re.escape(spelling), ".", text, flags=re.IGNORECASE)
    for spelling in _spellings(Path.home()):
        text = re.sub(re.escape(spelling), "~", text, flags=re.IGNORECASE)
    return text
