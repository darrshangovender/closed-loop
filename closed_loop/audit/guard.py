"""Gates that stand between a run and the things it must not touch.

Interface text: docs/adr/0002-interfaces.md, section 3.12; method: docs/adr/0001-architecture.md,
sections 12 and 13.

* ``HeldoutGuard`` hands out held-out seeds only when the ledger holds a freeze entry for the
  current pre-registration hash, and writes a ``heldout-use`` entry *before* returning the seeds,
  so a crash after the call still leaves the use on record.
* ``require_headroom`` refuses to start a heavy run when the Windows commit charge is at or above
  a limit (88% by default). The reader is a parameter, so tests do not depend on the machine.
* ``HeavyRunLock`` is the owner's shared heavy-run lock file: refuse while a recent lock exists,
  report an old one as stale and leave it alone, delete our own lock on the way out.
"""

from __future__ import annotations

import ctypes
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from closed_loop.audit.ledger import Ledger, LedgerError
from closed_loop.audit.seeds import f2_seed, heldout_seed, search_seed

__all__ = [
    "COMMIT_LIMIT_PCT",
    "GuardError",
    "HeavyRunLock",
    "HeldoutGuard",
    "commit_charge_percent",
    "require_headroom",
]

COMMIT_LIMIT_PCT = 88.0
LOCK_FRESH_S = 2 * 3600.0  # a lock younger than this belongs to a live run


class GuardError(Exception):
    """A gate refused: no freeze entry, too little memory headroom, or a heavy-run lock is held."""


# ----------------------------------------------------------------------------- held-out seeds


class HeldoutGuard:
    """Hands out held-out seeds for one pre-registration hash, and counts every use."""

    def __init__(self, ledger: Ledger, prereg_sha: str) -> None:
        self.ledger = ledger
        self.prereg_sha = prereg_sha

    def is_frozen(self) -> bool:
        """True if the (verified) ledger holds a freeze entry for this hash."""
        return any(
            e["kind"] == "freeze" and e.get("prereg_sha256") == self.prereg_sha
            for e in self.ledger.entries()
        )

    def seeds(self, n: int, start: int = 0) -> list[int]:
        """Seeds ``start .. start + n - 1`` of this pre-registration, and record the use.

        Raises ``GuardError`` unless a freeze entry for ``prereg_sha`` exists in a ledger whose
        chain verifies; a changed pre-registration therefore has no seeds until it is frozen
        again. Every call adds ``n`` to the ledger's ``heldout_evaluations``, also when the same
        seeds are asked for twice.
        """
        return self._hand_out(heldout_seed, n, start, {})

    def search_seeds(self, n: int, start: int = 0) -> list[int]:
        """The seeds of the published search runs (0003 A14), counted as held-out uses.

        The same gate and the same count as ``seeds``; the ledger entry says ``role="search"``.
        """
        return self._hand_out(search_seed, n, start, {"role": "search"})

    def f2_seeds(self, n: int, start: int = 0) -> list[int]:
        """The seeds of the F2 arms (prereg section 3, 0003 A16), counted as held-out uses.

        The same gate and the same count as ``seeds`` and ``search_seeds``; the ledger entry says
        ``role="f2"``. Before this a full run derived them with ``f2_seed`` directly, which left
        no trace in the ledger.
        """
        return self._hand_out(f2_seed, n, start, {"role": "f2"})

    def _hand_out(
        self, derive: Callable[[str, int], int], n: int, start: int, extra: dict[str, Any]
    ) -> list[int]:
        for name, value, low in (("n", n, 1), ("start", start, 0)):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be an int of at least {low}, got {value!r}")
        try:
            frozen = self.is_frozen()
        except LedgerError as exc:
            raise GuardError(f"the ledger does not verify: {exc}") from exc
        if not frozen:
            raise GuardError(
                f"no freeze entry for pre-registration {self.prereg_sha[:16]}... in the ledger: "
                "held-out seeds do not exist before the freeze, and a changed document has new ones"
            )
        seeds = [derive(self.prereg_sha, start + k) for k in range(n)]
        self.ledger.append("heldout-use", prereg_sha256=self.prereg_sha, n=n, start=start, **extra)
        return seeds


# ----------------------------------------------------------------------------- memory headroom


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def commit_charge_percent() -> float | None:
    """Committed memory as a percentage of the commit limit (Windows), else None.

    Read with ``GlobalMemoryStatusEx``: the commit limit is ``ullTotalPageFile`` and what is left
    of it is ``ullAvailPageFile``. Returns None on other systems or if the call fails.
    """
    if sys.platform != "win32":
        return None
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    windll: Any = getattr(ctypes, "windll", None)
    if windll is None or not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    if status.ullTotalPageFile == 0:
        return None
    return 100.0 * (status.ullTotalPageFile - status.ullAvailPageFile) / status.ullTotalPageFile


def require_headroom(
    reader: Callable[[], float | None] = commit_charge_percent,
    limit: float = COMMIT_LIMIT_PCT,
    *,
    strict: bool = False,
) -> float | None:
    """Return the commit charge, or raise ``GuardError`` if it is at or above ``limit``.

    An unreadable charge (``None``) passes unless ``strict`` is set, because the rule was written
    for one Windows laptop and must not stop a run on a machine that cannot report it.
    """
    charge = reader()
    if charge is None:
        if strict:
            raise GuardError("commit charge cannot be read on this machine")
        return None
    if charge >= limit:
        raise GuardError(
            f"commit charge is {charge:.1f}%, at or above the {limit:.0f}% limit: "
            "not starting a heavy run"
        )
    return charge


# ----------------------------------------------------------------------------- heavy-run lock


class HeavyRunLock:
    """The shared heavy-run lock file, as a context manager.

    Entering writes ``owner``, ``task`` and the start time. If the file exists and is younger than
    two hours another run holds it and ``GuardError`` is raised; if it is older it is reported as
    stale in ``stale_note`` and left alone, and the run is still refused (a human removes it).
    Leaving deletes the file only if this object created it.
    """

    def __init__(
        self,
        path: Path,
        owner: str,
        task: str,
        *,
        clock: Callable[[], float] = time.time,
        fresh_s: float = LOCK_FRESH_S,
    ) -> None:
        self.path = Path(path)
        self.owner = owner
        self.task = task
        self._clock = clock
        self._fresh_s = fresh_s
        self._mine = False
        self.stale_note: str | None = None

    def __enter__(self) -> HeavyRunLock:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = self._clock() - self.path.stat().st_mtime
            if age < self._fresh_s:
                raise GuardError(
                    f"heavy-run lock {self.path} is held ({age / 60:.0f} min old): "
                    "do lighter work or wait"
                ) from None
            self.stale_note = (
                f"stale heavy-run lock {self.path} ({age / 3600:.1f} h old), left alone"
            )
            raise GuardError(self.stale_note) from None
        started = datetime.fromtimestamp(self._clock(), UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"owner={self.owner} task={self.task} start={started}\n")
        self._mine = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._mine:
            try:
                self.path.unlink()
            except OSError:
                pass
            self._mine = False
