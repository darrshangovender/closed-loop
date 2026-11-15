"""Module-level tasks for the tests of the worker pool (a spawned worker has to import them)."""

from __future__ import annotations

import ctypes
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LONG_S = 600.0  # a worker in the middle of "an episode" that will not end in this test's lifetime


@dataclass(frozen=True)
class DieOnSeed:
    """Returns the seed, except for one seed, which takes its worker process down outright."""

    seed: int

    def label(self) -> str:
        return "die-on-seed"

    def fingerprint(self) -> dict[str, Any]:
        return {"kind": "die-on-seed", "seed": self.seed}

    def __call__(self, seed: int) -> int:
        if seed == self.seed:
            os._exit(3)
        return seed


@dataclass(frozen=True)
class Echo:
    """Returns the seed."""

    def label(self) -> str:
        return "echo"

    def fingerprint(self) -> dict[str, Any]:
        return {"kind": "echo"}

    def __call__(self, seed: int) -> int:
        return seed


@dataclass(frozen=True)
class Hang:
    """Writes the pid of its worker to ``pid_dir``, then stays in "an episode" for ten minutes.

    With ``quick_seed`` set, that one seed returns at once (after at least one worker has started
    hanging), so a result reaches the parent while another worker is still busy.
    """

    pid_dir: str
    quick_seed: int | None = None

    def label(self) -> str:
        return "hang"

    def fingerprint(self) -> dict[str, Any]:
        return {"kind": "hang", "dir": self.pid_dir, "quick": self.quick_seed}

    def __call__(self, seed: int) -> int:
        if seed == self.quick_seed:
            deadline = time.time() + 120.0
            while not list(Path(self.pid_dir).glob("*.pid")) and time.time() < deadline:
                time.sleep(0.05)
            return seed
        (Path(self.pid_dir) / f"{os.getpid()}.pid").write_text("x", encoding="utf-8")
        time.sleep(LONG_S)
        return seed


def pids_in(pid_dir: Path | str) -> list[int]:
    return sorted(int(p.stem) for p in Path(pid_dir).glob("*.pid"))


def wait_for_pids(pid_dir: Path | str, n: int, timeout_s: float = 180.0) -> list[int]:
    """Block until ``n`` workers have written their pids (spawning a worker takes seconds)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        found = pids_in(pid_dir)
        if len(found) >= n:
            return found
        time.sleep(0.2)
    raise TimeoutError(f"only {len(pids_in(pid_dir))} of {n} workers started")


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def all_gone(pids: list[int], timeout_s: float = 30.0) -> bool:
    """True once no pid of ``pids`` is alive (polls, because a process takes a moment to go)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not any(pid_alive(p) for p in pids):
            return True
        time.sleep(0.2)
    return False
