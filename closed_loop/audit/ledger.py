"""Append-only, hash-chained ledger of runs (JSON lines).

Interface text: docs/adr/0002-interfaces.md, section 3.12; method: docs/adr/0001-architecture.md,
section 12, as amended by docs/adr/0003-review-amendments.md, A1 (the file is
``benchmarks/ledger.jsonl``).

Each line is one entry in canonical form (sorted keys, no spaces, ASCII only)::

    {"seq": 0, "utc": "...", "kind": "freeze", ..., "heldout_evaluations": 0,
     "prev_hash": "000...0", "hash": "<sha256 of the same entry without 'hash'>"}

``prev_hash`` is the previous entry's ``hash`` (64 zeros for the first). ``verify`` rebuilds every
line from its parsed content and demands the identical bytes, so a change to any byte of any line
fails, including a changed line ending. It cannot see a *consistent* rewrite of the whole file, or
the deletion of the last lines unless the head is supplied: keep the head hash somewhere else (the
release notes, a timestamping service). That limit is part of the design (0001, section 12).

``kind`` is one of ``ALLOWED_KINDS``. ``heldout-use`` is the entry the held-out guard writes each
time it hands out seeds; it is not in the list of 0001, section 12, and is added here so that the
use counter lives in the chain.

Appends take a lock file (``<ledger>.lock``, created with ``O_CREAT | O_EXCL``) so two processes
cannot write the same ``seq``; a lock older than ``stale_after_s`` is taken to belong to a dead
process and is removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["ALLOWED_KINDS", "GENESIS_HASH", "Ledger", "LedgerError"]

GENESIS_HASH = "0" * 64
ALLOWED_KINDS: tuple[str, ...] = ("freeze", "full", "full-failed", "heldout-use")
_RESERVED = frozenset({"seq", "utc", "kind", "heldout_evaluations", "prev_hash", "hash"})


class LedgerError(Exception):
    """The ledger is broken, locked, or an append was refused."""


def _canonical(entry: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise LedgerError(f"entry cannot be written as JSON: {exc}") from exc


def _hash_of(entry: Mapping[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()


class Ledger:
    """A hash-chained JSON-lines file. Create it by appending; a missing file is an empty ledger."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
        lock_timeout_s: float = 10.0,
        stale_after_s: float = 120.0,
    ) -> None:
        self.path = Path(path)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock_timeout_s = lock_timeout_s
        self._stale_after_s = stale_after_s

    # ------------------------------------------------------------------ reading

    def entries(self) -> list[dict[str, Any]]:
        """Every entry, after ``verify``. Raises ``LedgerError`` if the chain is broken."""
        return self._read_verified()

    def verify(self, head: str | None = None) -> None:
        """Check the whole chain; raises ``LedgerError`` on the first problem found.

        With ``head`` (the last hash, kept elsewhere) it also detects removal of the last lines.
        """
        entries = self._read_verified()
        if head is not None:
            actual = entries[-1]["hash"] if entries else GENESIS_HASH
            if actual != head:
                raise LedgerError(
                    f"ledger head is {actual[:16]}... but {head[:16]}... was expected: "
                    "lines were removed or the ledger was rewritten"
                )

    def head(self) -> str:
        """The hash of the last entry (``GENESIS_HASH`` for an empty ledger). Verifies first."""
        entries = self._read_verified()
        return entries[-1]["hash"] if entries else GENESIS_HASH

    def heldout_evaluations(self) -> int:
        """How many held-out seeds have been handed out, over the whole life of the ledger."""
        entries = self._read_verified()
        return int(entries[-1]["heldout_evaluations"]) if entries else 0

    def _read_verified(self) -> list[dict[str, Any]]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise LedgerError(f"cannot read {self.path}: {exc}") from exc
        if not raw:
            return []
        if not raw.endswith(b"\n"):
            raise LedgerError("last line is not terminated: the file was truncated or edited")
        try:
            text = raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise LedgerError(f"line contains a non-ASCII byte at offset {exc.start}") from exc
        entries: list[dict[str, Any]] = []
        prev = GENESIS_HASH
        for number, line in enumerate(text[:-1].split("\n")):
            where = f"line {number + 1}"
            try:
                entry = json.loads(line)
            except ValueError as exc:
                raise LedgerError(f"{where}: not valid JSON ({exc})") from exc
            if not isinstance(entry, dict):
                raise LedgerError(f"{where}: not a JSON object")
            if _canonical(entry) != line:
                raise LedgerError(f"{where}: bytes differ from the canonical form")
            if entry.get("seq") != number:
                raise LedgerError(f"{where}: seq is {entry.get('seq')!r}, expected {number}")
            if entry.get("prev_hash") != prev:
                raise LedgerError(f"{where}: prev_hash does not match the previous entry")
            if entry.get("hash") != _hash_of(entry):
                raise LedgerError(f"{where}: hash does not match the content")
            if entry.get("kind") not in ALLOWED_KINDS:
                raise LedgerError(f"{where}: unknown kind {entry.get('kind')!r}")
            if not isinstance(entry.get("heldout_evaluations"), int):
                raise LedgerError(f"{where}: heldout_evaluations missing")
            entries.append(entry)
            prev = entry["hash"]
        return entries

    # ------------------------------------------------------------------ writing

    def append(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Add an entry and return it as written. Verifies the chain first.

        ``fields`` may not use the names ``seq``, ``utc``, ``kind``, ``heldout_evaluations``,
        ``prev_hash`` or ``hash``. A ``heldout-use`` entry must carry an integer ``n`` (the number
        of seeds handed out), which is added to the running ``heldout_evaluations``.
        """
        if kind not in ALLOWED_KINDS:
            raise LedgerError(f"kind must be one of {ALLOWED_KINDS}, got {kind!r}")
        clash = _RESERVED & set(fields)
        if clash:
            raise LedgerError(f"reserved field names: {sorted(clash)}")
        if kind == "heldout-use" and (
            isinstance(fields.get("n"), bool) or not isinstance(fields.get("n"), int)
        ):
            raise LedgerError("a heldout-use entry needs an integer n")
        with self._locked():
            entries = self._read_verified()
            last = entries[-1] if entries else None
            used = int(last["heldout_evaluations"]) if last else 0
            if kind == "heldout-use":
                used += int(fields["n"])
            entry: dict[str, Any] = {
                **fields,
                "seq": len(entries),
                "utc": self._clock().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "kind": kind,
                "heldout_evaluations": used,
                "prev_hash": last["hash"] if last else GENESIS_HASH,
            }
            entry["hash"] = _hash_of(entry)
            line = _canonical(entry).encode("ascii") + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return json.loads(line)

    # ------------------------------------------------------------------ locking

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock = self.path.with_name(self.path.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._lock_timeout_s
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._remove_if_stale(lock):
                    continue
                if time.monotonic() >= deadline:
                    raise LedgerError(
                        f"ledger is locked by {lock} (another run, or a dead one)"
                    ) from None
                time.sleep(0.02)
                continue
            except PermissionError:  # Windows: the file is being deleted by its owner
                if time.monotonic() >= deadline:
                    raise LedgerError(f"cannot take the ledger lock {lock}") from None
                time.sleep(0.02)
                continue
            break
        try:
            os.write(fd, f"pid={os.getpid()}\n".encode("ascii"))
            os.close(fd)
            yield
        finally:
            try:
                os.remove(lock)
            except OSError:
                pass

    def _remove_if_stale(self, lock: Path) -> bool:
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            return True  # it vanished between the failed create and now: try again
        if age < self._stale_after_s:
            return False
        try:
            os.remove(lock)
        except OSError:
            return False
        return True
