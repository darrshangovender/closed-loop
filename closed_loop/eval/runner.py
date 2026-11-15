"""The runner: cells in batches, sequential stopping, a worker pool and a resumable cache (P12b).

``run_cell`` is the frozen entry of 0002 section 3.13. It runs one arm under one scenario on a
list of seeds, in batches (50 by default), and after each batch asks whether the confidence
sequence of the cell's primary metric is narrow enough to stop (prereg section 7); it never goes
past ``n_max`` and it drops no seed. A ``Runner`` keeps the worker pool and the cache between cells,
so that a whole benchmark pays the interpreter start-up and the fitted-context transfer once per
worker.

Determinism. An episode is a function of its arm, scenario, seed, tuned values and the fitted
context, none of which depends on which process runs it or when. Results are collected by seed, the
stopping rule looks at whole batches in seed order, so **the same cell gives the same episodes for
1 and for 3 workers** (``wall_s`` apart: it is a stopwatch reading, and ``fingerprint`` leaves it
out). Workers are capped at ``MAX_WORKERS`` = 3.

Failures. An episode that raises, runs past its time limit or takes its worker with it is an
episode with the worst value of each primary metric, reported as such (prereg section 7): the cell
gets a stand-in built by ``worst_episode``, ``EpisodeResult.failure`` says why, and ``CellRun``
lists them. Nothing is retried, so a rerun gives the same result as the run it resumes.

The cache. Results are kept under ``benchmarks/.cache/`` in one append-only file per cell. A cell is
identified by a hash of everything that could change one of its episodes: the arm, the scenario,
the tuned values, the specification of the fitted context and the plant parameters, the source of
every module that takes part in a simulation (the post-processing modules are left out: editing a
table does not invalidate a simulation), and the versions of Python and the numerical libraries.
Each record carries its length and a SHA-256 of its payload; a torn tail (the app stopped in the
middle of a write) is ignored and overwritten. An interrupted run therefore resumes at the first
seed it has no record of, and gives bit-identical results to an uninterrupted one.

Workers do not outlive their runner. ``Runner.close`` shuts the pool down and then **terminates**
any worker still alive (an interrupted or failed run leaves workers in the middle of an episode,
and the interpreter would wait for them at exit, up to the episode time limit); ``__exit__`` kills
at once when it is leaving on an exception. A last-resort hook reaps every live pool before
``concurrent.futures`` joins its workers at interpreter exit, and each worker watches its parent
and leaves when the parent is gone (a parent that was killed cannot reap anything).

Resource gate. Before every batch the commit charge is read (``audit.guard``); at or above the limit
(88% by default) the runner raises ``CommitChargeStop`` **at that batch boundary**, after the
previous batch is on disk, and a later run resumes from there.

Trust. The cache holds pickles written by this program on this machine; it is not a place to put
files from anywhere else.
"""

from __future__ import annotations

import ctypes
import dataclasses
import functools
import hashlib
import json
import os
import pickle
import platform
import struct
import sys
import threading
import time
import traceback
import weakref
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol

from closed_loop.audit.guard import COMMIT_LIMIT_PCT, GuardError, commit_charge_percent
from closed_loop.audit.prereg import normalised_bytes
from closed_loop.core.types import Fidelity
from closed_loop.eval.arms import Arm
from closed_loop.eval.context import PreparedContext, install_context, use_context
from closed_loop.eval.counts import ALPHA_PRIMARY, StopRule, stop_rule_for
from closed_loop.eval.episode import EpisodeResult, run_episode
from closed_loop.eval.metrics import COST_CAP, FAILURES_CAP, EpisodeCoverage, aggregate
from closed_loop.eval.paths import scrub
from closed_loop.eval.tuned import TunedValues, use_tuned
from closed_loop.eval.tuned import current as current_tuned
from closed_loop.plant.scenario import Scenario
from closed_loop.spec.properties import PROPERTY_IDS

__all__ = [
    "MAX_WORKERS",
    "CellCache",
    "CellRun",
    "CommitChargeStop",
    "Done",
    "EpisodeTask",
    "EpisodeTimeout",
    "Failed",
    "TRANSIENT_KINDS",
    "WORST_COVERAGE",
    "Runner",
    "Task",
    "arm_fingerprint",
    "effective_workers",
    "environment_fingerprint",
    "fingerprint",
    "read_cell_header",
    "run_cell",
    "simulation_code_sha256",
    "slim_result",
    "worst_episode",
]

MAX_WORKERS: Final[int] = 3  # the shared laptop has four logical CPUs (0001 section 13)
_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
# Modules that only read results: editing them does not change a simulated episode. The runner and
# the stages are not among them: the tasks that run in the workers (an episode, an open-loop replay,
# a search run) are defined there, and F2 episodes are simulations too. ``assemble.py`` is the claim
# assembly moved out of the stages (0003 A18, F3): it reads cells and writes records and defines no
# worker task, so a fix to what a verdict reads, made after the full run, keeps every cell key.
_NOT_SIMULATION: Final[tuple[str, ...]] = (
    "closed_loop/eval/assemble.py",
    "closed_loop/eval/claims.py",
    "closed_loop/eval/report.py",
    "closed_loop/eval/counts.py",
    "closed_loop/audit/",
)
# Data files that change a result as a module does: the clause cache the specifications are read
# from and the cache of note verdicts (0003 A15). Hashed with the code.
_SIMULATION_DATA: Final[tuple[str, ...]] = (
    "closed_loop/spec/clause_cache.json",
    "closed_loop/spec/note_verdicts.json",
)


class CommitChargeStop(GuardError):
    """The commit charge is at or above the limit at a batch boundary: stop, and resume later."""

    def __init__(self, charge: float, limit: float, where: str) -> None:
        super().__init__(
            f"commit charge {charge:.1f}% is at or above the {limit:.0f}% limit before {where}; "
            "the cache holds everything finished so far: run again to resume"
        )
        self.charge, self.limit, self.where = charge, limit, where


class EpisodeTimeout(BaseException):  # noqa: N818 - not an error to be caught by ``except Exception``
    """Raised inside a worker when an episode runs past its time limit."""


@dataclass(frozen=True)
class Done:
    seed: int
    value: Any


# A timeout or a dead worker says something about the machine as much as about the episode (a
# laptop at 100% CPU kills episodes nothing is wrong with): such a failure is retried once, and it
# is never written to the cache, so a resume tries the seed again (0003 A15). An episode that
# raised is deterministic and is cached as the failure it is.
TRANSIENT_KINDS: Final[frozenset[str]] = frozenset({"timeout", "worker-died"})


@dataclass(frozen=True)
class Failed:
    """An episode that did not produce a result: ``kind`` is error, timeout or worker-died."""

    seed: int
    kind: str
    detail: str


def effective_workers(requested: int) -> int:
    """The worker count actually used: at least 1, at most ``MAX_WORKERS``."""
    return max(1, min(int(requested), MAX_WORKERS))


# ----------------------------------------------------------------------------- the time limit


def _with_deadline(fn: Callable[[], Any], seconds: float | None) -> Any:
    """Run ``fn``; raise ``EpisodeTimeout`` in this thread if it takes longer than ``seconds``.

    A timer thread injects the exception with ``PyThreadState_SetAsyncExc``, which acts at the next
    bytecode boundary: enough for the Python and numpy loops of a simulation (a single long C call
    would be interrupted when it returns).
    """
    if seconds is None:
        return fn()
    tid = threading.get_ident()
    fired = threading.Event()

    def trip() -> None:
        fired.set()
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid), ctypes.py_object(EpisodeTimeout)
        )

    timer = threading.Timer(seconds, trip)
    timer.daemon = True
    timer.start()
    try:
        return fn()
    finally:
        timer.cancel()
        if fired.is_set():
            # a timeout that landed just after ``fn`` returned must not surface later
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)


# ----------------------------------------------------------------------------- fingerprints


def describe_callable(fn: Any) -> str:
    """A stable text for a proposer factory: a module-level function or a ``partial`` of one."""
    if isinstance(fn, functools.partial):
        args = [repr(a) for a in fn.args]
        kw = sorted((k, repr(v)) for k, v in (fn.keywords or {}).items())
        return f"partial({describe_callable(fn.func)}, {args}, {kw})"
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"


def arm_fingerprint(arm: Arm) -> dict[str, Any]:
    """Everything about an arm that can change an episode (the proposer by what builds it)."""
    out: dict[str, Any] = {
        f.name: getattr(arm, f.name) for f in dataclasses.fields(arm) if f.name != "proposer"
    }
    out["proposer"] = describe_callable(arm.proposer)
    return out


@functools.cache
def simulation_code_sha256(root: str | None = None) -> str:
    """One hash over the source of every module that takes part in a simulated episode.

    The same construction as ``audit.prereg.code_sha256`` (relative path and normalised bytes of
    each ``*.py``, sorted) over the ``closed_loop`` package, leaving out ``_NOT_SIMULATION``.
    """
    base = Path(root) if root is not None else _ROOT
    package = base / "closed_loop"
    digest = hashlib.sha256()
    names: list[tuple[str, Path]] = []
    for path in package.rglob("*.py"):
        rel = path.relative_to(base).as_posix()
        if "__pycache__" in rel or any(rel.startswith(x) or rel == x for x in _NOT_SIMULATION):
            continue
        names.append((rel, path))
    names.extend((rel, base / rel) for rel in _SIMULATION_DATA if (base / rel).is_file())
    if not names:
        raise ValueError(f"no source found under {package}")
    for rel, path in sorted(names):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(normalised_bytes(path)).digest())
    return digest.hexdigest()


def environment_fingerprint(context: PreparedContext | None) -> dict[str, Any]:
    """What besides the task decides a result: code, fitted-context specification, libraries.

    The context is identified by its specification and the plant parameters (it is a deterministic
    function of those and of the code), not by a hash of its pickle: a pickle of a set of strings
    changes with the interpreter's hash seed, and a key that changed between the interrupted run and
    the resumed one would make every resume a miss.
    """
    from importlib.metadata import version

    import numpy

    return {
        "code": simulation_code_sha256(),
        "context": None if context is None else repr(context.spec),
        "params": None if context is None else repr(context.params),
        # the specifications the monitor and the stale-specification shield are built from, with the
        # verbatim clause text in their provenance
        "spec_clause": None if context is None else repr(context.spec_clause),
        "spec_stale": None if context is None else repr(context.spec_stale),
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": version("scipy"),
        "platform": sys.platform,
    }


def fingerprint(result: Any) -> bytes:
    """A comparable digest of a result that leaves out the stopwatch (``wall_s``)."""
    if isinstance(result, EpisodeResult):
        result = dataclasses.replace(result, wall_s=0.0)
    return hashlib.sha256(pickle.dumps(result, protocol=4)).digest()


# ----------------------------------------------------------------------------- tasks


class Task(Protocol):
    """One unit of work per seed. Picklable; runs in a worker; its result is picklable."""

    def label(self) -> str: ...

    def fingerprint(self) -> dict[str, Any]:
        """JSON-able description of everything but the seed that can change a result."""
        ...

    def __call__(self, seed: int) -> Any: ...


def slim_result(r: EpisodeResult) -> EpisodeResult:
    """The result without its bulky records (labels, checkpoints, the plant's own event list).

    ``EpisodeResult.coverage``, ``contract`` and ``learner`` carry what the metrics and the gaps
    need from them, so a cell of 500 episodes is a few megabytes and not a few gigabytes.
    """
    return dataclasses.replace(r, labels=(), checkpoints=(), plant_events=())


@dataclass(frozen=True)
class EpisodeTask:
    """One arm under one scenario; the tuned values travel with the task, not in a global."""

    arm: Arm
    scenario: Scenario
    fidelity: Fidelity
    tuned: TunedValues
    slim: bool = True

    def label(self) -> str:
        return f"{self.arm.name}/{self.scenario.condition}"

    def fingerprint(self) -> dict[str, Any]:
        theta = self.scenario.theta
        return {
            "kind": "episode",
            "arm": arm_fingerprint(self.arm),
            "condition": self.scenario.condition,
            "regime": self.scenario.regime,
            "theta": None if theta is None else hashlib.sha256(theta.tobytes()).hexdigest(),
            "fidelity": self.fidelity.value,
            "tuned": dataclasses.asdict(self.tuned),
            "slim": self.slim,
        }

    def __call__(self, seed: int) -> EpisodeResult:
        with use_tuned(self.tuned):
            result = run_episode(self.arm, seed, self.scenario, self.fidelity)
        return slim_result(result) if self.slim else result


# What a failed episode (or a failed open-loop seed) contributes to coverage: nothing was covered.
WORST_COVERAGE: Final[EpisodeCoverage] = EpisodeCoverage(0.0, 0.0, 1, 0.0, 0.0, 1, 0, 1)


def worst_episode(
    arm: Arm, seed: int, condition: str, fidelity: Fidelity, failure: str
) -> EpisodeResult:
    """The stand-in for an episode that failed: the worst value of each primary metric.

    A violation of every property, no throughput, the cost cap, the failure cap, coverage zero, a
    monitor that fired on an unshifted episode and did not fire on a shifted one. Metrics that are
    counts of what happened (jobs, attributions, overrides) are left at zero: they have no worst
    value, and ``CellRun.failures`` is what says the episode did not run.
    """
    shifted = condition == "shifted"
    has_perception = arm.perception != "none"
    modes = ("opt", "pess", "inf", "proxy")
    monitors = {m: (None if shifted else 0.0) for m in modes} if has_perception else {}
    return EpisodeResult(
        arm=arm.name,
        seed=seed,
        condition=condition,
        fidelity=fidelity,
        violations=dict.fromkeys(PROPERTY_IDS, 1),
        min_robustness=dict.fromkeys(PROPERTY_IDS, -1.0),
        throughput=0.0,
        cost=COST_CAP,
        unplanned_failures=int(FAILURES_CAP),
        planned_jobs=0,
        shield_overrides=0,
        shield_admitted=0,
        monitor_fired_day=None,
        labels=(),
        wall_s=0.0,
        coverage=WORST_COVERAGE
        if has_perception
        else EpisodeCoverage(None, None, 0, None, None, 0, 0, 0),
        monitors=monitors,
        shift_day=60.0 if shifted else None,
        failure=failure,
    )


# ----------------------------------------------------------------------------- the cache

_MAGIC: Final[bytes] = b"CLCACHE1\n"
_HEAD: Final[struct.Struct] = struct.Struct(">Q")


def _record(payload: bytes) -> bytes:
    return _HEAD.pack(len(payload)) + hashlib.sha256(payload).digest() + payload


class CellCache:
    """An append-only file of ``(seed, outcome)`` records for one cell. See the module docstring."""

    def __init__(self, directory: Path, key: str, header: dict[str, Any]) -> None:
        self.path = Path(directory) / f"{key}.cell"
        self.key = key
        self._header = json.dumps(
            {"key": key, **header}, sort_keys=True, default=str, separators=(",", ":")
        ).encode("utf-8")
        self._good_end = 0

    def load(self) -> dict[int, Done | Failed]:
        """Every outcome on file; reading stops at the first torn or corrupt record."""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {}
        if not raw.startswith(_MAGIC):
            return {}
        pos = len(_MAGIC)
        records: list[bytes] = []
        while pos + _HEAD.size + 32 <= len(raw):
            (length,) = _HEAD.unpack_from(raw, pos)
            start = pos + _HEAD.size + 32
            payload = raw[start : start + length]
            if (
                len(payload) != length
                or hashlib.sha256(payload).digest() != raw[pos + _HEAD.size : start]
            ):
                break
            records.append(payload)
            pos = start + length
        self._good_end = pos
        if not records or records[0] != self._header:
            # a header that is not ours (an altered file, or a key collision): use none of it
            self._good_end = 0
            return {}
        out: dict[int, Done | Failed] = {}
        for payload in records[1:]:
            seed, kind, value = pickle.loads(payload)
            out[seed] = Done(seed, value) if kind == "ok" else Failed(seed, *value)
        return out

    def append(self, outcome: Done | Failed) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = self._good_end == 0
        if fresh:
            self.path.write_bytes(_MAGIC + _record(self._header))
            self._good_end = self.path.stat().st_size
        elif self.path.stat().st_size != self._good_end:
            with self.path.open("r+b") as handle:  # a torn tail: cut it off before appending
                handle.truncate(self._good_end)
        if isinstance(outcome, Done):
            payload = pickle.dumps((outcome.seed, "ok", outcome.value), protocol=4)
        else:
            payload = pickle.dumps(
                (outcome.seed, "failed", (outcome.kind, outcome.detail)), protocol=4
            )
        with self.path.open("ab") as handle:
            handle.write(_record(payload))
            handle.flush()
        self._good_end += _HEAD.size + 32 + len(payload)

    def sync(self) -> None:
        if self.path.exists():
            with self.path.open("ab") as handle:  # (a read-only descriptor cannot be synced)
                os.fsync(handle.fileno())


def read_cell_header(path: Path | str) -> dict[str, Any]:
    """The header record of a cell file: the key, the task's fingerprint and the environment it was
    made in. Reads the first record only and checks its digest, so a file that is not a cell, or
    whose header was altered, raises ``ValueError`` (P12g; the public form of what
    ``scripts/render_what_gets_through.py`` needed to find the search cells of a cache)."""
    with Path(path).open("rb") as handle:
        if handle.read(len(_MAGIC)) != _MAGIC:
            raise ValueError(f"{Path(path).name} is not a cell file")
        head = handle.read(_HEAD.size)
        if len(head) != _HEAD.size:
            raise ValueError(f"{Path(path).name} has no header record")
        (length,) = _HEAD.unpack(head)
        digest = handle.read(32)
        payload = handle.read(length)
    if len(payload) != length or hashlib.sha256(payload).digest() != digest:
        raise ValueError(f"{Path(path).name}: the header record is torn or altered")
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError(f"{Path(path).name}: the header is not an object")
    return header


def cell_key(task: Task, env: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    header = {"task": task.fingerprint(), "env": env}
    blob = json.dumps(header, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40], header


# ----------------------------------------------------------------------------- workers


_LIVE_POOLS: weakref.WeakSet[ProcessPoolExecutor] = weakref.WeakSet()
REAP_GRACE_S: Final[float] = 5.0  # an idle worker leaves by itself within this time
POLL_S: Final[float] = 1.0  # how often a wait for a worker looks up (so that Ctrl+C is heard)


def reap(pool: ProcessPoolExecutor, *, kill: bool = False, grace_s: float = REAP_GRACE_S) -> None:
    """Shut ``pool`` down and make sure none of its worker processes survives.

    Pending work is cancelled. With ``kill`` the workers are terminated at once (a run that is
    failing or being interrupted has workers in the middle of an episode); otherwise idle workers
    get ``grace_s`` seconds to leave by themselves, and whatever is left is terminated.
    """
    processes = list((getattr(pool, "_processes", None) or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    if not kill:
        deadline = time.monotonic() + grace_s
        for p in processes:
            p.join(timeout=max(0.0, deadline - time.monotonic()))
    for p in processes:
        if p.is_alive():
            p.terminate()
    for p in processes:
        p.join(timeout=5.0)
        if p.is_alive():  # terminate did not take
            p.kill()
            p.join(timeout=5.0)
    _LIVE_POOLS.discard(pool)


def _reap_all_pools() -> None:
    for pool in list(_LIVE_POOLS):
        try:
            reap(pool, kill=True)
        except Exception:  # at interpreter exit there is nobody to tell
            pass


# ``concurrent.futures`` joins its workers in a callback it registers at import; callbacks run in
# reverse order of registration, so this one (registered after the import) runs first and a hung
# worker cannot hold the interpreter at exit. The hook is private API; without it, close() reaps.
_register_exit_hook = getattr(threading, "_register_atexit", None)
if _register_exit_hook is not None:
    _register_exit_hook(_reap_all_pools)


def _watch_parent(parent_pid: int) -> None:
    """Exit this worker when its parent is gone (a parent that was killed cannot reap it)."""

    def wait_windows() -> None:
        windll: Any = getattr(ctypes, "windll", None)
        synchronize = 0x00100000
        handle = windll.kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:  # no such process any more
            os._exit(1)
        windll.kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
        os._exit(1)

    def wait_posix() -> None:
        while os.getppid() == parent_pid:
            time.sleep(2.0)
        os._exit(1)

    target = wait_windows if sys.platform == "win32" else wait_posix
    threading.Thread(target=target, name="parent-watch", daemon=True).start()


def _lower_priority() -> None:
    """Below-normal priority: the machine is shared and a benchmark is background work."""
    if sys.platform == "win32":
        windll: Any = getattr(ctypes, "windll", None)
        if windll is not None:
            windll.kernel32.SetPriorityClass(windll.kernel32.GetCurrentProcess(), 0x00004000)


def _init_worker(context: PreparedContext | None, parent_pid: int | None = None) -> None:
    """The pool's initializer: runs once per worker, so the fitted context crosses once."""
    _lower_priority()
    if parent_pid is not None:
        _watch_parent(parent_pid)
    if context is not None:
        install_context(context)


def _execute(task: Task, seed: int, timeout_s: float | None) -> tuple[str, Any]:
    """Run one seed. Returns ``("ok", value)`` or ``(kind, detail)``; never raises an Exception."""
    try:
        return "ok", _with_deadline(lambda: task(seed), timeout_s)
    except EpisodeTimeout:
        return "timeout", f"no result after {timeout_s:g} s"
    except Exception as exc:  # an episode that raises is a failed episode, not a crashed run
        tb = traceback.format_exc(limit=-4).strip().splitlines()
        # the type, the message and the frames, with the paths of this machine taken out (G5)
        return "error", scrub(
            f"{type(exc).__name__}: {exc} | {' / '.join(s.strip() for s in tb[-4:])}"
        )


# ----------------------------------------------------------------------------- the runner


@dataclass
class CellRun:
    """What a cell produced: the values in seed order, the failures, and why it stopped."""

    label: str
    seeds: tuple[int, ...]  # the seeds run: a prefix of those offered
    values: list[Any]  # one per seed run; a failure is represented by its stand-in
    failures: list[Failed]
    n_max: int
    stopped_by: str  # "target" | "n_max" | "seeds"
    halfwidth: float | None
    computed: int
    cached: int
    seconds: float = 0.0
    retried: int = 0  # seeds that timed out or lost a worker once and were tried again

    @property
    def results(self) -> list[Any]:
        return self.values

    @property
    def n(self) -> int:
        return len(self.seeds)


def _transient(outcome: Done | Failed) -> bool:
    return isinstance(outcome, Failed) and outcome.kind in TRANSIENT_KINDS


class Runner:
    """A worker pool, a cache and a resource gate, kept between cells."""

    def __init__(
        self,
        *,
        workers: int = MAX_WORKERS,
        context: PreparedContext | None = None,
        cache_dir: Path | None = None,
        tuned: TunedValues | None = None,
        headroom: Callable[[], float | None] = commit_charge_percent,
        limit: float = COMMIT_LIMIT_PCT,
        timeout_s: float | None = 900.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.workers = effective_workers(workers)
        self.context = context
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.tuned = tuned if tuned is not None else current_tuned()
        self._headroom = headroom
        self.limit = limit
        self.timeout_s = timeout_s
        self._log = log if log is not None else (lambda _msg: None)
        self._pool: ProcessPoolExecutor | None = None
        self._env: dict[str, Any] | None = None

    # ------------------------------------------------------------------ lifecycle
    def __enter__(self) -> Runner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close(kill=exc_type is not None)

    def close(self, *, kill: bool = False) -> None:
        """Shut the pool down and reap its workers; ``kill`` terminates them without waiting."""
        pool, self._pool = self._pool, None
        if pool is not None:
            reap(pool, kill=kill)

    def in_process(self, *, timeout_s: float | None) -> Runner:
        """A runner for work that has to stay in this process (the F2 stage imports the three
        siblings once): one worker, so ``run_seeds`` runs the task here, with the same cache, fitted
        context, tuned values, commit-charge gate and log, and its own time limit."""
        return Runner(
            workers=1,
            context=self.context,
            cache_dir=self.cache_dir,
            tuned=self.tuned,
            headroom=self._headroom,
            limit=self.limit,
            timeout_s=timeout_s,
            log=self._log,
        )

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.context, os.getpid()),
            )
            _LIVE_POOLS.add(self._pool)
        return self._pool

    # ------------------------------------------------------------------ the gate
    def check_headroom(self, where: str) -> None:
        charge = self._headroom()
        if charge is not None and charge >= self.limit:
            raise CommitChargeStop(charge, self.limit, where)

    # ------------------------------------------------------------------ one seed at a time
    def _local(self, task: Task, seed: int) -> tuple[str, Any]:
        if self.context is None:
            return _execute(task, seed, self.timeout_s)
        with use_context(self.context):
            return _execute(task, seed, self.timeout_s)

    @staticmethod
    def _outcome(seed: int, kind: str, payload: Any) -> Done | Failed:
        return Done(seed, payload) if kind == "ok" else Failed(seed, kind, str(payload))

    def _compute(self, task: Task, seeds: Sequence[int]) -> Iterator[Done | Failed]:
        """Outcomes of ``seeds`` as they finish (not in seed order).

        The waits poll (``POLL_S``): a wait without a timeout is not interrupted by Ctrl+C on
        Windows, and an interrupted run must get to reap its workers. On any exception, including
        an interrupt and the consumer's own failure, the workers are terminated.
        """
        if self.workers == 1:
            for seed in seeds:
                yield self._outcome(seed, *self._local(task, seed))
            return
        pending = list(seeds)
        pool = self._ensure_pool()
        futures = {pool.submit(_execute, task, s, self.timeout_s): s for s in pending}
        try:
            remaining = set(futures)
            while remaining:
                done, remaining = wait(remaining, timeout=POLL_S, return_when=FIRST_COMPLETED)
                for fut in done:
                    seed = futures[fut]
                    kind, payload = fut.result()
                    pending.remove(seed)
                    yield self._outcome(seed, kind, payload)
            return
        except BrokenProcessPool:
            self.close(kill=True)
        except BaseException:  # an interrupt, or the consumer failed: leave no worker behind
            self.close(kill=True)
            raise
        # A worker died, and the pool takes every unfinished seed with it. Run what is left one at
        # a time on a fresh pool: the seed that kills a worker is then known, and is the only one
        # reported failed.
        try:
            for seed in pending:
                pool = self._ensure_pool()
                future = pool.submit(_execute, task, seed, self.timeout_s)
                try:
                    while not wait([future], timeout=POLL_S)[0]:
                        pass
                    kind, payload = future.result()
                except BrokenProcessPool:
                    self.close(kill=True)
                    yield Failed(
                        seed, "worker-died", "the worker process died while running this seed"
                    )
                else:
                    yield self._outcome(seed, kind, payload)
        except BaseException:
            self.close(kill=True)
            raise

    def _attempt(
        self, task: Task, seeds: Sequence[int], retried: list[int]
    ) -> Iterator[Done | Failed]:
        """Outcomes of ``seeds``, each seed that timed out or lost its worker tried once more.

        ``retried`` collects the seeds that were. Whatever the second try gives is final (and a
        transient failure is still not cached, see ``TRANSIENT_KINDS``).
        """
        again: list[int] = []
        for out in self._compute(task, seeds):
            if _transient(out):
                again.append(out.seed)
            else:
                yield out
        if again:
            retried.extend(again)
            self._log(f"  retrying {len(again)} episodes that timed out or lost a worker")
            yield from self._compute(task, again)

    # ------------------------------------------------------------------ cells
    def run_seeds(
        self,
        task: Task,
        seeds: Sequence[int],
        *,
        batch: int,
        n_max: int,
        stop: Callable[[list[Any]], tuple[bool, float | None]] | None = None,
        standin: Callable[[Failed], Any] | None = None,
        use_cache: bool = True,
    ) -> CellRun:
        """Run ``task`` over the first ``n_max`` of ``seeds`` in batches; see the module docstring.

        ``stop`` is called with the values so far (stand-ins included) after each batch and returns
        ``(stop now, the half-width it saw)``. ``standin`` turns a failure into a value; without it
        a failed seed contributes nothing to ``values`` (and is in ``failures``).
        """
        chosen = [int(s) for s in seeds][: int(n_max)]
        if len(set(chosen)) != len(chosen):
            raise ValueError("a cell's seeds must be distinct")
        if batch < 1:
            raise ValueError("batch must be at least 1")
        if self._env is None:
            self._env = environment_fingerprint(self.context)
        cache: CellCache | None = None
        known: dict[int, Done | Failed] = {}
        if use_cache and self.cache_dir is not None:
            key, header = cell_key(task, self._env)
            cache = CellCache(self.cache_dir, key, header)
            # a transient failure that an older run left behind is not an answer: try the seed again
            known = {s: o for s, o in cache.load().items() if not _transient(o)}
        label = task.label()
        started = time.perf_counter()
        try:
            return self._run_batches(
                task, chosen, batch, n_max, stop, standin, cache, known, label, started
            )
        except GuardError:  # a pause at a batch boundary: the pool is idle and can stay
            raise
        except BaseException:  # a failure or an interrupt: no worker may outlive it
            self.close(kill=True)
            raise

    def _run_batches(
        self,
        task: Task,
        chosen: list[int],
        batch: int,
        n_max: int,
        stop: Callable[[list[Any]], tuple[bool, float | None]] | None,
        standin: Callable[[Failed], Any] | None,
        cache: CellCache | None,
        known: dict[int, Done | Failed],
        label: str,
        started: float,
    ) -> CellRun:
        outcomes: list[Done | Failed] = []
        values: list[Any] = []
        halfwidth: float | None = None
        stopped_by = "seeds" if len(chosen) < int(n_max) else "n_max"
        computed = cached = 0
        retried: list[int] = []
        for start in range(0, len(chosen), batch):
            chunk = chosen[start : start + batch]
            missing = [s for s in chunk if s not in known]
            if missing:  # a batch that is all in the cache costs no memory: no reason to pause
                self.check_headroom(f"batch {start // batch + 1} of {label}")
            cached += len(chunk) - len(missing)
            for out in self._attempt(task, missing, retried):
                known[out.seed] = out
                computed += 1
                if cache is not None and not _transient(out):
                    cache.append(out)
            if cache is not None:
                cache.sync()
            for s in chunk:
                out = known[s]
                outcomes.append(out)
                if isinstance(out, Done):
                    values.append(out.value)
                elif standin is not None:
                    values.append(standin(out))
            if stop is not None:
                done, halfwidth = stop(values)
                self._log(
                    f"  {label}: {len(outcomes)}/{len(chosen)} episodes"
                    + (f", half-width {halfwidth:.4f}" if halfwidth is not None else "")
                    + f" ({computed} run, {cached} cached, {time.perf_counter() - started:.0f} s)"
                )
                if done and len(outcomes) < len(chosen):
                    stopped_by = "target"
                    break
        return CellRun(
            label=label,
            seeds=tuple(o.seed for o in outcomes),
            values=values,
            failures=[o for o in outcomes if isinstance(o, Failed)],
            n_max=int(n_max),
            stopped_by=stopped_by,
            halfwidth=halfwidth,
            computed=computed,
            cached=cached,
            seconds=time.perf_counter() - started,
            retried=len(retried),
        )

    def run_cell(
        self,
        arm: Arm,
        seeds: Sequence[int],
        scenario: Scenario,
        *,
        fidelity: Fidelity = Fidelity.F1,
        batch: int = 50,
        target_halfwidth: float | None = None,
        n_max: int = 500,
        rule: StopRule | None = None,
        alpha: float = ALPHA_PRIMARY,
    ) -> CellRun:
        """One arm, one scenario: episodes in batches until the primary metric's confidence
        sequence has half-width ``target_halfwidth`` or ``n_max`` is reached (prereg section 7).

        ``rule`` names the primary metric (default: the registered one for the arm,
        ``counts.stop_rule_for``) and its target; ``target_halfwidth``, when given, replaces the
        target of the registered rule. ``alpha`` is the level of the sequence the half-width is
        read on.
        """
        base = rule if rule is not None else stop_rule_for(arm.name)
        used = (
            rule
            if rule is not None
            else base
            if target_halfwidth is None
            else dataclasses.replace(base, target=target_halfwidth)
        )
        task = EpisodeTask(arm, scenario, fidelity, self.tuned)

        def standin(failure: Failed) -> EpisodeResult:
            return worst_episode(
                arm,
                failure.seed,
                scenario.condition,
                fidelity,
                f"{failure.kind}: {failure.detail}",
            )

        def stop(values: list[EpisodeResult]) -> tuple[bool, float | None]:
            try:
                rec = aggregate(values, used.metric, alpha)
            except ValueError:  # no usable episode yet for this metric
                return False, None
            assert rec.ci_lo is not None and rec.ci_hi is not None
            half = (rec.ci_hi - rec.ci_lo) / 2.0
            if used.fixed:
                return False, half
            if used.zero_event_runs_on and rec.extra.get("k") == 0:
                return False, half
            return half <= used.target, half

        return self.run_seeds(task, seeds, batch=batch, n_max=n_max, stop=stop, standin=standin)


def run_cell(
    arm: Arm,
    seeds: Sequence[int],
    scenario: Scenario,
    batch: int = 50,
    target_halfwidth: float = 0.05,
    n_max: int = 500,
    workers: int = MAX_WORKERS,
    *,
    runner: Runner | None = None,
    **kw: Any,
) -> list[EpisodeResult]:
    """The frozen entry (0002 section 3.13): the episodes of one cell, stopping sequentially.

    Without ``runner`` a private one is made (and its pool closed on the way out); pass a ``Runner``
    to share a pool and a cache across cells. Extra keywords go to ``Runner.run_cell``.
    """
    own = runner is None
    r = runner if runner is not None else Runner(workers=workers)
    try:
        return list(
            r.run_cell(
                arm,
                seeds,
                scenario,
                batch=batch,
                target_halfwidth=target_halfwidth,
                n_max=n_max,
                **kw,
            ).results
        )
    finally:
        if own:
            r.close()
