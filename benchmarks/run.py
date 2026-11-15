"""Entry point for the benchmark: freeze, check, quick run and full run (packets P10 and P12b).

    python benchmarks/run.py            the full run (same as --full)
    python benchmarks/run.py --quick    a short run that writes nothing to the ledger
    python benchmarks/run.py --check    verify the ledger, the hashes and the published results
    python benchmarks/run.py --freeze   record the pre-registration in an empty ledger

The full run is the one thing that regenerates every published number: it checks that the
pre-registration is frozen and unchanged, asks the held-out guard for the seeds (which records the
use), runs the ten stages of ``closed_loop.eval.stages``, writes ``benchmarks/results.json``,
``benchmarks/RESULTS.md`` and ``docs/PROOF-SCOPE.md``, and appends a ``full`` entry to the ledger,
or a ``full-failed`` one if anything went wrong (also when the run is stopped at a batch boundary
because the commit charge is high, or interrupted: finished episodes are in the cache under
``benchmarks/.cache/full/`` and the next run resumes from them).

After the ledger entry of a full run is written and verified, the run publishes the two documents
that are not part of ``results.json``'s own outputs (P12g): ``docs/WHAT-GETS-THROUGH.md``
(``scripts/render_what_gets_through.py``, with the same worker count; this process holds the
heavy-run lock, so the script does not take it again) and the Results block of ``README.md``
(``scripts/render_readme_results.py --write``, when the README carries its markers). If either
fails the run says which and exits 1; the ledger entry stands and the script can be run by hand.
So ``python benchmarks/run.py`` regenerates every published number and nothing else is needed.

``--quick`` uses development seeds (never a held-out seed), divides every count by 25 (minimum 4),
prints ``QUICK RUN, NOT A RESULT`` and writes its files under ``benchmarks/.cache/quick-output/``,
not over the published ones. It adds no ledger entry and publishes neither document. ``--stages
a,b`` limits a quick run to the named stages.

``--check`` needs no simulation: it verifies the ledger chain, the pre-registration and code hashes
and, if results are published, that ``RESULTS.md`` and ``docs/PROOF-SCOPE.md`` are exactly what
``results.json`` generates, that the verdicts rebuilt from its records are the stored ones, and
that its digest is the one the ledger recorded for the last full run. It also reports
``docs/WHAT-GETS-THROUGH.md`` as missing when it is, and the README Results block as stale when it
differs from what ``results.json`` generates.

``--freeze`` renames ``docs/PREREGISTRATION.draft.md`` to ``docs/PREREGISTRATION.md`` (when the
draft is there and the frozen file is not), hashes it and writes the first ledger entry. One freeze
per ledger.

The commit charge is read through ``GlobalMemoryStatusEx``: a run does not start at 88% or more, and
the runner stops at the next batch boundary if it rises to that level. The heavy-run lock is taken
from the path in the environment variable ``CLOSED_LOOP_LOCK`` (or ``--lock``); with neither, no
lock is taken. At most three worker processes are used.

Exit codes: 0 done; 1 a check failed (broken ledger, pre-registration changed since the freeze,
no freeze entry for a full run, results that do not match their record, a run that raised);
2 usage; 3 refused or paused (commit charge at or above the limit, heavy-run lock held); 4 not run
(an operational error); 5 a full run finished and was recorded but is not complete (a stage whose
packet is not built).

Decisions: a change to the code since the freeze is *reported* by ``--check`` but is not a failure,
because fixing a bug after the freeze is legitimate and the ledger keeps both hashes; a change to
the pre-registration is a failure. ``--check`` on a draft that is not frozen yet exits 0 and says
so.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closed_loop.audit.guard import (  # noqa: E402
    COMMIT_LIMIT_PCT,
    GuardError,
    HeavyRunLock,
    HeldoutGuard,
    commit_charge_percent,
    require_headroom,
)
from closed_loop.audit.ledger import Ledger, LedgerError  # noqa: E402
from closed_loop.audit.prereg import code_sha256, file_sha256, prereg_sha256  # noqa: E402
from closed_loop.audit.seeds import CALIB_SEEDS, DEV_SEEDS, SEARCH_DEV_SEEDS  # noqa: E402

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_REFUSED, EXIT_NOT_RUN, EXIT_INCOMPLETE = 0, 1, 2, 3, 4, 5
QUICK_BANNER = "QUICK RUN, NOT A RESULT"
LOCK_ENV = "CLOSED_LOOP_LOCK"
WORKERS_ENV = "CLOSED_LOOP_WORKERS"
MAX_WORKERS = 3
SCRIPTS = ROOT / "scripts"  # the generators belong to the tool, not to the repository being run
README_BEGIN = "<!-- RESULTS:BEGIN -->"  # the same markers as scripts/render_readme_results.py
README_END = "<!-- RESULTS:END -->"


@dataclass(frozen=True)
class Paths:
    """Where everything lives, relative to a repository root."""

    root: Path
    prereg: Path
    ledger: Path
    tuned: Path
    siblings_lock: Path

    @property
    def draft(self) -> Path:
        return self.prereg.with_name("PREREGISTRATION.draft.md")

    @property
    def results_json(self) -> Path:
        return self.root / "benchmarks" / "results.json"

    @property
    def results_md(self) -> Path:
        return self.root / "benchmarks" / "RESULTS.md"

    @property
    def proof_scope(self) -> Path:
        return self.root / "docs" / "PROOF-SCOPE.md"

    @property
    def what_gets_through(self) -> Path:
        return self.root / "docs" / "WHAT-GETS-THROUGH.md"

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def full_cache(self) -> Path:
        """The cells of the full run: the search cells that WHAT-GETS-THROUGH is made from."""
        return self.root / "benchmarks" / ".cache" / "full"

    @property
    def wgt_cache(self) -> Path:
        return self.root / "benchmarks" / ".cache" / "wgt"

    @property
    def runs_dir(self) -> Path:
        """Each full run's ``results.json`` and run log, so that run 1 survives a rerun."""
        return self.root / "benchmarks" / "runs"

    @property
    def quick_dir(self) -> Path:
        return self.root / "benchmarks" / ".cache" / "quick-output"

    @property
    def code_roots(self) -> tuple[Path, ...]:
        # the two caches the specification and the note bank are read from change a result as
        # surely as a module does, so they are hashed with the code (0003 A15)
        spec = self.root / "closed_loop" / "spec"
        roots = (
            self.root / "closed_loop",
            self.root / "benchmarks" / "run.py",
            spec / "clause_cache.json",
            spec / "note_verdicts.json",
        )
        return tuple(r for r in roots if r.exists())


def paths_for(root: Path) -> Paths:
    return Paths(
        root=root,
        prereg=root / "docs" / "PREREGISTRATION.md",
        ledger=root / "benchmarks" / "ledger.jsonl",
        tuned=root / "benchmarks" / "tuned.json",
        siblings_lock=root / "integration" / "siblings.lock",
    )


@dataclass(frozen=True)
class RunContext:
    """What the experiments receive. ``guard`` is None for a quick run: no held-out seeds."""

    mode: str  # "quick" | "full"
    paths: Paths
    prereg_sha: str | None
    guard: HeldoutGuard | None
    workers: int = MAX_WORKERS
    stages: tuple[str, ...] | None = None
    previous_headlines: tuple[dict[str, Any], ...] = ()
    commit_reader: Callable[[], float | None] = commit_charge_percent
    limit: float = COMMIT_LIMIT_PCT
    out: TextIO = field(default=sys.stdout)


ExperimentBody = Callable[[RunContext], dict[str, Any]]


def run_experiments(ctx: RunContext) -> dict[str, Any]:
    """The benchmark: run the stages, write the three published files, return the headline.

    The headline is what the ledger records for a full run: every verdict, whether the run is
    complete, the number of seeds, and the SHA-256 of ``results.json``. A quick run writes the same
    three files under ``benchmarks/.cache/quick-output/``.
    """
    from closed_loop.eval.report import headline_of, write_outputs
    from closed_loop.eval.stages import run_benchmark

    lines: list[str] = []  # the run log: what ran and how long it took, which results.json omits

    def log(message: str) -> None:
        lines.append(message)
        print(message, file=ctx.out, flush=True)

    facts = _ledger_facts(ctx) if ctx.mode == "full" else None
    result = run_benchmark(
        mode=ctx.mode,
        root=ctx.paths.root,
        workers=ctx.workers,
        guard=ctx.guard,
        prereg_sha=ctx.prereg_sha,
        only=ctx.stages,
        previous_headlines=ctx.previous_headlines,
        ledger_facts=facts,
        headroom=ctx.commit_reader,
        limit=ctx.limit,
        log=log,
    )
    if ctx.mode == "quick":
        base = ctx.paths.quick_dir
        targets = (base / "results.json", base / "RESULTS.md", base / "PROOF-SCOPE.md")
    else:
        targets = (ctx.paths.results_json, ctx.paths.results_md, ctx.paths.proof_scope)
    digest = write_outputs(result.records, *targets)
    log(f"wrote {targets[0]}, {targets[1]} and {targets[2]}")
    if ctx.mode == "full":
        number = len(ctx.previous_headlines) + 1
        ctx.paths.runs_dir.mkdir(parents=True, exist_ok=True)
        archived = ctx.paths.runs_dir / f"results-run-{number}.json"
        while archived.exists():  # never replace the record of an earlier run
            archived = archived.with_name(archived.stem + "-again.json")
        shutil.copyfile(targets[0], archived)
        archived.with_suffix(".log").write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )
        log(f"archived {archived} and its run log")
    if not result.complete:
        log(
            "INCOMPLETE RUN: this run does not claim completeness. Not built: "
            f"{', '.join(result.pipeline.not_built) or 'none'}; see the stage table in RESULTS.md."
        )
    headline = headline_of(result.records)
    headline["results_sha256"] = digest
    headline["n_records"] = len(result.records)
    return headline


def _ledger_facts(ctx: RunContext) -> dict[str, Any]:
    """What RESULTS.md says about the ledger: this run's number, and whether the code and the tuned
    file that run are the frozen ones (the count of held-out seeds is read after they are drawn)."""
    ledger = Ledger(ctx.paths.ledger)
    freeze = _freeze_entry(ledger) or {}
    now = current_hashes(ctx.paths)
    return {
        "run_number": len(ctx.previous_headlines) + 1,
        "code_equals_frozen": now["code_sha256"] == freeze.get("code_sha256"),
        "tuned_equals_frozen": now["tuned_sha256"] == freeze.get("tuned_sha256"),
        "frozen_code_sha256": str(freeze.get("code_sha256", "")),
        "running_code_sha256": str(now["code_sha256"]),
    }


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:  # argparse exits 2 already; say it in one place
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="run.py",
        description="Closed-loop benchmark (SYNTHETIC plant). With no flag: the full run.",
        epilog="exit codes: 0 done, 1 check failed, 2 usage, 3 refused or paused, 4 not run, "
        "5 full run recorded but incomplete",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="the full run (also the default)")
    mode.add_argument(
        "--quick", action="store_true", help="short run, no ledger, no held-out seeds"
    )
    mode.add_argument("--check", action="store_true", help="verify the ledger and the results")
    mode.add_argument("--freeze", action="store_true", help="record the freeze entry")
    parser.add_argument(
        "--lock", metavar="PATH", help=f"heavy-run lock file (default: ${LOCK_ENV})"
    )
    parser.add_argument("--head", metavar="HASH", help="expected ledger head, to detect truncation")
    parser.add_argument(
        "--limit", type=float, default=COMMIT_LIMIT_PCT, help="commit-charge limit in percent"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"worker processes, at most {MAX_WORKERS} (default: ${WORKERS_ENV} or {MAX_WORKERS})",
    )
    parser.add_argument(
        "--stages", metavar="A,B", help="with --quick: run only these stages (comma-separated)"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return parser


# ----------------------------------------------------------------------------- hashes


def current_hashes(paths: Paths) -> dict[str, str | None]:
    return {
        "prereg_sha256": prereg_sha256(paths.prereg) if paths.prereg.is_file() else None,
        "code_sha256": code_sha256(paths.code_roots),
        "tuned_sha256": file_sha256(paths.tuned) if paths.tuned.is_file() else None,
        "siblings_lock_sha256": (
            file_sha256(paths.siblings_lock) if paths.siblings_lock.is_file() else None
        ),
    }


def tuned_problems(paths: Paths) -> list[str]:
    """Why ``benchmarks/tuned.json`` cannot be frozen: it must exist, validate, and name both the
    primary perception variant and the monitor mode (the choices of prereg sections 3 and 7)."""
    from closed_loop.eval.tuned import TunedError, load_tuned

    if not paths.tuned.is_file():
        return [f"{paths.tuned.name} does not exist: run scripts/tune.py on dev seeds first"]
    try:
        values = load_tuned(paths.tuned)
    except TunedError as exc:
        return [str(exc)]
    problems = []
    if not values.variant_tuned:
        problems.append(f"{paths.tuned.name} does not name primary_variant")
    if not values.mode_tuned:
        problems.append(f"{paths.tuned.name} does not name monitor_mode")
    return problems


def library_versions() -> dict[str, str]:
    """The interpreter and the numerical libraries a result was computed with (read from their
    metadata, so nothing heavy is imported)."""
    from importlib.metadata import version

    return {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scipy": version("scipy"),
    }


def _freeze_entry(ledger: Ledger) -> dict[str, Any] | None:
    for entry in ledger.entries():
        if entry["kind"] == "freeze":
            return entry
    return None


# ----------------------------------------------------------------------------- commands


def check_published(paths: Paths, ledger: Ledger, out: TextIO) -> int:
    """The published results against their generator and the ledger. Needs no simulation."""
    files = (paths.results_json, paths.results_md, paths.proof_scope)
    present = [p for p in files if p.exists()]
    fulls = [e for e in ledger.entries() if e["kind"] == "full"]
    if not present and not fulls:
        print("results: none published yet", file=out)
        return EXIT_OK
    from closed_loop.eval.report import check_outputs

    problems: list[str] = []
    if not fulls:
        problems.append("results are published but the ledger has no full run")
    for p in files:
        if not p.exists():
            problems.append(f"{p.name} is missing")
    expected = None
    if fulls:
        expected = fulls[-1].get("headline", {}).get("results_sha256")
    if paths.results_json.exists():
        problems += check_outputs(
            paths.results_json, paths.results_md, paths.proof_scope, expected_sha256=expected
        )
    if not paths.what_gets_through.exists():
        problems.append(
            f"{_shown(paths.what_gets_through, paths)} is missing; python benchmarks/run.py writes "
            "it, or scripts/render_what_gets_through.py by hand"
        )
    block = "the README has no Results block"
    if _readme_has_markers(paths.readme):
        block = "the README Results block is current"
        if paths.results_json.exists():
            problem = _readme_block_problem(paths)
            if problem:
                problems.append(problem)
    if problems:
        for line in problems:
            print(f"results: FAILED ({line})", file=out)
        return EXIT_FAILED
    print(
        f"results: OK ({len(fulls)} full runs recorded; documents match results.json; "
        f"{paths.what_gets_through.name} is there; {block})",
        file=out,
    )
    return EXIT_OK


def _shown(path: Path, paths: Paths) -> str:
    """A path as the repository names it: relative, with forward slashes."""
    try:
        return path.relative_to(paths.root).as_posix()
    except ValueError:
        return path.name


def _readme_has_markers(readme: Path) -> bool:
    """Whether the README carries (either of) the markers of the Results block."""
    if not readme.is_file():
        return False
    text = readme.read_text(encoding="utf-8", errors="replace")
    return README_BEGIN in text or README_END in text


def _readme_block_problem(paths: Paths) -> str | None:
    """Why the README block is not what ``results.json`` generates, or None if it is."""
    command = [
        sys.executable,
        str(SCRIPTS / "render_readme_results.py"),
        "--check",
        "--results",
        str(paths.results_json),
        "--readme",
        str(paths.readme),
    ]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8", timeout=300, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"the README Results block could not be checked ({type(exc).__name__}: {exc})"
    if done.returncode == 0:
        return None
    if done.returncode == 1:
        return (
            "the README Results block is stale: it differs from what results.json generates; "
            "python benchmarks/run.py rewrites it, or scripts/render_readme_results.py --write"
        )
    detail = (done.stderr or done.stdout).strip().splitlines()
    return "the README Results block could not be checked: " + (
        detail[-1] if detail else "no output"
    )


# ----------------------------------------------------------------------------- publishing

Streamer = Callable[[list[str], TextIO], int]
Publisher = Callable[[Paths, int, TextIO], int]


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Stop a child and what it started (the generators start workers of their own)."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    else:
        proc.kill()
    proc.wait()


def stream_command(command: list[str], out: TextIO) -> int:
    """Run ``command``, copying its output to ``out`` line by line so a long step shows progress."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=ROOT,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print("  | " + line.rstrip("\r\n"), file=out, flush=True)
        return proc.wait()
    except BaseException:
        _kill_tree(proc)
        raise


def publish_documents(
    paths: Paths, workers: int, out: TextIO, *, stream: Streamer | None = None
) -> int:
    """Write ``docs/WHAT-GETS-THROUGH.md`` and the README Results block from the run just made.

    Two steps, both attempted whatever the first one does: the search findings document (the
    script re-evaluates and shrinks the finds the full run cached; it does not take the heavy-run
    lock, which this process holds) and the README block (only if the README has its markers: a
    README without them is said so and left alone). Returns ``EXIT_FAILED`` if either step failed.
    """
    steps: list[tuple[str, list[str]]] = [
        (
            _shown(paths.what_gets_through, paths),
            [
                sys.executable,
                str(SCRIPTS / "render_what_gets_through.py"),
                "--results",
                str(paths.results_json),
                "--cells",
                str(paths.full_cache),
                "--cache",
                str(paths.wgt_cache),
                "--out",
                str(paths.what_gets_through),
                "--workers",
                str(workers),
            ],
        )  # fmt: skip
    ]
    if _readme_has_markers(paths.readme):
        steps.append(
            (
                f"the Results block of {_shown(paths.readme, paths)}",
                [
                    sys.executable,
                    str(SCRIPTS / "render_readme_results.py"),
                    "--write",
                    "--results",
                    str(paths.results_json),
                    "--readme",
                    str(paths.readme),
                ],
            )  # fmt: skip
        )
    else:
        print(
            f"publish: {_shown(paths.readme, paths)} has no Results block markers "
            f"({README_BEGIN} and {README_END}); its block was not written",
            file=out,
        )
    failed: list[str] = []
    run_step = stream if stream is not None else stream_command
    for label, command in steps:
        print(f"publish: writing {label}", file=out, flush=True)
        try:
            code = run_step(command, out)
        except OSError as exc:
            print(f"publish: could not start the step ({type(exc).__name__}: {exc})", file=out)
            code = -1
        if code == 0:
            print(f"publish: wrote {label}", file=out)
        else:
            failed.append(f"{label}: exit {code}")
    if failed:
        print(
            f"publish: FAILED ({'; '.join(failed)}); the ledger entry stands and the run's "
            "results are published; run the named script by hand once the cause is fixed",
            file=out,
        )
        return EXIT_FAILED
    return EXIT_OK


def cmd_check(paths: Paths, head: str | None, out: TextIO) -> int:
    ledger = Ledger(paths.ledger)
    try:
        ledger.verify(head)
    except LedgerError as exc:
        print(f"ledger: BROKEN ({exc})", file=out)
        return EXIT_FAILED
    entries = ledger.entries()
    print(f"ledger: OK ({len(entries)} entries, head {ledger.head()[:16]}...)", file=out)
    print(f"held-out seeds handed out: {ledger.heldout_evaluations()}", file=out)
    now = current_hashes(paths)
    freeze = _freeze_entry(ledger)
    if freeze is None:
        shown = (now["prereg_sha256"] or "no file")[:16]
        print(f"prereg: NOT FROZEN (no freeze entry; the document hashes to {shown})", file=out)
        return check_published(paths, ledger, out)
    status = EXIT_OK
    if now["prereg_sha256"] != freeze["prereg_sha256"]:
        print(
            f"prereg: MISMATCH (frozen {freeze['prereg_sha256'][:16]}..., now "
            f"{(now['prereg_sha256'] or 'missing')[:16]})",
            file=out,
        )
        status = EXIT_FAILED
    else:
        print(f"prereg: MATCH ({str(now['prereg_sha256'])[:16]}...)", file=out)
    same = now["code_sha256"] == freeze["code_sha256"]
    print(
        "code: UNCHANGED since the freeze"
        if same
        else f"code: CHANGED since the freeze (frozen {freeze['code_sha256'][:16]}..., "
        f"now {str(now['code_sha256'])[:16]}...)",
        file=out,
    )
    for name in ("tuned_sha256", "siblings_lock_sha256"):
        if freeze.get(name) != now[name]:
            print(f"{name.removesuffix('_sha256')}: CHANGED since the freeze", file=out)
            if name == "tuned_sha256":  # the values the arms ran on: a changed file is a failure
                status = EXIT_FAILED
    published = check_published(paths, ledger, out)
    return status if status != EXIT_OK else published


def cmd_freeze(paths: Paths, out: TextIO) -> int:
    ledger = Ledger(paths.ledger)
    try:
        existing = ledger.entries()
    except LedgerError as exc:
        print(f"ledger: BROKEN ({exc})", file=out)
        return EXIT_FAILED
    if existing:
        print(
            "refusing to freeze: the ledger already has entries (one freeze per ledger)", file=out
        )
        return EXIT_FAILED
    problems = tuned_problems(paths)
    if problems:  # before anything is renamed or written
        for line in problems:
            print(f"refusing to freeze: {line}", file=out)
        return EXIT_FAILED
    renamed = False
    if not paths.prereg.is_file() and paths.draft.is_file():
        os.replace(paths.draft, paths.prereg)  # the draft becomes the frozen document
        renamed = True
    if not paths.prereg.is_file():
        print(f"refusing to freeze: {paths.prereg} does not exist", file=out)
        return EXIT_NOT_RUN
    hashes = current_hashes(paths)
    roles = {"dev": DEV_SEEDS, "calib": CALIB_SEEDS, "search-dev": SEARCH_DEV_SEEDS}
    try:
        entry = ledger.append(
            "freeze",
            **hashes,
            seed_roles={k: [r.start, r.stop - 1] for k, r in roles.items()},
        )
    except LedgerError:
        if renamed:  # nothing was recorded, so put the draft back
            os.replace(paths.prereg, paths.draft)
        raise
    if renamed:
        print(f"renamed {paths.draft.name} to {paths.prereg.name}", file=out)
    print(f"froze pre-registration {hashes['prereg_sha256']}", file=out)
    print(f"ledger head {entry['hash']} (record it outside this repository)", file=out)
    return EXIT_OK


def _headline_sha(headline: dict[str, Any]) -> str:
    text = json.dumps(headline, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cmd_run(
    mode: str,
    paths: Paths,
    body: ExperimentBody,
    head: str | None,
    out: TextIO,
    *,
    workers: int = MAX_WORKERS,
    stages: tuple[str, ...] | None = None,
    commit_reader: Callable[[], float | None] = commit_charge_percent,
    limit: float = COMMIT_LIMIT_PCT,
    publish: Publisher = publish_documents,
) -> int:
    common: dict[str, Any] = {
        "workers": workers,
        "stages": stages,
        "commit_reader": commit_reader,
        "limit": limit,
        "out": out,
    }
    if mode == "quick":
        print(QUICK_BANNER, file=out)
        ctx = RunContext("quick", paths, None, None, **common)
        try:
            body(ctx)
        except NotImplementedError as exc:
            print(f"not run: {exc}", file=out)
            return EXIT_NOT_RUN
        except GuardError as exc:
            print(f"paused: {exc}", file=out)
            return EXIT_REFUSED
        print(QUICK_BANNER, file=out)
        return EXIT_OK

    if stages is not None:
        print("--stages is for --quick; a full run runs every stage", file=out)
        return EXIT_USAGE
    ledger = Ledger(paths.ledger)
    try:
        ledger.verify(head)
        freeze = _freeze_entry(ledger)
    except LedgerError as exc:
        print(f"refusing a full run: the ledger is broken ({exc})", file=out)
        return EXIT_FAILED
    if freeze is None:
        print("refusing a full run: no freeze entry in the ledger (run --freeze first)", file=out)
        return EXIT_FAILED
    if not paths.prereg.is_file():
        print(f"refusing a full run: {paths.prereg} does not exist", file=out)
        return EXIT_FAILED
    before = current_hashes(paths)
    if before["prereg_sha256"] != freeze["prereg_sha256"]:
        print("refusing a full run: the pre-registration differs from the freeze entry", file=out)
        return EXIT_FAILED
    if before["tuned_sha256"] != freeze.get("tuned_sha256"):
        print("refusing a full run: tuned.json differs from the freeze entry", file=out)
        return EXIT_FAILED
    prereg_sha = str(before["prereg_sha256"])
    previous = tuple(e["headline"] for e in ledger.entries() if e["kind"] == "full")
    ctx = RunContext(
        "full",
        paths,
        prereg_sha,
        HeldoutGuard(ledger, prereg_sha),
        previous_headlines=previous,
        **common,
    )
    try:
        headline = body(ctx)
    except NotImplementedError as exc:
        print(f"not run: {exc}", file=out)
        return EXIT_NOT_RUN
    except GuardError as exc:  # a pause at a batch boundary: finished episodes are cached
        ledger.append("full-failed", **before, error=f"paused: {exc}")
        print(f"full run paused and recorded: {exc}", file=out)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        ledger.append("full-failed", **before, error="interrupted")
        print("full run interrupted and recorded; run again to resume from the cache", file=out)
        raise
    except Exception as exc:  # the ledger records a failed run too
        ledger.append("full-failed", **before, error=f"{type(exc).__name__}: {exc}")
        print(f"full run failed and was recorded: {type(exc).__name__}: {exc}", file=out)
        return EXIT_FAILED
    after = current_hashes(paths)
    ledger.append(
        "full",
        **before,
        code_sha256_after=after["code_sha256"],
        prereg_sha256_after=after["prereg_sha256"],
        headline=headline,
        headline_sha256=_headline_sha(headline),
        versions=library_versions(),
    )
    try:
        ledger.verify()
    except LedgerError as exc:
        print(f"ledger: BROKEN after the run ({exc})", file=out)
        return EXIT_FAILED
    if after["prereg_sha256"] != before["prereg_sha256"]:
        print("the pre-registration changed during the run; the result is not valid", file=out)
        return EXIT_FAILED
    if after["code_sha256"] != before["code_sha256"]:
        print("the code changed during the run; both hashes are in the ledger entry", file=out)
        return EXIT_FAILED
    print(f"full run recorded; ledger head {ledger.head()}", file=out)
    published = publish(paths, workers, out)  # after the entry: a failure here leaves it standing
    if published != EXIT_OK:
        return published
    if headline.get("complete") is False:
        print("INCOMPLETE: a stage is not built; the run does not claim completeness", file=out)
        return EXIT_INCOMPLETE
    return EXIT_OK


def _workers(requested: int | None) -> int:
    if requested is None:
        env = os.environ.get(WORKERS_ENV)
        requested = int(env) if env and env.isdigit() else MAX_WORKERS
    return max(1, min(requested, MAX_WORKERS))


def main(
    argv: Sequence[str] | None = None,
    *,
    commit_reader: Callable[[], float | None] = commit_charge_percent,
    body: ExperimentBody | None = None,
    out: TextIO | None = None,
    publish: Publisher = publish_documents,
) -> int:
    stream = out if out is not None else sys.stdout
    args = build_parser().parse_args(argv)
    paths = paths_for(Path(args.root).resolve())
    try:
        if args.check:
            return cmd_check(paths, args.head, stream)
        if args.freeze:
            return cmd_freeze(paths, stream)
        mode = "quick" if args.quick else "full"
        try:
            charge = require_headroom(commit_reader, args.limit)
        except GuardError as exc:
            print(f"refused: {exc}", file=stream)
            return EXIT_REFUSED
        if charge is not None:
            print(f"commit charge {charge:.1f}% (limit {args.limit:.0f}%)", file=stream)
        lock_path = args.lock or os.environ.get(LOCK_ENV)
        experiments = body if body is not None else run_experiments
        stages = tuple(s for s in args.stages.split(",") if s) if args.stages else None
        kwargs: dict[str, Any] = {
            "workers": _workers(args.workers),
            "stages": stages,
            "commit_reader": commit_reader,
            "limit": args.limit,
            "publish": publish,
        }
        if not lock_path:
            return cmd_run(mode, paths, experiments, args.head, stream, **kwargs)
        try:
            with HeavyRunLock(Path(lock_path), "closed-loop", f"benchmarks/run.py {mode}"):
                return cmd_run(mode, paths, experiments, args.head, stream, **kwargs)
        except GuardError as exc:
            print(f"refused: {exc}", file=stream)
            return EXIT_REFUSED
    except (OSError, ValueError, LedgerError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=stream)
        return EXIT_NOT_RUN


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    raise SystemExit(main())
