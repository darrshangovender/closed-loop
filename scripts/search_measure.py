"""Measurements and long runs of the falsification loop, in pieces (packet P9b).

    python scripts/search_measure.py timing  --context ctx.pkl --out timing.json
    python scripts/search_measure.py sanity  --context ctx.pkl --out sanity.json
    python scripts/search_measure.py batch   --context ctx.pkl --pieces DIR --arm target \
        --regimes in --algos cma random --seeds 3000 3001 3002 --budget 200
    python scripts/search_measure.py merge   --pieces DIR --out merged.json

Everything here uses ``search`` seeds (3000 to 3039) and development numbers: nothing is a held-out
or F2 result. ``timing`` measures one evaluation per arm; ``sanity`` is the check that the loop can
minimise the objective at all (the registry's H-C0, shield off, regime ``in``: a violation within
the budget); ``batch`` runs one search per (arm, regime, algorithm, seed) as a *piece*, each its
own JSON file, and skips a piece whose file exists, so a stopped run resumes.

A piece is run in *slices*: every evaluation is appended to a log beside the piece, and a slice
ends after ``--slice`` new evaluations (about two minutes of one core), the process pauses and the
next slice replays the log (the optimisers are deterministic, so the replay costs nothing but
their own overhead) and goes on. A slice takes the heavy-run lock, in the format its current
holder writes, and runs ``--locked-slice`` evaluations when the lock is free; when another session
holds it the slice is the small one, at below-normal priority, which is how the owner's rule asks a
run to behave. The commit charge is read before every slice and the run pauses at or above 88%. A
piece that finds a defect (0003 A5) stops the batch. ``merge`` joins pieces into the structured
list that ``docs/WHAT-GETS-THROUGH.md`` is generated from.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pickle
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.audit.guard import (  # noqa: E402
    GuardError,
    commit_charge_percent,
    require_headroom,
)
from closed_loop.eval.arms import Arm, arm  # noqa: E402
from closed_loop.eval.context import PreparedContext, default_context, install_context  # noqa: E402
from closed_loop.eval.metrics import Attribution  # noqa: E402
from closed_loop.search.falsify import (  # noqa: E402
    Evaluation,
    FalsifyResult,
    collect_findings,
    evaluate_theta,
    falsify_detailed,
    merge_documents,
    sanity_arm,
    target_arm,
    what_gets_through,
)
from closed_loop.search.shrink import nominal_theta  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCK_ENV = "CLOSED_LOOP_LOCK"  # the same variable benchmarks/run.py reads
OWNER = "mnesarchus-p9b"
DEFECT_EXIT = 7  # a shielded R_in violation labelled assumption_held: the caller must stop


class SliceLock:
    """The owner's heavy-run lock file, in the format its current holder writes.

    ``owner=<name> pid=<pid> start=<local ISO time>``, created with ``O_EXCL``; ``acquire`` says
    whether this process got it, and a lock held by anyone is left alone, however old.
    """

    def __init__(self, path: Path, owner: str) -> None:
        self.path = path
        self.owner = owner
        self.mine = False

    def acquire(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except (FileExistsError, PermissionError):
            return False
        stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(f"owner={self.owner} pid={os.getpid()} start={stamp}\n")
        self.mine = True
        return True

    def release(self) -> None:
        if not self.mine:
            return
        self.mine = False
        try:
            if self.path.read_text(encoding="utf-8").startswith(f"owner={self.owner} "):
                self.path.unlink()
        except OSError:
            pass


def lower_priority() -> None:
    if sys.platform == "win32":
        windll = getattr(ctypes, "windll", None)
        if windll is not None:
            windll.kernel32.SetPriorityClass(windll.kernel32.GetCurrentProcess(), 0x00004000)


def load_context(path: Path | None) -> PreparedContext:
    t0 = time.perf_counter()
    if path is not None and path.exists():
        with path.open("rb") as handle:
            ctx: PreparedContext = pickle.load(handle)  # our own file, written below
        install_context(ctx)
        print(f"context loaded in {time.perf_counter() - t0:.1f} s", flush=True)
        return ctx
    ctx = default_context()
    print(f"context fitted in {time.perf_counter() - t0:.0f} s", flush=True)
    if path is not None:
        with path.open("wb") as handle:
            pickle.dump(ctx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return ctx


def arm_named(name: str) -> Arm:
    if name == "target":
        return target_arm()
    if name == "sanity":
        return sanity_arm()
    return arm(name)


# ----------------------------------------------------------------------------- timing


def run_timing(args: argparse.Namespace, ctx: PreparedContext) -> dict[str, Any]:
    out: dict[str, Any] = {"seed": args.seed, "arms": {}}
    rng_seeds = range(args.random_points)
    for name in args.arms:
        a = arm_named(name)
        points = [nominal_theta("in")] + [
            np.random.default_rng(100 + i).random(32) for i in rng_seeds
        ]
        wall, cpu = [], []
        for theta in points:
            ev = evaluate_theta(a, theta, "in", args.seed, ctx)
            wall.append(ev.wall_s)
            cpu.append(ev.cpu_s)
            print(
                f"{name}: wall {ev.wall_s:.2f} s, cpu {ev.cpu_s:.2f} s, f = {ev.objective:+.4f}",
                flush=True,
            )
        out["arms"][name] = {
            "evaluations": len(points),
            "wall_s": wall,
            "cpu_s": cpu,
            "wall_median": statistics.median(wall),
            "cpu_median": statistics.median(cpu),
            "wall_mean": statistics.fmean(wall),
            "cpu_mean": statistics.fmean(cpu),
        }
    return out


# ----------------------------------------------------------------------------- pieces


def piece_path(pieces: Path, arm_name: str, regime: str, algo: str, seed: int) -> Path:
    return pieces / f"{arm_name}-{regime}-{algo}-{seed}.json"


class SliceDone(Exception):
    """The slice's evaluations are spent; the piece resumes from its log on the next slice."""


class SliceBudget:
    def __init__(self, n: int) -> None:
        self.left = n


def evaluation_to_json(ev: Evaluation) -> dict[str, Any]:
    return {
        "theta": [float(v) for v in ev.theta],
        "regime": ev.regime,
        "seed": ev.seed,
        "arm": ev.arm,
        "objective": ev.objective,
        "robustness": dict(ev.robustness),
        "violated": list(ev.violated),
        "first_events": [[a.prop, a.row, a.label, a.broken_row, a.asset] for a in ev.first_events],
        "label_counts": dict(ev.label_counts),
        "event_counts": [[prop, label, n] for (prop, label), n in sorted(ev.event_counts.items())],
        "wall_s": ev.wall_s,
        "cpu_s": ev.cpu_s,
    }


def evaluation_from_json(d: dict[str, Any]) -> Evaluation:
    return Evaluation(
        theta=np.array(d["theta"], dtype=np.float64),
        regime=d["regime"],
        seed=d["seed"],
        arm=d["arm"],
        objective=d["objective"],
        robustness=d["robustness"],
        violated=tuple(d["violated"]),
        first_events=tuple(Attribution(*e) for e in d["first_events"]),
        label_counts=d["label_counts"],
        wall_s=d["wall_s"],
        cpu_s=d["cpu_s"],
        event_counts={(prop, label): n for prop, label, n in d.get("event_counts", [])},
    )


class LoggedRunner:
    """``evaluate_theta`` that keeps its answers in a file, so that a long run can stop and resume.

    The n-th call replays the n-th line of the log (and checks that the optimiser asked for the
    same point: a mismatch means the run is not deterministic, which is a bug to stop for); a call
    past the log evaluates, appends, and counts against the slice. The replayed evaluation keeps
    the wall and CPU seconds it was measured with.
    """

    def __init__(self, path: Path, budget: SliceBudget) -> None:
        self.path = path
        self.budget = budget
        self.n = 0
        self.cache: list[Evaluation] = []
        if path.exists():
            self.cache = self._valid_prefix()

    def _valid_prefix(self) -> list[Evaluation]:
        """The evaluations of the log up to the first line that is not one.

        A run that was stopped while writing (a restart of the app, a power cut) can leave a torn
        or zeroed last line. The run is deterministic, so what follows the damage is evaluated
        again; the file is cut back to the good part first so that it can be appended to.
        """
        good: list[Evaluation] = []
        keep = 0
        raw = self.path.read_bytes()
        for line in raw.splitlines(keepends=True):
            # a record is a whole line: it ends in a newline, which a torn write does not
            if not line.endswith(b"\n"):
                break
            try:
                good.append(evaluation_from_json(json.loads(line.decode("utf-8"))))
            except (ValueError, KeyError, TypeError):
                break
            keep += len(line)
        if keep != len(raw):
            print(f"{self.path.name}: cut back to {len(good)} good lines", flush=True)
            with self.path.open("r+b") as handle:
                handle.truncate(keep)
        return good

    def __call__(
        self,
        arm: Arm,
        theta: NDArray[np.float64],
        regime: str,
        seed: int,
        context: PreparedContext | None,
    ) -> Evaluation:
        i = self.n
        self.n += 1
        if i < len(self.cache):
            if not np.array_equal(self.cache[i].theta, theta):
                raise RuntimeError(f"{self.path.name}: the replay diverged at evaluation {i + 1}")
            return self.cache[i]
        if self.budget.left <= 0:
            raise SliceDone
        ev = evaluate_theta(arm, theta, regime, seed, context)
        self.budget.left -= 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evaluation_to_json(ev)) + "\n")
        self.cache.append(ev)
        return ev


def run_piece(
    args: argparse.Namespace,
    ctx: PreparedContext,
    regime: str,
    algo: str,
    seed: int,
    budget: SliceBudget,
) -> dict[str, Any] | None:
    """One search and the shrinking of its find; ``None`` when the slice ran out before it ended."""
    a = arm_named(args.arm)
    stem = str(piece_path(args.pieces, args.arm, regime, algo, seed).with_suffix(""))
    try:
        res: FalsifyResult = falsify_detailed(
            lambda: a,
            regime,
            algo,
            args.budget,
            seed,
            context=ctx,
            stop_on_first=args.stop_on_first,
            runner=LoggedRunner(Path(stem + ".search.jsonl"), budget),
        )
        findings = []
        if res.violations and not args.no_shrink:
            findings = collect_findings(
                [res],
                arm_factory=lambda: a,
                per_run=args.per_run,
                context=ctx,
                max_calls=args.shrink_calls,
                runner=LoggedRunner(Path(stem + ".shrink.jsonl"), budget),
            )
    except SliceDone:
        return None
    wall, cpu = res.seconds, res.cpu_seconds
    nominal = None
    if args.nominal_check:
        nev = evaluate_theta(a, nominal_theta(regime), regime, seed, ctx)
        nominal = {"objective": nev.objective, "violated": list(nev.violated)}
    doc = what_gets_through(findings, [res])
    return {
        "piece": {
            "arm": args.arm,
            "shielded": a.shield,
            "regime": regime,
            "algo": algo,
            "seed": seed,
            "budget": args.budget,
        },
        "run": {
            "evals": res.run.evals,
            "first_violation_eval": res.run.first_violation_eval,
            "best_objective": res.run.best_f,
            "theta_best": [float(v) for v in res.run.theta_best],
            "history": [float(v) for v in res.run.history],
        },
        "timing": {
            "wall_s": wall,
            "cpu_s": cpu,
            "wall_s_per_eval": wall / max(res.run.evals, 1),
            "cpu_s_per_eval": cpu / max(res.run.evals, 1),
        },
        "nominal": nominal,
        "violations": [
            {
                "evaluation": e.index,
                "objective": e.objective,
                "violated": list(e.violated),
                "labels": dict(e.label_counts),
                "events": [[prop, label, n] for (prop, label), n in sorted(e.event_counts.items())],
                "first_events": [
                    {
                        "property": x.prop,
                        "row": x.row,
                        "label": x.label,
                        "broken_row": x.broken_row,
                        "asset": x.asset,
                    }
                    for x in e.first_events
                ],
            }
            for e in res.violations[:50]
        ],
        "document": doc,
    }


def has_defect(piece: dict[str, Any]) -> bool:
    """Any violation of a shielded arm in regime ``in``: the batch stops and the owner is told.

    0003 A5 calls only a violation labelled ``assumption_held`` a defect; one labelled with a
    broken assumption is a contract failure. Either is a reason to stop and look in regime ``in``,
    where the scenario is built to break none of them, and the report says which it is.
    """
    p = piece["piece"]
    return bool(p["shielded"] and p["regime"] == "in") and bool(piece["violations"])


def wait_for_room(limit: float) -> None:
    while True:
        try:
            require_headroom(limit=limit)
            return
        except GuardError as exc:
            print(f"paused: {exc}", flush=True)
            time.sleep(60)


def run_batch(args: argparse.Namespace, ctx: PreparedContext) -> bool:
    """Run the pieces; True if one of them found a defect (the batch stops there)."""
    args.pieces.mkdir(parents=True, exist_ok=True)
    plan = [
        (regime, algo, seed)
        for regime in args.regimes
        for seed in args.seeds
        for algo in args.algos
    ]
    for regime, algo, seed in plan:
        path = piece_path(args.pieces, args.arm, regime, algo, seed)
        if path.exists():
            print(f"skip {path.name}: done", flush=True)
            continue
        while True:
            wait_for_room(args.limit)
            lock = SliceLock(args.lock, OWNER) if args.lock else None
            held = lock.acquire() if lock is not None else False
            budget = SliceBudget(args.locked_slice if held else args.slice)
            started = time.perf_counter()
            try:
                piece = run_piece(args, ctx, regime, algo, seed, budget)
            finally:
                if lock is not None:
                    lock.release()
            if piece is not None:
                break
            print(
                f"{path.name}: slice of {'locked ' if held else ''}"
                f"{(args.locked_slice if held else args.slice)} evaluations done in "
                f"{time.perf_counter() - started:.0f} s; pausing {args.pause:.0f} s",
                flush=True,
            )
            time.sleep(args.pause)
        path.write_text(json.dumps(piece, indent=1), encoding="utf-8")
        run = piece["run"]
        print(
            f"{path.name}: evals {run['evals']}, first violation {run['first_violation_eval']}, "
            f"best {run['best_objective']:+.4f}, "
            f"cpu/eval {piece['timing']['cpu_s_per_eval']:.2f} s, "
            f"wall/eval {piece['timing']['wall_s_per_eval']:.2f} s",
            flush=True,
        )
        if has_defect(piece):
            print(
                "VIOLATION of the shielded target in R_in (labels in the piece); batch stopped",
                flush=True,
            )
            return True
    return False


# ----------------------------------------------------------------------------- merge


def merge(pieces: Path) -> dict[str, Any]:
    files = sorted(pieces.glob("*.json"))
    docs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    merged = merge_documents([d["document"] for d in docs])
    cells: dict[str, dict[str, Any]] = {}
    for d in docs:
        p, r = d["piece"], d["run"]
        key = f"{p['arm']} R_{p['regime']} {p['algo']}"
        cell = cells.setdefault(
            key,
            {
                "runs": 0,
                "found": 0,
                "first": [],
                "cpu_s_per_eval": [],
                "wall_s_per_eval": [],
                "budget": p["budget"],
            },
        )
        cell["runs"] += 1
        cell["found"] += r["first_violation_eval"] is not None
        cell["first"].append(r["first_violation_eval"])
        cell["cpu_s_per_eval"].append(d["timing"]["cpu_s_per_eval"])
        cell["wall_s_per_eval"].append(d["timing"]["wall_s_per_eval"])
    merged["cells"] = cells
    merged["pieces"] = [f.name for f in files]
    return merged


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["timing", "sanity", "batch", "merge"])
    ap.add_argument("--context", type=Path, help="pickle of the fitted context (made if absent)")
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--pieces", type=Path, default=ROOT / "docs" / "adr" / "progress" / "P9b-pieces"
    )
    ap.add_argument("--arm", default="target", help="target, sanity or a registry arm name")
    ap.add_argument("--arms", nargs="+", default=["sanity", "H-C1", "target"])
    ap.add_argument("--regimes", nargs="+", default=["in"], choices=["in", "out"])
    ap.add_argument("--algos", nargs="+", default=["cma", "random"], choices=["cma", "random"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3000, 3001, 3002])
    ap.add_argument("--seed", type=int, default=3000, help="timing: the plant seed")
    ap.add_argument("--random-points", type=int, default=3)
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--stop-on-first", action="store_true")
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument(
        "--nominal-check",
        action="store_true",
        help="record the objective of the nominal scenario on each piece's plant seed",
    )
    ap.add_argument("--per-run", type=int, default=1)
    ap.add_argument("--shrink-calls", type=int, default=150)
    ap.add_argument(
        "--lock",
        type=Path,
        default=Path(os.environ[LOCK_ENV]) if LOCK_ENV in os.environ else None,
        help=f"heavy-run lock file, taken for each piece (default: ${LOCK_ENV}; none if unset)",
    )
    ap.add_argument("--limit", type=float, default=88.0, help="commit charge to pause at")
    ap.add_argument("--slice", type=int, default=60, help="evaluations per slice without the lock")
    ap.add_argument("--locked-slice", type=int, default=150, help="evaluations per slice with it")
    ap.add_argument("--pause", type=float, default=30.0, help="seconds between slices")
    args = ap.parse_args()

    if args.mode == "merge":
        result = merge(args.pieces)
    else:
        lower_priority()
        charge = commit_charge_percent()
        print(f"commit charge {charge if charge is None else round(charge, 1)}%", flush=True)
        wait_for_room(args.limit)
        ctx = load_context(args.context)
        if args.mode == "timing":
            result = run_timing(args, ctx)
        elif args.mode == "sanity":
            args.arm, args.stop_on_first, args.no_shrink, args.lock = "sanity", True, True, None
            args.pieces = args.pieces.parent / "P9b-sanity-pieces"
            run_batch(args, ctx)
            result = merge(args.pieces)
        else:
            if run_batch(args, ctx):
                sys.exit(DEFECT_EXIT)
            return
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    else:
        print(json.dumps(result, indent=1)[:4000])


if __name__ == "__main__":
    main()
