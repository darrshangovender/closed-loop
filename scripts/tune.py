"""Tuning on ``dev`` seeds: the numbers the pre-registration says are fixed before the freeze.

    python scripts/tune.py [--episodes 40] [--workers 3] [--out benchmarks/tuned.json]

Six choices, all made on development seeds (1000-1999; the default starts at 1100 because the
planner's world model is fitted on 1000-1099) and written to ``benchmarks/tuned.json``, which the
ledger hashes (prereg section 10, item 1):

* the **primary perception variant** of the planner arms, by the rule of prereg section 7: among
  V2, V3, V4, V5, V2L and V3L, the highest mean oracle lower-bound coverage ``c_e^lo`` under
  ``shifted``, ties broken by the narrower mean width of the 90% interval in health bins; measured
  on the belief planner P0 with ``N = 256`` particles;
* the **monitor mode** of arm P2 and of the C-MON verdict, by the rule of prereg section 3: of
  ``opt``, ``pess``, ``inf`` and ``proxy``, the one with the highest detection rate minus
  false-alarm rate on ``dev``, ties to the lower false-alarm rate (then to the order listed). The
  rates are those of the planner with the shield and passive monitors (P1, the C-MON arm, with the
  chosen variant and ``delta_plan``): false alarms on nominal episodes, a firing within 30 days of
  the shift on shifted ones;
* the bound rule's **k** (B3, days), the **B2 limit**, the **B1 fraction** of the characteristic
  life, and the planner's **delta_plan**: for each, the value on a small stated grid with the
  minimum mean M-COST over ``dev`` nominal episodes. Ties go to the first value of the grid.

The order is fixed: the variant first (with the provisional ``delta_plan``), then ``k`` with B3
under that variant, then the B2 limit, the B1 fraction, ``delta_plan`` with P0 under that variant,
and last the monitor mode (P1 under that variant and ``delta_plan``).
Every grid point is a table row; the tables are printed and saved next to the output as
``<out>.tables.txt``. Nothing is tuned on a held-out, search or calibration seed, and the script
refuses seeds outside ``dev``.

The grids are command-line arguments. All but ``k`` were chosen here before looking at any result:

* ``k``: 0.25, 0.5, 1, 2, 3, 5, 7, 10, 14 days (first 3 to 14; extended twice because the best
  value lay on the low edge, which can only help the baseline, prereg section 11);
  B2 limit: 4.0, 5.0, 5.5, 6.0, 7.0; B1 fraction: 0.3, 0.5, 0.7, 0.9;
  ``delta_plan``: 0.005, 0.01, 0.02, 0.05, 0.1.

It runs through the same runner as the benchmark (a cache under ``benchmarks/.cache/tune/``, so an
interrupted tuning resumes), refuses to start at 88% commit charge, and takes the heavy-run lock
named by ``--lock`` or ``$CLOSED_LOOP_LOCK`` if there is one. It refuses to overwrite an existing
output without ``--force``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from closed_loop.audit.guard import (  # noqa: E402
    COMMIT_LIMIT_PCT,
    GuardError,
    HeavyRunLock,
    commit_charge_percent,
    require_headroom,
)
from closed_loop.audit.seeds import DEV_SEEDS  # noqa: E402

SCRIPT = "scripts/tune.py"
FIRST_SEED = 1100
CANDIDATES = ("V2", "V3", "V4", "V5", "V2L", "V3L")
K_GRID = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 14.0)
B2_GRID = (4.0, 5.0, 5.5, 6.0, 7.0)
B1_GRID = (0.3, 0.5, 0.7, 0.9)
DELTA_GRID = (0.005, 0.01, 0.02, 0.05, 0.1)
MODES = ("opt", "pess", "inf", "proxy")


# ----------------------------------------------------------------------------- the two rules


@dataclass(frozen=True)
class GridRow:
    value: float
    mean_cost: float  # mean M-COST (cost capped at 200, divided by 200)
    n: int
    ci_lo: float
    ci_hi: float
    mean_failures: float
    mean_jobs: float


@dataclass(frozen=True)
class VariantRow:
    variant: str
    cov_lo: float  # mean oracle lower-bound coverage c_e^lo under shifted
    width_bins: float  # mean width of the 90% interval, in health bins
    n: int


@dataclass(frozen=True)
class ModeRow:
    mode: str
    false_alarms: int  # nominal episodes in which the mode fired
    n_nominal: int
    detections: int  # shifted episodes with a firing within 30 days of the shift
    n_shifted: int

    @property
    def false_alarm(self) -> Fraction:
        return Fraction(self.false_alarms, self.n_nominal)

    @property
    def detection(self) -> Fraction:
        return Fraction(self.detections, self.n_shifted)

    @property
    def score(self) -> Fraction:
        """Detection rate minus false-alarm rate, exact (a tie is a tie, with no tolerance)."""
        return self.detection - self.false_alarm


def choose_mode(rows: Sequence[ModeRow]) -> ModeRow:
    """Prereg section 3: the highest detection rate minus false-alarm rate on ``dev``; ties to the
    lower false-alarm rate, then to the order of ``MODES``."""
    if not rows:
        raise ValueError("no monitor mode")
    return min(rows, key=lambda r: (-r.score, r.false_alarm, MODES.index(r.mode)))


def choose_minimum(rows: Sequence[GridRow]) -> GridRow:
    """The row with the minimum mean M-COST; a tie goes to the first row (the earlier value)."""
    if not rows:
        raise ValueError("an empty grid")
    return min(rows, key=lambda r: (r.mean_cost, rows.index(r)))


def choose_variant(rows: Sequence[VariantRow]) -> VariantRow:
    """Prereg section 7: the highest ``c_e^lo`` under ``shifted``; ties broken by the narrower mean
    width. A tie is exact equality (no tolerance is registered)."""
    if not rows:
        raise ValueError("no candidate variant")
    return min(rows, key=lambda r: (-r.cov_lo, r.width_bins, CANDIDATES.index(r.variant)))


# ----------------------------------------------------------------------------- measuring


def mean_width_bins(results: Sequence[Any]) -> float:
    """Mean width, in health bins, of the 90% scheduling interval over post-blind checkpoints."""
    from closed_loop.core.types import bins_of_interval
    from closed_loop.perception.coverage import select

    widths: list[int] = []
    for r in results:
        for cp in select(r.checkpoints, "post_blind"):
            lo, hi = bins_of_interval(cp.lo, cp.hi)
            widths.append(max(0, min(3, hi)) - max(0, min(3, lo)) + 1)
    return sum(widths) / len(widths) if widths else math.nan


def _row(value: float, results: Sequence[Any]) -> GridRow:
    from closed_loop.eval.metrics import aggregate

    cost = aggregate(results, "m_cost", 0.05)
    assert cost.ci_lo is not None and cost.ci_hi is not None
    return GridRow(
        value=value,
        mean_cost=cost.value,
        n=cost.n,
        ci_lo=cost.ci_lo,
        ci_hi=cost.ci_hi,
        mean_failures=aggregate(results, "u_e", 0.05).value,
        mean_jobs=sum(r.planned_jobs for r in results) / len(results),
    )


def _no_failures(failure: Any) -> Any:
    raise RuntimeError(f"a tuning episode failed, and tuning cannot go on without it: {failure}")


def run_task(
    runner: Any, arm: Any, condition: str, tuned: Any, seeds: Sequence[int], slim: bool = True
) -> list[Any]:
    """The episodes of ``arm`` under ``condition`` on ``seeds`` (all of them: tuning has no
    stopping rule), with the tuned values of ``tuned`` in force."""
    from closed_loop.core.types import Fidelity
    from closed_loop.eval.episode import scenario_for
    from closed_loop.eval.runner import EpisodeTask

    task = EpisodeTask(arm, scenario_for(condition), Fidelity.F1, tuned, slim=slim)
    run = runner.run_seeds(
        task,
        seeds,
        batch=max(1, min(10, len(seeds))),
        n_max=len(seeds),
        standin=_no_failures,
    )
    return list(run.values)


def grid_search(
    runner: Any,
    arm: Any,
    field: str,
    grid: Sequence[float],
    tuned: Any,
    seeds: Sequence[int],
    log: Callable[[str], None],
) -> list[GridRow]:
    rows: list[GridRow] = []
    for value in grid:
        t = dataclasses.replace(tuned, **{field: value})
        results = run_task(runner, arm, "nominal", t, seeds)
        rows.append(_row(value, results))
        log(f"  {field} = {value:g}: mean M-COST {rows[-1].mean_cost:.4f} over {rows[-1].n}")
    return rows


def select_variant(
    runner: Any,
    tuned: Any,
    seeds: Sequence[int],
    candidates: Sequence[str],
    log: Callable[[str], None],
) -> list[VariantRow]:
    from closed_loop.eval.arms import ARMS
    from closed_loop.eval.metrics import aggregate

    rows: list[VariantRow] = []
    for v in candidates:
        arm = dataclasses.replace(ARMS["P0"], name=f"P0-{v}", perception=v)
        results = run_task(runner, arm, "shifted", tuned, seeds, slim=False)
        cov = aggregate(results, "m_cov_lo", 0.05)
        rows.append(VariantRow(v, cov.value, mean_width_bins(results), cov.n))
        log(f"  {v}: c_e^lo {rows[-1].cov_lo:.4f}, interval width {rows[-1].width_bins:.3f} bins")
    return rows


def select_mode(
    runner: Any, tuned: Any, seeds: Sequence[int], log: Callable[[str], None]
) -> list[ModeRow]:
    """The four monitor modes on the C-MON arm: nominal episodes for false alarms, shifted for
    detection (the metrics of ``eval.metrics``, so the choice and the verdict read the same)."""
    from closed_loop.eval.arms import ARMS
    from closed_loop.eval.metrics import METRICS

    arm = dataclasses.replace(ARMS["P1"], perception=tuned.primary_variant)
    nominal = run_task(runner, arm, "nominal", tuned, seeds)
    shifted = run_task(runner, arm, "shifted", tuned, seeds)
    rows: list[ModeRow] = []
    for mode in MODES:
        fa = [METRICS[f"m_false_alarm_{mode}"].value(r) for r in nominal]
        det = [METRICS[f"m_detect_{mode}"].value(r) for r in shifted]
        fa_known = [v for v in fa if v is not None]
        det_known = [v for v in det if v is not None]
        rows.append(
            ModeRow(
                mode,
                int(round(sum(fa_known))),
                len(fa_known),
                int(round(sum(det_known))),
                len(det_known),
            )
        )
        r = rows[-1]
        log(
            f"  {mode}: false alarms {r.false_alarms} of {r.n_nominal}, detected "
            f"{r.detections} of {r.n_shifted}"
        )
    return rows


# ----------------------------------------------------------------------------- output


def fmt_grid(title: str, field: str, rows: Sequence[GridRow], best: GridRow) -> list[str]:
    out = [
        title,
        f"  {field:>12}  {'mean M-COST':>12}  {'interval (0.05)':>20}  {'u_e':>6}  {'jobs':>6}  n",
    ]
    for r in rows:
        mark = "  <- chosen" if r is best else ""
        out.append(
            f"  {r.value:>12g}  {r.mean_cost:>12.4f}  [{r.ci_lo:.4f}, {r.ci_hi:.4f}]  "
            f"{r.mean_failures:>6.3f}  {r.mean_jobs:>6.2f}  {r.n}{mark}"
        )
    return [*out, ""]


def fmt_variants(rows: Sequence[VariantRow], best: VariantRow) -> list[str]:
    out = [
        "Primary perception variant (P0, shifted, N = 256): highest mean c_e^lo, ties by width",
        f"  {'variant':>8}  {'c_e^lo':>8}  {'width (bins)':>12}  n",
    ]
    for r in rows:
        mark = "  <- chosen" if r is best else ""
        out.append(f"  {r.variant:>8}  {r.cov_lo:>8.4f}  {r.width_bins:>12.3f}  {r.n}{mark}")
    return [*out, ""]


def fmt_modes(rows: Sequence[ModeRow], best: ModeRow) -> list[str]:
    out = [
        "Monitor mode (P1, dev): highest detection minus false alarm, ties to lower false alarm",
        f"  {'mode':>8}  {'false alarm':>12}  {'detection':>10}  {'score':>8}",
    ]
    for r in rows:
        mark = "  <- chosen" if r is best else ""
        out.append(
            f"  {r.mode:>8}  {r.false_alarms:>4} of {r.n_nominal:<4}  "
            f"{r.detections:>3} of {r.n_shifted:<3}  {float(r.score):>8.4f}{mark}"
        )
    return [*out, ""]


def write_tuned(path: Path, doc: dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def check_dev(seeds: Sequence[int]) -> None:
    bad = [s for s in seeds if s not in DEV_SEEDS]
    if bad:
        raise ValueError(f"tuning uses dev seeds (1000-1999) only, got {bad[:3]}")


# ----------------------------------------------------------------------------- main


def parse(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    ap.add_argument("--episodes", type=int, default=40, help="dev episodes per grid point")
    ap.add_argument("--first-seed", type=int, default=FIRST_SEED)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "tuned.json")
    ap.add_argument("--force", action="store_true", help="replace an existing output")
    ap.add_argument("--cache", type=Path, default=ROOT / "benchmarks" / ".cache" / "tune")
    ap.add_argument("--lock", metavar="PATH")
    ap.add_argument("--limit", type=float, default=COMMIT_LIMIT_PCT)
    ap.add_argument("--candidates", default=",".join(CANDIDATES))
    ap.add_argument("--k-grid", default=",".join(map(str, K_GRID)))
    ap.add_argument("--b2-grid", default=",".join(map(str, B2_GRID)))
    ap.add_argument("--b1-grid", default=",".join(map(str, B1_GRID)))
    ap.add_argument("--delta-grid", default=",".join(map(str, DELTA_GRID)))
    return ap.parse_args(argv)


def floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x)


def tune(args: argparse.Namespace, log: Callable[[str], None] = print) -> dict[str, Any]:
    from closed_loop.eval.context import prepare_context
    from closed_loop.eval.runner import Runner
    from closed_loop.eval.tuned import current

    seeds = tuple(range(args.first_seed, args.first_seed + args.episodes))
    check_dev(seeds)
    if args.out.exists() and not args.force:
        raise FileExistsError(f"{args.out} exists; pass --force to replace it")
    candidates = tuple(c for c in args.candidates.split(",") if c)
    unknown = [c for c in candidates if c not in CANDIDATES]
    if unknown:
        raise ValueError(f"candidates must be among {CANDIDATES}, got {unknown}")
    from closed_loop.eval.arms import ARMS

    tuned = current()
    ctx = prepare_context()
    tables: list[str] = []
    with Runner(
        workers=args.workers,
        context=ctx,
        cache_dir=args.cache,
        tuned=tuned,
        limit=args.limit,
        log=lambda m: None,
    ) as runner:
        log(f"primary variant ({len(candidates)} candidates, {len(seeds)} shifted episodes each)")
        vrows = select_variant(runner, tuned, seeds, candidates, log)
        vbest = choose_variant(vrows)
        tables += fmt_variants(vrows, vbest)
        tuned = dataclasses.replace(tuned, primary_variant=vbest.variant)
        chosen: dict[str, float] = {}
        for title, field, arm_name, grid in (
            (
                f"B3 k (days), B3 with {vbest.variant}",
                "k_b3",
                f"B3-{vbest.variant}",
                floats(args.k_grid),
            ),
            ("B2 limit on the health index", "b2_limit", "B2", floats(args.b2_grid)),
            (
                "B1 interval, fraction of the characteristic life",
                "b1_fraction",
                "B1",
                floats(args.b1_grid),
            ),
        ):
            log(title)
            rows = grid_search(runner, ARMS[arm_name], field, grid, tuned, seeds, log)
            best = choose_minimum(rows)
            tables += fmt_grid(
                f"{title}: minimum mean M-COST over dev nominal episodes", field, rows, best
            )
            chosen[field] = best.value
            tuned = dataclasses.replace(tuned, **{field: best.value})
        log(f"planner delta_plan, P0 with {vbest.variant} and N = 256")
        p0 = dataclasses.replace(ARMS["P0"], perception=vbest.variant)
        rows = grid_search(runner, p0, "delta_plan", floats(args.delta_grid), tuned, seeds, log)
        best = choose_minimum(rows)
        tables += fmt_grid(
            "Planner delta_plan (P0, N = 256): minimum mean M-COST", "delta_plan", rows, best
        )
        chosen["delta_plan"] = best.value
        tuned = dataclasses.replace(tuned, delta_plan=best.value)
        log(f"monitor mode, P1 with {vbest.variant} and delta_plan {best.value:g}")
        mrows = select_mode(runner, tuned, seeds, log)
        mbest = choose_mode(mrows)
        tables += fmt_modes(mrows, mbest)
    doc: dict[str, Any] = {
        "k_b3": chosen["k_b3"],
        "b2_limit": chosen["b2_limit"],
        "b1_fraction": chosen["b1_fraction"],
        "delta_plan": chosen["delta_plan"],
        "primary_variant": vbest.variant,
        "monitor_mode": mbest.mode,
        "tuned_on": (
            f"dev seeds {seeds[0]}-{seeds[-1]}, {len(seeds)} episodes per grid point (nominal; "
            f"shifted for the variant), planner N = 256; grids k {args.k_grid}, B2 {args.b2_grid}, "
            f"B1 {args.b1_grid}, delta_plan {args.delta_grid}; candidates {args.candidates}; "
            "monitor mode on P1, nominal for false alarms and shifted for detection"
        ),
        "script": SCRIPT,
    }
    write_tuned(args.out, doc, force=args.force)
    text = "\n".join(tables)
    args.out.with_suffix(".tables.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
    log("\n" + text)
    log(
        f"wrote {args.out}: "
        + ", ".join(f"{k} = {v}" for k, v in doc.items() if k not in ("tuned_on", "script"))
    )
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    args = parse(argv)
    try:
        require_headroom(commit_charge_percent, args.limit)
        lock = args.lock or os.environ.get("CLOSED_LOOP_LOCK")
        if lock:
            with HeavyRunLock(Path(lock), "closed-loop", "scripts/tune.py"):
                tune(args)
        else:
            tune(args)
    except GuardError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 3
    except (FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
