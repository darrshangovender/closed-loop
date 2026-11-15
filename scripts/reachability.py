"""What can each verdict show at its registered size? Made-up data only (0003 A16, packet P12c).

Prints, with a fixed seed:

1. the floors and half-width factors of prereg section 6 (the betting sequence has a half-width set
   by the range; the spread-scaled sequence has one that follows the spread);
2. the reachability table of ``closed_loop.eval.reachability``: for each claim and leg, the label
   the registered rule gives a perfect input and a useless one, through the code the benchmark
   uses, with how far the stated spreads can grow before the label is lost;
3. the time-uniform coverage of the spread-scaled sequence, a port of the throw-away simulation
   that chose the kurtosis guard of ``stats.confseq.paired_cs``.

Nothing here is a plant result. The same numbers are written by ``benchmarks/run.py`` as records
named ``reach.*``.

    python scripts/reachability.py [--reps N] [--no-break-even]

``--reps`` is the number of simulated runs per shape (default 40,000, about a minute of CPU);
``--no-break-even`` leaves out the search for how much noisier the plant may be (about two minutes).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.eval import reachability as reach  # noqa: E402
from closed_loop.eval.counts import ALPHA_PERC, ALPHA_PRIMARY  # noqa: E402
from closed_loop.stats.confseq import (  # noqa: E402
    SPREAD_KURTOSIS_DIVISOR,
    SPREAD_MIN_PAIRS,
    SPREAD_R,
)


def print_floors() -> None:
    print("== floors: a paired difference in [-1, 1], level 0.05 / 9 ==")
    print(f"spread-scaled sequence: r = {SPREAD_R:.6f}; used from {SPREAD_MIN_PAIRS} pairs, with")
    print(
        f"sample kurtosis at most n / {SPREAD_KURTOSIS_DIVISOR:g}; the betting sequence otherwise"
    )
    print(
        f"{'n':>5s}  {'betting half-width':>19s}  {'spread factor (x s)':>20s}  at level 0.05 / 18"
    )
    for f, g in zip(reach.floors(), reach.floors(ALPHA_PERC), strict=True):
        print(f"{f.n:5d}  {f.betting:19.3f}  {f.spread_factor:20.3f}  {g.spread_factor:.3f} x s")
    print()


def print_table(break_even: bool) -> None:
    print("== reachability: made-up per-seed values at the registered counts ==")
    sp = reach.SPREADS
    print(f"spreads (per episode): {sp}")
    started = time.perf_counter()
    rows = reach.reachability_rows()
    print(
        f"({len(rows)} cases through the benchmark's code in {time.perf_counter() - started:.0f} s)"
    )
    head = f"{'claim / leg':28s} {'input':8s} {'n':>5s} {'rule':>12s} {'gives':>12s}  "
    print(head + ("spreads may grow to  " if break_even else "") + "bounds")
    for row in rows:
        k = row.case
        if break_even and k.reachable:
            k_max = reach.break_even(k)
            tail = (
                "none" if k_max is None else (f">= x{k_max:.1f}" if k_max >= 8 else f"x{k_max:.2f}")
            )
        else:
            tail = "-"
        mark = (
            ""
            if row.reached
            else ("   <- cannot reach it" if not k.reachable else "   <- NOT AS RECORDED")
        )
        label = f"{k.claim}{'/' + k.leg if k.leg else ''}"
        print(
            f"{label:28s} {k.kind:8s} {k.n:5d} {k.want:>12s} {row.label:>12s}  "
            + (f"{tail:>19s}  " if break_even else "")
            + f"{row.bound}{mark}"
        )
    print()
    for row in rows:
        if not row.case.reachable:
            print(f"{row.case.id}: {row.case.why}\n")


def print_coverage(reps: int) -> None:
    print("== coverage: time-uniform, the spread-scaled sequence at looks of 100 to 500 pairs ==")
    print(
        f"alpha {ALPHA_PRIMARY:.5f}; per guard G: miss / alpha, share of runs using the spread "
        "rule at n=300"
    )
    print(
        "(a miss of the whole rule is at most this miss + alpha; registered guard "
        f"G = {SPREAD_KURTOSIS_DIVISOR:g})"
    )
    started = time.perf_counter()
    sim = reach.coverage_sim(reps)
    by_case: dict[str, dict[int, reach.SimRow]] = {}
    for row in sim:
        by_case.setdefault(row.case, {})[row.guard] = row
    for name, rows in by_case.items():
        print(
            f"{name:22s}",
            "  ".join(
                f"G={g}: {r.miss_over_alpha:5.2f} x, used {r.used_at_300:.2f}"
                for g, r in rows.items()
            ),
        )
    print(f"({reps} runs a shape, {time.perf_counter() - started:.0f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reps", type=int, default=40000)
    parser.add_argument("--no-break-even", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    print_floors()
    print_table(not args.no_break_even)
    print_coverage(args.reps)


if __name__ == "__main__":
    main()
