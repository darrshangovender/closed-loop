"""CPU seconds of shifted B3 episodes and a digest of what they did (P5d-a; P12a finding F6).

    python scripts/f6_timing.py --context ctx.pkl --out f6.json [--tag after]

``--context`` is a pickle of a fitted ``PreparedContext`` (``eval_measure.py ... --context`` writes
one). The arm is ``B3-V3`` in the shifted condition on development seeds 1100 to 1104, three repeats
after one warm-up episode; the figure is ``time.process_time`` (a shared laptop makes wall time
unreadable). ``behaviour_sha256`` digests the violations, throughput, cost and job counts of the
first repeat, so that two runs with different code can be shown to have done the same thing.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.eval.arms import ARMS  # noqa: E402
from closed_loop.eval.context import install_context  # noqa: E402
from closed_loop.eval.episode import run_episode, scenario_for  # noqa: E402

SEEDS = list(range(1100, 1105))
REPEATS = 3


def lower_priority() -> None:
    windll = getattr(ctypes, "windll", None)
    if sys.platform == "win32" and windll is not None:
        windll.kernel32.SetPriorityClass(windll.kernel32.GetCurrentProcess(), 0x00004000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    lower_priority()
    with args.context.open("rb") as handle:
        ctx = pickle.load(handle)  # our own file, written by eval_measure.py
    install_context(ctx)
    arm = ARMS["B3-V3"]
    run_episode(arm, SEEDS[0] - 1, scenario_for("shifted"), context=ctx)  # warm-up
    per_seed: dict[int, list[float]] = {s: [] for s in SEEDS}
    digest = hashlib.sha256()
    for rep in range(REPEATS):
        for s in SEEDS:
            c0 = time.process_time()
            r = run_episode(arm, s, scenario_for("shifted"), context=ctx)
            per_seed[s].append(time.process_time() - c0)
            if rep == 0:
                what = [
                    r.seed, r.violations, r.throughput, r.cost, r.planned_jobs,
                    r.unplanned_failures, r.wrong_mode_jobs, sorted(r.plant_violations.items()),
                ]  # fmt: skip
                digest.update(json.dumps(what, sort_keys=True, default=str).encode())
    every = [t for v in per_seed.values() for t in v]
    out = {
        "tag": args.tag,
        "arm": "B3-V3",
        "condition": "shifted",
        "seeds": SEEDS,
        "repeats": REPEATS,
        "cpu_s_median_all": statistics.median(every),
        "cpu_s_min_all": min(every),
        "cpu_s_mean_of_per_seed_min": statistics.fmean(min(v) for v in per_seed.values()),
        "cpu_s_per_seed_min": {s: min(v) for s, v in per_seed.items()},
        "behaviour_sha256": digest.hexdigest(),
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
