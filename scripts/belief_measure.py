"""Measure the belief tracker (packet P8a): update time, and calibration against the plant.

Two measurements, both on SYNTHETIC lives from development seeds (1000-1999) only:

1. *Update time.* One episode is 360 belief updates (four assets, 90 days). The script replays the
   readings of development lives through ``BeliefTracker`` with the default 128 particles, several
   times, and prints the best and the median wall time per update and per episode, and the share of
   it spent in ``rng_for``. The target of 0001 section 3 is 1.0 s per planner episode in total, of
   which the belief is only a part; the budget this leaves is printed. The numbers depend on the
   machine and on what else is running; the CPU load at the time is printed with them.
2. *Calibration.* The world model is fitted on one block of development seeds and the tracker is run
   on another, disjoint block. At every third post-blind day on which the asset is still alive,
   the central 90% interval of the model's failure-day distribution (a stochastic rollout) is
   compared with the realised remaining life (``LifeTable.true_rul_days``, 0003 A10). The coverage
   is recorded as it comes out; nothing is tuned to it.

    python scripts/belief_measure.py [--fit-seeds 100] [--test-seeds 30] [--particles 128]
        [--repeats 7] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.core.rng import rng_for  # noqa: E402
from closed_loop.planning.belief import BeliefTracker  # noqa: E402
from closed_loop.planning.fitting import fit_world_model  # noqa: E402
from closed_loop.planning.worldmodel import PlantWorldModel  # noqa: E402
from closed_loop.plant import PlantParams, Scenario  # noqa: E402
from closed_loop.plant.observe import simulate_lives  # noqa: E402
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS  # noqa: E402

FIT_START, TEST_START = 1000, 1500


def cpu_load_percent() -> float | None:
    try:
        import ctypes

        class FileTime(ctypes.Structure):
            _fields_ = [("lo", ctypes.c_uint32), ("hi", ctypes.c_uint32)]

        def read() -> tuple[int, int, int]:
            idle, kern, user = FileTime(), FileTime(), FileTime()
            ctypes.windll.kernel32.GetSystemTimes(  # type: ignore[attr-defined]
                ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)
            )
            f = lambda t: (t.hi << 32) | t.lo  # noqa: E731
            return f(idle), f(kern), f(user)

        a = read()
        time.sleep(0.5)
        b = read()
        idle, total = b[0] - a[0], (b[1] - a[1]) + (b[2] - a[2])
        return 100.0 * (1.0 - idle / total) if total else None
    except Exception:  # not Windows, or no access: the number is only context
        return None


def models_for(fitted: dict[str, float], mismatch: float = 0.0) -> dict[str, PlantWorldModel]:
    return {
        aid: PlantWorldModel(fitted, mismatch, asset_class=ASSET_CLASS[aid]) for aid in ASSET_IDS
    }


def time_updates(fitted: dict[str, float], n_particles: int, repeats: int) -> dict[str, float]:
    params = PlantParams()
    tables = [simulate_lives(params, s, Scenario()) for s in range(FIT_START, FIT_START + 4)]
    models = models_for(fitted)
    wall: list[float] = []
    cpu: list[float] = []
    n_updates = 0
    for rep in range(repeats):
        table = tables[rep % len(tables)]
        tracker = BeliefTracker(models, n_particles=n_particles, seed=rep)
        n_updates = 0
        w0, c0 = time.perf_counter(), time.process_time()
        for aid in ASSET_IDS:
            for d in range(table.n_days):
                hi = table.hi[aid][d]
                tracker.update(
                    aid,
                    d,
                    None if np.isnan(hi) else float(hi),
                    False,
                    load=float(table.load[aid][d]),
                )
                n_updates += 1
        wall.append(time.perf_counter() - w0)
        cpu.append(time.process_time() - c0)
    t0 = time.perf_counter()
    for i in range(2000):
        rng_for(1, "belief", "PU-201", 0, i, 0)
    rng_us = (time.perf_counter() - t0) / 2000 * 1e6
    return {
        "updates_per_episode": n_updates,
        "n_particles": n_particles,
        "wall_episode_s_min": min(wall),
        "wall_episode_s_median": statistics.median(wall),
        "cpu_episode_s_min": min(cpu),
        "cpu_episode_s_median": statistics.median(cpu),
        "wall_update_us_min": 1e6 * min(wall) / n_updates,
        "wall_update_us_median": 1e6 * statistics.median(wall) / n_updates,
        "rng_for_us": rng_us,
    }


def calibration(fitted: dict[str, float], n_particles: int, n_seeds: int) -> dict[str, float | int]:
    params = PlantParams()
    models = models_for(fitted)
    inside = total = inside_deg = total_deg = 0
    for seed in range(TEST_START, TEST_START + n_seeds):
        table = simulate_lives(params, seed, Scenario())
        tracker = BeliefTracker(models, n_particles=n_particles, seed=seed)
        for aid in ASSET_IDS:
            for d in range(table.n_days):
                hi = table.hi[aid][d]
                tracker.update(
                    aid,
                    d,
                    None if np.isnan(hi) else float(hi),
                    False,
                    load=float(table.load[aid][d]),
                )
                rul = float(table.true_rul_days[aid][d])
                if d >= 23 and rul > 0.0 and d % 3 == 0:
                    lo, _, up = tracker.failure_day_quantiles(aid)
                    hit = lo <= rul <= up
                    total += 1
                    inside += int(hit)
                    if not np.isnan(hi) and hi >= params.hi_alert:
                        total_deg += 1
                        inside_deg += int(hit)
    return {
        "n_particles": n_particles,
        "test_seeds": n_seeds,
        "checkpoints_all": total,
        "coverage_all": inside / total,
        "checkpoints_degrading": total_deg,
        "coverage_degrading": inside_deg / total_deg if total_deg else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit-seeds", type=int, default=100)
    ap.add_argument("--test-seeds", type=int, default=30)
    ap.add_argument("--particles", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    load = cpu_load_percent()
    fitted = fit_world_model(range(FIT_START, FIT_START + args.fit_seeds))
    result: dict[str, object] = {
        "cpu_load_percent_when_started": load,
        "fit_seeds": f"{FIT_START}..{FIT_START + args.fit_seeds - 1} (dev)",
        "test_seeds": f"{TEST_START}..{TEST_START + args.test_seeds - 1} (dev)",
    }
    timing = time_updates(fitted, args.particles, args.repeats)
    result["timing"] = timing
    budget_s = 1.0
    result["share_of_1s_planner_episode_best"] = timing["wall_episode_s_min"] / budget_s
    result["calibration"] = calibration(fitted, args.particles, args.test_seeds)
    text = json.dumps(result, indent=2, default=float)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
