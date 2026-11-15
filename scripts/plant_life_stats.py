"""Life statistics of the SYNTHETIC plant over many seeds (the P3a done criterion).

Prints the share of first components that fail inside the horizon with no action, the earliest
onset, and for each failure mode its share, nominal-clock duration range and shape range. The same
quantities are asserted in tests/plant/test_degradation.py::TestLifeStatistics.

    python scripts/plant_life_stats.py [seeds]
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from closed_loop.core.timegrid import DAY_Q  # noqa: E402
from closed_loop.plant.degradation import build_physical, first_life  # noqa: E402
from closed_loop.plant.params import ASSET_CLASS, ASSET_IDS, MODES_BY_CLASS, PlantParams  # noqa: E402
from closed_loop.plant.scenario import Scenario  # noqa: E402


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    params = PlantParams()
    failing = total = 0
    onsets: list[float] = []
    modes: Counter[tuple[str, str]] = Counter()
    durations: dict[str, list[float]] = {}
    shapes: dict[str, list[float]] = {}
    for seed in range(n_seeds):
        phys = build_physical(params, seed, Scenario().resolve(seed, params))
        for aid in ASSET_IDS:
            life = first_life(phys, aid)
            cum = phys.cum[aid]
            onsets.append(float(np.searchsorted(cum, life.tau_onset)) / DAY_Q)
            failing += float(np.searchsorted(cum, life.tau_fail)) / DAY_Q < params.horizon_days
            total += 1
            modes[(ASSET_CLASS[aid], life.mode)] += 1
            durations.setdefault(life.mode, []).append(life.duration)
            shapes.setdefault(life.mode, []).append(life.shape)
    print(f"SYNTHETIC plant, {n_seeds} seeds x {len(ASSET_IDS)} assets, nominal condition")
    horizon = params.horizon_days
    print(
        f"first components failing inside {horizon} days: {failing}/{total} = {failing / total:.4f}"
    )
    print(f"first onset: min {min(onsets):.2f} d, mean {np.mean(onsets):.1f} d")
    for cls, specs in MODES_BY_CLASS.items():
        class_total = sum(modes[(cls, m.mode)] for m in specs)
        for m in specs:
            print(
                f"{cls:14s} {m.mode:15s} share {modes[(cls, m.mode)] / class_total:.3f} "
                f"(weight {m.weight}); catalogue duration {m.duration_days}, shape {m.shape}"
            )
    for mode, ds in durations.items():
        lo, hi = min(shapes[mode]), max(shapes[mode])
        print(
            f"{mode:15s} drawn duration {min(ds):.1f} to {max(ds):.1f}; shape {lo:.2f} to {hi:.2f}"
        )


if __name__ == "__main__":
    main()
