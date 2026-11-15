"""Wall time of one SYNTHETIC F1 plant episode (90 days, 8,640 quarters, four assets).

Three cases: the bare plant, the plant with the job runner carrying four jobs, and the slow path
(a command every quarter, so that the quiet-quarter fast paths never apply). The numbers go into
docs/adr/progress/P3b-des-plant-step.md by hand because they depend on the machine; run it on a
quiet one. The target in docs/adr/0001-architecture.md section 3 is about 0.3 s without the planner.

    python scripts/time_plant_episode.py [episodes]
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from closed_loop.core.types import Cmd, Command  # noqa: E402
from closed_loop.plant import PlantParams, Scenario, make_plant  # noqa: E402
from closed_loop.plant.jobs import JobRunner, work_order_command  # noqa: E402

CR, CV, P1, P2 = "CR-501", "CV-101", "PU-201", "PU-202"


def bare(seed: int) -> float:
    t0 = time.perf_counter()
    plant = make_plant(PlantParams())
    plant.reset(seed, Scenario())
    while not plant.step(()).done:
        pass
    return time.perf_counter() - t0


def slow(seed: int) -> float:
    t0 = time.perf_counter()
    plant = make_plant(PlantParams())
    plant.reset(seed, Scenario())
    noop = [Command(CV, Cmd.NOOP)]
    while not plant.step(noop).done:
        pass
    return time.perf_counter() - t0


def with_runner(seed: int) -> float:
    t0 = time.perf_counter()
    plant = make_plant(PlantParams())
    obs = plant.reset(seed, Scenario())
    runner = JobRunner()
    plan = {200: P1, 1200: CV, 2400: P2, 3600: CR}
    while True:
        cmds = []
        if plant.quarter in plan:
            aid = plan[plant.quarter]
            cmds.append(work_order_command(aid, plant.truth.life_mode[aid]))
        for view in obs.assets.values():
            if view.job is not None:
                cmd = runner.next_command(view, view.job, quarter=obs.quarter)
                if cmd is not None:
                    cmds.append(cmd)
        result = plant.step(cmds)
        obs = result.obs
        if result.done:
            return time.perf_counter() - t0


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    bare(0)  # warm the imports and the caches
    for name, fn in (
        ("bare plant, no commands", bare),
        ("slow path, a command every quarter", slow),
        ("plant + job runner, 4 jobs", with_runner),
    ):
        ts = [fn(s) for s in range(n)]
        print(
            f"{name:38s} min {min(ts):.3f} s  median {statistics.median(ts):.3f} s  "
            f"max {max(ts):.3f} s  (n={n})"
        )


if __name__ == "__main__":
    main()
