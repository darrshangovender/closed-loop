"""Scripted episodes for the plant tests. SYNTHETIC; fixed-quarter schedules, not a job runner.

A schedule maps a quarter to the commands issued at it. The parameters below switch off every random
refusal and shorten approvals to one quarter, so that a fixed schedule is deterministic.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from closed_loop.core.types import Cmd, Command, Permit, PlanCmd
from closed_loop.plant import PlantParams, Scenario, make_plant
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.plant import F1Plant

CR, CV, P1, P2 = "CR-501", "CV-101", "PU-201", "PU-202"

Schedule = Mapping[int, Sequence[Command]]


def scripted_params(**overrides: object) -> PlantParams:
    """No random refusals, one-quarter approvals, holders always present, nothing fails early."""
    base: dict[str, object] = {
        "p_refuse": 0.0,
        "approver_median_h": 0.25,
        "approver_sigma": 0.0,
        "holder_present_p": 1.0,
        "gas_excursion_rate": 0.0,
        "p_gain_fault": 0.0,
        "p_stuck_day": 0.0,
        "p_spike_day": 0.0,
        "upset_p_per_circuit": 0.0,
        "p_fail_in_episode": 0.0,
    }
    base.update(overrides)
    return PlantParams(**base)  # type: ignore[arg-type]


def C(cmd: Cmd | PlanCmd, asset: str = CR, arg: str | None = None) -> Command:
    return Command(asset, cmd, arg, source="test")


def at(**by_quarter: Sequence[Command]) -> dict[int, Sequence[Command]]:
    """``at(q0=[...], q5=[...])``: a schedule written with the quarter in the keyword."""
    return {int(k.lstrip("q")): v for k, v in by_quarter.items()}


def drive(plant: F1Plant, schedule: Schedule, until: int) -> list[tuple[str, str, int]]:
    """Step quarters 0..until inclusive; return every violation the plant recorded."""
    out: list[tuple[str, str, int]] = []
    for q in range(until + 1):
        out.extend(plant.step(list(schedule.get(q, ()))).violations)
    return out


def advance(plant: F1Plant, schedule: Schedule, until: int):
    """Continue from the current quarter up to and including ``until``; return the last result."""
    result = None
    while plant.des.q <= until:
        result = plant.step(list(schedule.get(plant.des.q, ())))
    return result


def fresh(seed: int = 5, scenario: Scenario | None = None, **overrides: object) -> F1Plant:
    p = make_plant(scripted_params(**overrides))
    p.reset(seed, scenario if scenario is not None else Scenario())
    return p


def permit_runs(permit: np.ndarray) -> list[tuple[int, int]]:
    """``(first row, rows)`` of every unbroken run of permit values other than NONE."""
    none = list(Permit).index(Permit.NONE)
    held = permit != none
    runs: list[tuple[int, int]] = []
    start = None
    for r, h in enumerate(held):
        if h and start is None:
            start = r
        if not h and start is not None:
            runs.append((start, r - start))
            start = None
    if start is not None:
        runs.append((start, len(held) - start))
    return runs


def seed_with_mode(asset: str, mode: str, start: int = 0) -> int:
    """First seed at or after ``start`` whose first life of ``asset`` has failure mode ``mode``."""
    params = scripted_params()
    for seed in range(start, start + 400):
        if simulate_lives(params, seed, Scenario()).lives[asset].mode == mode:
            return seed
    raise AssertionError(f"no seed with {asset} {mode}")


def isolate(start: int = 0, asset: str = CR) -> dict[int, list[Command]]:
    """STOP, LOCK, PROVE, ISSUE on consecutive quarters from ``start``."""
    return {
        start + i: [C(c, asset)] for i, c in enumerate((Cmd.STOP, Cmd.LOCK, Cmd.PROVE, Cmd.ISSUE))
    }


def merge(*parts: Schedule) -> dict[int, list[Command]]:
    out: dict[int, list[Command]] = {}
    for part in parts:
        for q, cmds in part.items():
            out.setdefault(q, []).extend(cmds)
    return out


def confined_job(mode: str, raise_at: int = 3) -> dict[int, list[Command]]:
    """A complete correct crusher liner job: the schedule derived in tests/plant/test_violations.py.

    Raised at the start of the day shift. Entry at 10, tests every 2 quarters, exit before the
    handover at 31, revalidation and a fresh test at 32, re-entry at 33, exit at 52, then COMPLETE,
    CLOSE, UNISO and START. 40 quarters of work inside the chamber.
    """
    s: dict[int, list[Command]] = {raise_at: [C(PlanCmd.RAISE_WO, CR, mode)]}
    s.update(isolate(4))
    s[8] = [C(Cmd.BEGIN)]
    s[9] = [C(Cmd.TEST)]
    s[10] = [C(Cmd.ENTER)]
    for q in range(11, 30, 2):
        s[q] = [C(Cmd.TEST)]
    s[31] = [C(Cmd.EXIT)]
    s[32] = [C(Cmd.REVAL), C(Cmd.TEST)]
    s[33] = [C(Cmd.ENTER)]
    for q in range(34, 51, 2):
        s[q] = [C(Cmd.TEST)]
    s[52] = [C(Cmd.EXIT)]
    s[53] = [C(Cmd.COMPLETE)]
    s[54] = [C(Cmd.CLOSE)]
    s[55] = [C(Cmd.UNISO)]
    s[56] = [C(Cmd.START)]
    return s


def open_job(mode: str, raise_at: int = 3) -> dict[int, list[Command]]:
    """A complete job on a crusher repair that needs no entry (bearing): 24 quarters of work."""
    s: dict[int, list[Command]] = {raise_at: [C(PlanCmd.RAISE_WO, CR, mode)]}
    s.update(isolate(4))
    s[8] = [C(Cmd.BEGIN)]
    s[32] = [C(Cmd.COMPLETE)]
    s[33] = [C(Cmd.CLOSE)]
    s[34] = [C(Cmd.UNISO)]
    s[35] = [C(Cmd.START)]
    return s


def shifted(schedule: Schedule, dq: int) -> dict[int, list[Command]]:
    """The same schedule ``dq`` quarters later; a multiple of 96 keeps the shift phase."""
    return {q + dq: list(v) for q, v in schedule.items()}


def run_with_runner(
    plant: F1Plant,
    plan: Mapping[int, Sequence[Command]],
    runner=None,
    until: int | None = None,
    skip: Sequence[str] = (),
):
    """Step the plant with the JobRunner driving every open job; ``plan`` adds fixed commands.

    Returns (violations, history) where history lists (quarter, command, executed) for the runner's
    commands. ``skip`` names assets the runner must leave alone.
    """
    from closed_loop.plant.jobs import JobRunner

    runner = runner if runner is not None else JobRunner()
    obs = plant.initial_observation
    violations: list[tuple[str, str, int]] = []
    history: list[tuple[int, Command, bool]] = []
    horizon = plant.params.horizon_q - 1 if until is None else until
    while plant.des.q <= horizon:
        q = plant.des.q
        cmds = list(plan.get(q, ()))
        mine = []
        for aid, view in obs.assets.items():
            if aid in skip or view.job is None:
                continue
            c = runner.next_command(view, view.job, quarter=obs.quarter)
            if c is not None:
                mine.append(c)
        r = plant.step(cmds + mine)
        violations.extend(r.violations)
        for c in mine:
            history.append((q, c, c in r.executed))
        obs = r.obs
        if r.done:
            break
    return violations, history
