"""Every arm of the registry runs, under every condition the registry lists for it."""

from __future__ import annotations

import math

import pytest

from closed_loop.core.types import Fidelity
from closed_loop.eval import tuned
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import EpisodeResult, run_episode, scenario_for

SEED = 1100  # a development seed

CASES = [(name, cond) for name, a in ARMS.items() for cond in a.conditions]


def check_filled(r: EpisodeResult, name: str, cond: str) -> None:
    arm = ARMS[name]
    assert (r.arm, r.seed, r.condition, r.fidelity) == (name, SEED, cond, Fidelity.F1)
    assert set(r.violations) == set(r.min_robustness) == {"P1", "P2", "P3", "P4", "P5", "P6"}
    assert all(math.isfinite(v) for v in r.min_robustness.values())
    assert 0.0 <= r.throughput <= 1.0 and r.cost >= 0.0
    assert r.planned_jobs >= 0 and r.unplanned_failures >= 0 and r.wall_s > 0.0
    if arm.perception == "none":
        assert r.labels == () and r.checkpoints == ()
    else:
        assert r.labels and r.checkpoints and set(r.monitors) >= {"opt", "pess", "inf", "proxy"}
        assert r.contract.checkpoints == len(r.checkpoints)
    if arm.shield:
        assert r.shield_overrides + r.shield_admitted > 0 and r.allowed_crusher[1] > 0
        assert r.interlock_admitted == 0
    elif arm.interlock:
        assert r.interlock_admitted + r.interlock_refused + r.interlock_forced > 0
        assert r.shield_overrides == r.shield_admitted == 0
    else:
        assert r.shield_overrides == r.shield_admitted == r.interlock_admitted == 0
    assert (r.shift_day is not None) == (cond == "shifted")
    # every number the arm reads that tuned.json does not give is flagged: all of them while the
    # file is absent (when this test was written), none now that it is final (benchmarks/tuned.json)
    assert set(r.provisional) == set(tuned.current().provisional_names(arm.tuned))


@pytest.mark.slow
@pytest.mark.parametrize(("name", "cond"), CASES, ids=[f"{n}-{c}" for n, c in CASES])
def test_the_arm_runs_and_fills_its_result(
    name: str, cond: str, small_context: PreparedContext
) -> None:
    r = run_episode(ARMS[name], SEED, scenario_for(cond), context=small_context)
    check_filled(r, name, cond)


def test_the_cases_are_the_registry_conditions() -> None:
    # 35 arms, 26 of them under both conditions and the 9 hostile and spec arms nominal only
    assert len(ARMS) == 35
    assert len(CASES) == sum(len(a.conditions) for a in ARMS.values()) == 61
