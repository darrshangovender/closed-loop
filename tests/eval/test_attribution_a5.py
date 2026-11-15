"""0003 A5 in the loop: a shielded arm's violations are explained, or they are defects.

The test is the one the brief asks for: Careless around B3 with the shield on, 200 episodes of the
nominal condition on development seeds 1000 to 1199, zero P1 to P5 violations that the trace
attributes ``assumption_held``. A violation attributed to C1 or A_phys is a contract failure, not
a defect; it is counted. If this test goes red, do not touch the shield, the plant or the
attribution to make it green: the failing seed and tick are in the message, and they are findings.

History. P12a ran this check on 1 Nov 2026 and it failed: 119 episodes held P2 violations attributed
``assumption_held`` (first at seed 1001, tick 7170), one held a P4 violation (seed 1167, tick 7399)
and the plant recorded 41 P1 events on CV-101 that the specification could not see. ADR 0003 A12
ruled that the plant and the proof move to meet the monitors and that no monitor is loosened: a
permit lives 12 hours from its issue (A12.1), a locked pump is not available (A12.2), and the
conveyor is inside the proof (A12.3). The check then returned zero (packet P5d-a; the numbers are in
its note). The strict ``xfail`` that recorded the failure is gone. The vacuity ``xfail`` that
stood below it said, in its own words, that it would turn red the day the plant runs behind the
shield (mean throughput 0.127 in the P5d-a run: nothing restarted a stopped crusher). The
operator's daily restart offer (0003 A18, ruling R4) did that, the strict ``xfail`` passed
unexpectedly, and it is now a plain test: the zero-violation checks above run on a plant that
runs.
"""

from __future__ import annotations

from collections import Counter

import pytest

from closed_loop.core.trace import Trace
from closed_loop.core.types import Op, Permit
from closed_loop.eval.arms import ARMS
from closed_loop.eval.context import PreparedContext
from closed_loop.eval.episode import EpisodeResult, scenario_for, simulate_episode
from closed_loop.eval.metrics import P15
from closed_loop.plant.params import CRUSHER, PUMPS

SEEDS = range(1000, 1200)
PERMIT_MAX_ROWS = 48  # CORP-HSE-004 clause 4.1: 12 hours from issue, in quarters
_NONE = list(Permit).index(Permit.NONE)
_PUMP_GONE = [list(Op).index(o) for o in (Op.LOCKED, Op.ISO, Op.WORK, Op.FAIL)]


def longest_permit_run(trace: Trace) -> int:
    """Rows in the longest unbroken run of permit values other than NONE, over every asset."""
    longest = 0
    for aid in trace.asset_ids:
        run = 0
        for held in trace.permit[aid] != _NONE:
            run = run + 1 if held else 0
            longest = max(longest, run)
    return longest


def both_pumps_gone(trace: Trace) -> int:
    """Rows in which neither pump is in RUN or STOP."""
    a, b = PUMPS
    gone = [
        pa in _PUMP_GONE and pb in _PUMP_GONE
        for pa, pb in zip(trace.op[a].tolist(), trace.op[b].tolist(), strict=True)
    ]
    return sum(gone)


@pytest.fixture(scope="module")
def episodes(full_context: PreparedContext) -> list[tuple[EpisodeResult, int, int]]:
    """Each episode's result, the longest permit run in its trace and its rows without a pump."""
    arm = ARMS["H-C1"]
    out: list[tuple[EpisodeResult, int, int]] = []
    for s in SEEDS:
        rec = simulate_episode(arm, s, scenario_for("nominal"), context=full_context)
        out.append((rec.result, longest_permit_run(rec.trace), both_pumps_gone(rec.trace)))
    return out


@pytest.fixture(scope="module")
def runs(episodes: list[tuple[EpisodeResult, int, int]]) -> list[EpisodeResult]:
    return [r for r, _permit, _pumps in episodes]


@pytest.mark.slow
def test_careless_with_the_shield_has_no_assumption_held_violation_of_p1_to_p5(
    runs: list[EpisodeResult], record_property: pytest.RecordProperty
) -> None:
    defects = [
        (r.seed, a.prop, a.row)
        for r in runs
        for a in r.attributions
        if a.prop in P15 and a.label == "assumption_held"
    ]
    contract = Counter(a.label for r in runs for a in r.attributions if a.prop in P15)
    record_property("episodes", len(runs))
    record_property("attributed", dict(contract))
    assert not defects, (
        f"{len(defects)} unexplained P1-P5 violation(s) behind the shield; first: seed "
        f"{defects[0][0]}, property {defects[0][1]}, tick {defects[0][2]}; all: {defects[:10]}"
    )
    assert len(runs) == 200


@pytest.mark.slow
def test_the_specification_and_the_plant_agree_on_p1_p2_and_p4_behind_the_shield(
    runs: list[EpisodeResult],
) -> None:
    """The three holes of 0003 A12, as counts of disagreement between the monitor and the plant's
    own checks, in the 200 episodes: P2 and P4 read by the specification and not recorded by the
    plant (119 and 1 episodes before), P1 recorded by the plant and not read by the specification
    (37 episodes before). All three are zero now, and so is every other disagreement."""
    kinds = Counter(f"{p}:{k}" for r in runs for p, k in r.disagreements.items())
    assert kinds["P2:spec_only"] == 0 and kinds["P4:spec_only"] == 0
    assert kinds["P1:plant_only"] == 0
    assert dict(kinds) == {}, dict(kinds)
    assert sum(r.plant_violations.get("P1", 0) for r in runs) == 0  # nor does the plant record one


@pytest.mark.slow
def test_no_permit_in_the_200_traces_is_older_than_twelve_hours(
    episodes: list[tuple[EpisodeResult, int, int]],
) -> None:
    """A12.1 on whole episodes: the longest unbroken run of non-NONE permit values, any asset, is
    inside the 12 hours the specification counts from the issue."""
    longest = max(permit for _r, permit, _pumps in episodes)
    assert 0 < longest <= PERMIT_MAX_ROWS - 1, longest  # and there were permits to count


@pytest.mark.slow
@pytest.mark.usefixtures(
    "provisional_defaults"
)  # the seed and tick are those of the provisional B3
def test_seed_1001_tick_7170_the_liner_job_across_handovers_is_no_longer_a_p2_violation(
    full_context: PreparedContext,
) -> None:
    """P12a F2: the crusher in WORK, occupied, permit VALID, 12.75 hours after the permit run began
    at row 7119, with two revalidations in between. The plant restarted the 12 hours at each
    revalidation (and recorded nothing); the specification counted from the issue."""
    rec = simulate_episode(ARMS["H-C1"], 1001, scenario_for("nominal"), context=full_context)
    r = rec.result
    assert r.violations["P2"] == 0 and r.min_robustness["P2"] >= 0.0
    assert "P2" not in r.disagreements and r.plant_violations.get("P2", 0) == 0
    assert not [a for a in r.attributions if a.prop == "P2"]
    assert longest_permit_run(rec.trace) <= PERMIT_MAX_ROWS - 1
    # the mechanism is not merely out of reach: the liner job still runs, with someone inside
    assert int((rec.trace.permit[CRUSHER] != _NONE).sum()) > 0
    assert int(rec.trace.occupied[CRUSHER].sum()) > 0


@pytest.mark.slow
def test_seed_1167_tick_7399_a_locked_pump_and_a_failing_pump_is_a_p4_violation_of_both(
    full_context: PreparedContext,
) -> None:
    """P12a F2: PU-201 locked and PU-202 failing. The plant and the abstraction counted a locked
    pump as available; the specification did not. Now they agree, and the shield does not lock a
    pump whose neighbour may fail."""
    rec = simulate_episode(ARMS["H-C1"], 1167, scenario_for("nominal"), context=full_context)
    r = rec.result
    assert r.violations["P4"] == 0 and r.min_robustness["P4"] >= 0.0
    assert "P4" not in r.disagreements and r.plant_violations.get("P4", 0) == 0
    assert both_pumps_gone(rec.trace) == 0
    assert not [a for a in r.attributions if a.prop == "P4"]


@pytest.mark.slow
def test_the_a5_run_exercised_a_plant_that_was_running(runs: list[EpisodeResult]) -> None:
    mean_throughput = sum(r.throughput for r in runs) / len(runs)
    assert mean_throughput > 0.5, f"mean throughput behind the shield was {mean_throughput:.3f}"
