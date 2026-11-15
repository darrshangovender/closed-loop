"""ADR 0003 A12.1 and A12.2 in the plant: a permit lives 12 hours from its issue, a locked pump is
not available.

The regressions here are the plant half of the three holes the closed loop found (P12a F2, F3):
the plant restarted the 12 hours at each revalidation while the specification counted from the
issue, and the plant counted a locked pump as available while the specification did not. The rule
the tests keep is that the plant is at least as strict as the monitor: wherever the specification
reads a violation of P2 or P4 the plant records it at the same row.
"""

import numpy as np
import pytest

from closed_loop.core.types import Cmd, Permit
from closed_loop.plant import PlantParams, Scenario, make_plant
from closed_loop.plant.des import PUMP_PAIR_ID
from closed_loop.plant.observe import simulate_lives
from closed_loop.proposers import make_baseline
from closed_loop.spec.properties import SpecParams, evaluate, signals_from_trace
from tests.proposers._loop import run_episode

from ._scripts import (
    CR,
    CV,
    P1,
    P2,
    C,
    at,
    drive,
    fresh,
    isolate,
    merge,
    permit_runs,
    scripted_params,
)

PUMPS = (P1, P2)
NONE, VALID, REVAL = (list(Permit).index(p) for p in (Permit.NONE, Permit.VALID, Permit.REVAL))


def first_spec_p2_row(tr, spec: SpecParams | None = None) -> int | None:
    """First row at which the specification's P2 is negative (inside, permit not ok)."""
    sig = signals_from_trace(tr, spec or SpecParams(), CR, PUMPS)
    bad = (sig["crusher.in_space"] > 0) & (sig["crusher.permit_ok"] < 0)
    hits = np.flatnonzero(bad)
    return int(hits[0]) if hits.size else None


class TestAPermitLivesTwelveHoursFromIssue:
    def test_a_revalidation_does_not_move_the_limit(self):
        plant = fresh()
        drive(plant, merge(isolate(0), at(q32=[C(Cmd.REVAL)])), until=60)
        tr = plant.trace()
        # issued at 3, so spent at the end of quarter 50 (3 + 48 = 51 is reached); revalidated at
        # 32 and valid until then, not until 32 + 48
        assert tr.permit[CR][31] == REVAL and tr.permit[CR][32] == VALID
        assert tr.permit[CR][49] == VALID and tr.permit[CR][50] == NONE
        assert tr.permit_margin_min[CR][49] == (51 - 49) * 15  # the limit, not the shift end (64)

    def test_a_permit_nobody_revalidated_is_spent_at_the_limit_too(self):
        plant = fresh()
        drive(plant, isolate(0), until=60)
        tr = plant.trace()
        assert tr.permit[CR][31] == REVAL and tr.permit[CR][49] == REVAL
        assert tr.permit[CR][50] == NONE and (tr.permit[CR][50:] == NONE).all()

    def test_no_permit_run_in_the_trace_reaches_the_twelve_hours(self):
        plant = fresh()
        drive(plant, merge(isolate(0), at(q32=[C(Cmd.REVAL)])), until=120)
        runs = permit_runs(plant.trace().permit[CR])
        assert runs == [(3, 47)]  # rows 3 to 49; the specification's cap allows 48 rows from issue

    def test_a_new_issue_after_the_permit_is_spent_starts_a_new_limit(self):
        plant = fresh()
        drive(plant, merge(isolate(0), at(q52=[C(Cmd.ISSUE)])), until=110)
        tr = plant.trace()
        assert tr.permit[CR][50] == NONE and tr.permit[CR][52] == VALID
        assert tr.permit_margin_min[CR][52] == (64 - 52) * 15  # the shift ends before the new limit
        assert tr.permit[CR][98] == REVAL and tr.permit[CR][99] == NONE  # 52 + 48 = 100 reached
        assert permit_runs(tr.permit[CR]) == [(3, 47), (52, 47)]

    def test_the_limit_follows_the_permit_hours_of_the_spec_params(self):
        class Spec:
            retest_q, break_q, alarm_exit_q, permit_max_h = 2, 1, 1, 3.0  # 12 quarters

        plant = make_plant(scripted_params(), spec=Spec())
        plant.reset(5, Scenario())
        drive(plant, isolate(0), until=20)
        tr = plant.trace()
        assert tr.permit[CR][13] == VALID and tr.permit[CR][14] == NONE  # 3 + 12 = 15 reached

    def test_someone_inside_when_the_permit_is_spent_is_a_p2_violation_the_spec_agrees_with(self):
        """The seed 1001 mechanism in miniature: a liner job that crosses a handover and is still
        inside 12 hours after the issue. The plant used to restart the clock at the revalidation
        (and record nothing); the specification counts from the issue and read a violation."""
        sched = merge(
            isolate(0),  # ISSUE at 3, so the limit is quarter 51
            at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)]),
            {q: [C(Cmd.TEST)] for q in range(7, 30, 2)},
            at(q31=[C(Cmd.EXIT)], q32=[C(Cmd.REVAL), C(Cmd.TEST)], q33=[C(Cmd.ENTER)]),
            {q: [C(Cmd.TEST)] for q in range(34, 60, 2)},
        )
        plant = fresh()
        got = drive(plant, sched, until=52)
        assert got == [("P2", CR, 50)]
        assert first_spec_p2_row(plant.trace()) == 50  # the same row, in the same direction
        assert evaluate(plant.trace(), SpecParams(), CR, PUMPS)["P2"] < 0

    def test_leaving_on_soon_keeps_the_job_clean(self):
        """The same job, but the occupant leaves when the view says the permit ends next quarter."""
        plant = fresh()
        sched = merge(
            isolate(0),
            at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)]),
            {q: [C(Cmd.TEST)] for q in range(7, 30, 2)},
            at(q31=[C(Cmd.EXIT)], q32=[C(Cmd.REVAL), C(Cmd.TEST)], q33=[C(Cmd.ENTER)]),
            {q: [C(Cmd.TEST)] for q in range(34, 49, 2)},
            at(q50=[C(Cmd.EXIT)]),
        )
        assert drive(plant, sched, until=52) == []
        assert evaluate(plant.trace(), SpecParams(), CR, PUMPS)["P2"] >= 0


class TestIssueInWork:
    def short(self):
        return fresh(permit_max_h=1.0)  # 4 quarters: issued at 3, spent at the end of quarter 6

    def test_a_spent_permit_in_work_with_nobody_inside_is_replaced(self):
        plant = self.short()
        drive(plant, merge(isolate(0), at(q4=[C(Cmd.BEGIN)])), until=6)
        assert plant.des.by_id[CR].permit is Permit.NONE and plant.des.by_id[CR].op.value == "WORK"
        r = plant.step([C(Cmd.ISSUE)])  # q = 7
        assert not r.refused and r.obs.assets[CR].permit is Permit.VALID
        assert r.obs.assets[CR].permit_limit_left_q == 11 - 8  # limit 7 + 4, decision quarter 8

    def test_with_someone_inside_it_is_refused(self):
        plant = self.short()
        got = drive(
            plant, merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)])), 6
        )
        assert got == [("P2", CR, 6)]  # inside at the quarter the permit is spent
        r = plant.step([C(Cmd.ISSUE)])  # q = 7, still inside
        assert r.refused and r.obs.assets[CR].permit is Permit.NONE
        plant.step([C(Cmd.EXIT)])  # q = 8
        r = plant.step([C(Cmd.ISSUE)])  # q = 9, empty
        assert not r.refused and r.obs.assets[CR].permit is Permit.VALID

    def test_an_issue_with_a_permit_already_held_changes_nothing(self):
        plant = fresh()
        drive(plant, merge(isolate(0), at(q4=[C(Cmd.BEGIN)])), until=5)
        before = plant.des.by_id[CR].limit_q
        r = plant.step([C(Cmd.ISSUE)])
        assert not r.refused and plant.des.by_id[CR].limit_q == before

    @pytest.mark.parametrize("asset", [CR, CV, P1])
    def test_it_is_refused_on_a_running_asset(self, asset):
        plant = fresh()
        r = plant.step([C(Cmd.ISSUE, asset)])
        assert r.refused and plant.des.by_id[asset].permit is Permit.NONE


class TestSoonAndTheViewOfTheLimit:
    def test_soon_is_true_one_quarter_before_the_permit_reaches_its_limit(self):
        plant = fresh(permit_max_h=2.0)  # issued at 3, limit 11
        seen = {}
        for q in range(14):
            r = plant.step(list(isolate(0).get(q, ())))
            seen[r.obs.quarter] = (r.obs.assets[CR].soon, r.obs.assets[CV].soon)
        assert seen[9] == (False, False)
        assert seen[10] == (True, False)  # the step at 10 ends the permit; the conveyor has none
        assert seen[11] == (False, False)  # spent: nothing left to announce

    def test_soon_still_announces_the_handover_for_every_asset(self):
        plant = fresh()
        for _ in range(31):
            r = plant.step([])
        assert r.obs.quarter == 31 and all(v.soon for v in r.obs.assets.values())
        r = plant.step([])
        assert r.obs.quarter == 32 and not any(v.soon for v in r.obs.assets.values())

    def test_a_revalidated_permit_announces_its_limit_not_the_shift_end(self):
        plant = fresh()
        drive(plant, merge(isolate(0), at(q32=[C(Cmd.REVAL)])), until=48)
        r = plant.step([])  # q = 49: the view for decision quarter 50
        v = r.obs.assets[CR]
        assert r.obs.quarter == 50 and v.permit is Permit.VALID
        assert v.soon and v.permit_limit_left_q == 1  # the step at 50 ends the permit
        assert not r.obs.assets[CV].soon and r.obs.assets[CV].permit_limit_left_q is None
        r = plant.step([])  # q = 50
        assert r.obs.assets[CR].permit is Permit.NONE and not r.obs.assets[CR].soon

    def test_no_permit_no_limit_on_the_view(self):
        plant = fresh()
        assert plant.initial_observation.assets[CR].permit_limit_left_q is None


class TestALockedPumpIsNotAvailable:
    def test_both_pumps_locked_is_a_p4_violation_when_the_second_lock_lands(self):
        plant = fresh()
        sched = {0: [C(Cmd.STOP, P1), C(Cmd.STOP, P2)], 1: [C(Cmd.LOCK, P1), C(Cmd.LOCK, P2)]}
        assert drive(plant, sched, until=4) == [("P4", PUMP_PAIR_ID, 1)]
        sig = signals_from_trace(plant.trace(), SpecParams(), CR, PUMPS)["pumps.available"]
        assert (sig[1:5] < 0).all() and sig[0] > 0  # the specification reads the same rows

    def test_one_locked_pump_with_the_other_running_is_fine(self):
        plant = fresh()
        sched = {0: [C(Cmd.STOP, P1)], 1: [C(Cmd.LOCK, P1)], 2: [C(Cmd.PROVE, P1)]}
        assert drive(plant, sched, until=6) == []

    def test_a_locked_pump_and_a_failure_of_the_other_pump_is_a_p4_violation(self):
        """P12a F2 seed 1167, tick 7399 in miniature: PU-201 is locked and PU-202 fails."""
        params = scripted_params(p_fail_in_episode=0.6)
        seed = next(
            s
            for s in range(400)
            if simulate_lives(params, s, Scenario()).fails_in_horizon(P2)
            and simulate_lives(params, s, Scenario()).failure_day[P2] > 1.0
        )
        table = simulate_lives(params, seed, Scenario())
        fail_q = round(table.failure_day[P2] * 96)
        plant = fresh(seed, p_fail_in_episode=0.6)
        got = []
        for q in range(fail_q + 2):
            cmds = [C(Cmd.STOP, P1)] if q == 0 else [C(Cmd.LOCK, P1)] if q == 1 else []
            got.extend(plant.step(cmds).violations)
        assert [v for v in got if v[0] == "P4"] == [("P4", PUMP_PAIR_ID, fail_q - 1)]
        tr = plant.trace()
        sig = signals_from_trace(tr, SpecParams(), CR, PUMPS)["pumps.available"]
        assert int(np.flatnonzero(sig < 0)[0]) == fail_q - 1  # the monitor's row is the plant's
        assert evaluate(tr, SpecParams(), CR, PUMPS)["P4"] < 0


@pytest.mark.parametrize("name", ["B0", "B1", "B2"])
def test_no_permit_older_than_twelve_hours_in_any_trace_of_an_honest_baseline(name):
    """The regression of A12.1: whatever an honest baseline does for 90 days, no unbroken run of
    permit values other than NONE reaches the 12 hours (48 rows) the specification allows."""
    longest = 0
    for seed in range(1, 9):
        ep = run_episode(make_baseline(name), seed)
        assert ep.trace is not None
        for aid in ep.trace.asset_ids:
            runs = permit_runs(ep.trace.permit[aid])
            longest = max([longest, *(n for _s, n in runs)])
    assert longest <= 47, longest


@pytest.mark.parametrize("name", ["B0", "B2"])
def test_an_honest_job_on_an_unshielded_arm_shows_no_p2_and_no_p2_or_p4_disagreement(name):
    """P12a: the P2 readings of honest runs were the bulk of P0's violations and the plant never
    recorded them. With the limit in the plant, honest crews leave on `soon`: no P2 by the monitor,
    and the plant and the monitor agree about P4 in both directions."""
    spec = SpecParams()
    for seed in range(1, 13):
        ep = run_episode(make_baseline(name), seed)
        assert ep.trace is not None
        rob = evaluate(ep.trace, spec, CR, PUMPS)
        assert rob["P2"] >= 0 and ep.violated("P2") == 0, (name, seed)
        assert (rob["P4"] < 0) == (ep.violated("P4") > 0), (name, seed)


def test_the_default_parameters_give_a_twelve_hour_limit_in_a_job_with_entries():
    """The honest baselines do enter the crusher in these episodes (the tests above would be
    vacuous otherwise)."""
    assert PlantParams().permit_max_h == 12.0
    entered = 0
    for seed in range(1, 9):
        ep = run_episode(make_baseline("B0"), seed)
        entered += sum(1 for _q, c in ep.executed if c.cmd is Cmd.ENTER)
    assert entered > 0
