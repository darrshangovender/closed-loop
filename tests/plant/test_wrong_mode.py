"""P3b / 0003 A3: a work order with the wrong mode completes a job and does not renew the component.

The cost is real and the life continues. A correct job renews it. A diagnosis of "sensor fault" or
"process upset" raises no job, so a wrong one is a missed intervention: the part fails.
"""

import pytest

from closed_loop.core.types import Cmd, Op, PlanCmd
from closed_loop.plant import Scenario
from closed_loop.plant.observe import simulate_lives

from ._scripts import (
    CR,
    CV,
    C,
    advance,
    confined_job,
    fresh,
    open_job,
    scripted_params,
    shifted,
)

DAY60 = 60 * 96
# every asset is designed to fail inside the horizon, and nobody catches a wrong work order
OV = {"p_fail_in_episode": 1.0, "approver_catch_p": 0.0}


def degraded_liner_seed() -> int:
    """A crusher whose liner is wearing at day 60 and fails late in the episode with no action."""
    params = scripted_params(p_fail_in_episode=1.0)
    for seed in range(600):
        t = simulate_lives(params, seed, Scenario())
        life = t.lives[CR]
        if life.mode == "LINER_WEAR" and t.onset_day[CR] < 50 and 80 < t.failure_day[CR] < 86:
            return seed
    raise AssertionError("no suitable seed")


SEED = degraded_liner_seed()


def run_job(schedule):
    """Run ``schedule`` (shifted to day 60) on a plant and a bare baseline to the same quarter."""
    plant = fresh(SEED, **OV)
    base = fresh(SEED, **OV)
    end = DAY60 + 160
    advance(plant, schedule, until=end)
    advance(base, {}, until=end)
    return plant, base


def wrong_worlds():
    # claims BEARING; the liner is what is worn
    return run_job(shifted(open_job("BEARING_DEFECT"), DAY60))


def right_worlds():
    return run_job(shifted(confined_job("LINER_WEAR"), DAY60))


class TestWrongMode:
    def test_the_job_is_done_and_paid_for(self):
        plant, _ = wrong_worlds()
        st = plant.stats()
        assert (st.planned_jobs, st.wrong_mode_jobs, st.wo_rejected) == (1, 1, 0)
        assert st.cost == pytest.approx(plant.params.planned_job_cost)
        assert plant.truth.life_mode[CR] == "LINER_WEAR"  # nobody swapped the part that was failing

    def test_the_component_is_not_renewed(self):
        plant, _ = wrong_worlds()
        assert plant.truth.life_id[CR] == 0
        assert plant.des.by_id[CR].renewals == 0
        assert plant.des.by_id[CR].life.life_id == 0

    def test_the_asset_is_back_in_service_and_the_work_order_closed(self):
        plant, _ = wrong_worlds()
        s = plant.des.by_id[CR]
        assert s.op is Op.RUN and s.job is None and s.jobs_done == 1
        obs = plant.step([]).obs.assets[CR]
        assert (
            obs.jobs_done == 1 and not obs.job_open
        )  # an operator sees a completed job, not its fate

    def test_the_wear_clock_was_paused_by_the_stop_and_then_ran_on(self):
        plant, base = wrong_worlds()
        sp, sb = plant.des.by_id[CR], base.des.by_id[CR]
        q = DAY60 + 160
        gained = sb.clock.remaining_clock(q) - sp.clock.remaining_clock(q)
        # stopped from quarter 4 to 34 of the job (31 quarters): about a third of a clock day behind
        assert -0.5 < gained < -0.1, gained
        assert plant.truth.true_rul_days[CR] > base.truth.true_rul_days[CR]
        assert plant.truth.true_rul_days[CR] - base.truth.true_rul_days[CR] < 0.6

    def test_the_degradation_signature_survives_the_job(self):
        plant, base = wrong_worlds()
        a, b = plant.step([]).obs.assets[CR], base.step([]).obs.assets[CR]
        assert a.last_obs is not None and b.last_obs is not None
        # same noise, same wear to within a third of a clock day: the readings agree
        assert abs(a.last_obs.sig_proc - b.last_obs.sig_proc) < 0.5
        assert plant.des.by_id[CR].clock.severity(plant.des.q) > 0.05  # still worn

    def test_a_wrong_job_does_not_stop_the_failure_that_was_coming(self):
        plant, base = wrong_worlds()
        # the baseline fails at day F; the wrong-job world fails only a third of a day later
        t = simulate_lives(scripted_params(p_fail_in_episode=1.0), SEED, Scenario())
        fail_q = round(t.failure_day[CR] * 96)
        for p in (plant, base):
            advance(p, {}, until=fail_q + 80)
        for p in (plant, base):
            assert p.trace().uncontrolled_failure[CR].any()
        assert plant.stats().unplanned_failures >= 1
        late = int(plant.trace().uncontrolled_failure[CR].argmax())
        early = int(base.trace().uncontrolled_failure[CR].argmax())
        assert 0 < late - early < 60  # about the 31 quarters the machine stood still


class TestCorrectMode:
    def test_the_component_is_renewed(self):
        plant, _ = right_worlds()
        assert plant.truth.life_id[CR] == 1
        st = plant.stats()
        assert (st.planned_jobs, st.wrong_mode_jobs) == (1, 0)
        assert plant.des.by_id[CR].renewals == 1

    def test_the_remaining_life_jumps_and_the_readings_return_to_baseline(self):
        plant, base = right_worlds()
        assert plant.truth.true_rul_days[CR] > base.truth.true_rul_days[CR] + 25
        assert plant.truth.health_bin[CR] == 0
        assert plant.des.by_id[CR].clock.severity(plant.des.q) == 0.0

    def test_a_correct_job_is_not_disturbed_by_the_wrong_one_before_it(self):
        plant = fresh(SEED, **OV)
        wrong = shifted(open_job("BEARING_DEFECT"), DAY60)
        right = shifted(confined_job("LINER_WEAR"), DAY60 + 96)  # the next day: same shift phase
        advance(plant, {**wrong, **right}, until=DAY60 + 96 + 160)
        st = plant.stats()
        assert (st.planned_jobs, st.wrong_mode_jobs) == (2, 1)
        assert plant.truth.life_id[CR] == 1  # only the second job renewed it
        assert st.cost == pytest.approx(2 * plant.params.planned_job_cost)


def test_the_wrong_job_also_consumed_a_part_that_the_right_one_did_not_need():
    plant = fresh(SEED, **OV)
    advance(plant, shifted(open_job("BEARING_DEFECT"), DAY60), until=DAY60 + 100)
    spares = plant.step([]).obs.assets[CR].spares
    assert spares["SYN-BRG-5001"] == 0 and spares["SYN-LNR-5001"] == 1


def test_a_missed_diagnosis_raises_no_job_so_the_part_fails():
    """'sensor_fault' or 'process_upset' means no work order; if the part was failing, it fails."""
    params = scripted_params(p_fail_in_episode=0.6)
    seed = next(s for s in range(300) if simulate_lives(params, s, Scenario()).fails_in_horizon(CV))
    plant = fresh(seed, p_fail_in_episode=0.6)
    while not plant.step(()).done:
        pass
    st = plant.stats()
    assert st.unplanned_failures >= 1 and st.planned_jobs == 0
    assert st.cost >= plant.params.unplanned_cost_ratio * plant.params.planned_job_cost


def test_on_a_failed_asset_a_wrong_repair_leaves_the_failure_in_place():
    """The repair is done and the asset restarted, and it fails again in the same quarter."""
    params = scripted_params(p_fail_in_episode=0.6)
    seed = next(
        s
        for s in range(400)
        if simulate_lives(params, s, Scenario()).lives[CR].mode == "LINER_WEAR"
        and simulate_lives(params, s, Scenario()).failure_day[CR] < 82
        and simulate_lives(params, s, Scenario()).fails_in_horizon(CR)
    )
    plant = fresh(seed, p_fail_in_episode=0.6, approver_catch_p=0.0)
    fail_step = round(simulate_lives(params, seed, Scenario()).failure_day[CR] * 96) - 1
    advance(plant, {}, until=fail_step)
    s = plant.des.by_id[CR]
    assert s.op is Op.FAIL and plant.stats().unplanned_failures >= 1
    extra = s.fail_extra_q
    assert extra > 0  # 36-96 hours of repair work on top of the job
    f = fail_step
    work = 24 + extra  # a bearing job on the crusher plus the damage repair
    # the officer is not needed for a bearing job; the approval takes one quarter
    sched = {
        f + 1: [C(PlanCmd.RAISE_WO, CR, "BEARING_DEFECT")],
        f + 2: [C(Cmd.LOCK)],
        f + 3: [C(Cmd.PROVE)],
        f + 4: [C(Cmd.ISSUE)],
        f + 5: [C(Cmd.BEGIN)],
        f + 5 + work: [C(Cmd.COMPLETE)],
        f + 6 + work: [C(Cmd.CLOSE)],
        f + 7 + work: [C(Cmd.UNISO)],
        f + 8 + work: [C(Cmd.START)],
    }
    advance(plant, sched, until=f + 7 + work)
    st = plant.stats()
    assert (st.planned_jobs, st.wrong_mode_jobs) == (1, 1)
    before = st.unplanned_failures
    plant.step([C(Cmd.START)])  # q = f + 8 + work
    assert plant.stats().unplanned_failures == before + 1  # the same worn liner fails at once
    assert plant.des.by_id[CR].op is Op.FAIL and plant.truth.life_id[CR] == 0
