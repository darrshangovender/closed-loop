"""P3b: command semantics of the DES: refusals, permits, gas, crew, spares, approvals, derate."""

import numpy as np
import pytest
from scipy.stats import binomtest

from closed_loop.core.types import Cmd, Command, Fidelity, Op, Permit, PlanCmd
from closed_loop.plant import PlantParams, Scenario, make_plant
from closed_loop.plant.des import APPROVED, AWAITING
from closed_loop.plant.params import PARTS

from ._helpers import theta_of
from ._scripts import (
    CR,
    CV,
    P1,
    P2,
    C,
    advance,
    at,
    drive,
    fresh,
    isolate,
    merge,
    open_job,
    scripted_params,
    seed_with_mode,
)


class TestRefusal:
    def test_non_forced_commands_are_refused_at_the_stated_probability(self):
        for p in (0.05, 0.3):
            plant = fresh(p_refuse=p)
            n = refused = 0
            for _ in range(8640):
                refused += len(plant.step([C(Cmd.START, CV)]).refused)  # START on a RUN asset
                n += 1
            assert binomtest(refused, n, p).pvalue > 1e-3, (p, refused / n)

    def test_the_draw_is_per_asset_and_quarter_not_per_call(self):
        # the same command at the same quarter meets the same fate in two worlds
        a, b = fresh(p_refuse=0.5), fresh(p_refuse=0.5)
        fa = [bool(a.step([C(Cmd.START, CV)]).refused) for _ in range(300)]
        fb = [bool(b.step([C(Cmd.START, CV)]).refused) for _ in range(300)]
        assert fa == fb and 100 < sum(fa) < 200

    def test_a_exec_stop_and_exit_are_never_refused_in_regime_in(self):
        plant = fresh(1, Scenario.decode(np.ones(32), "in"), p_refuse=1.0)
        for _ in range(600):
            r = plant.step([C(Cmd.EXIT, P1), C(Cmd.STOP, P2), C(Cmd.START, CV)])
            assert [c.cmd for c in r.executed] == [Cmd.EXIT, Cmd.STOP]
            assert [c.cmd for c in r.refused] == [Cmd.START]

    def test_regime_out_may_refuse_forced_commands_at_the_scenario_probability(self):
        plant = fresh(1, Scenario.decode(np.ones(32), "out"), p_refuse=0.0)
        assert plant.des.r.forced_refusal_p == pytest.approx(0.3)
        n = refused = 0
        for _ in range(8000):
            refused += len(plant.step([C(Cmd.EXIT, P1)]).refused)
            n += 1
        assert binomtest(refused, n, 0.3).pvalue > 1e-3, refused / n

    def test_noop_is_never_refused_and_a_command_for_an_unknown_asset_always_is(self):
        plant = fresh(p_refuse=1.0)
        r = plant.step([C(Cmd.NOOP, P1), Command("XX-999", Cmd.STOP)])
        assert [c.cmd for c in r.executed] == [Cmd.NOOP] and len(r.refused) == 1


def run_to(plant, schedule, last_q):
    """Step quarters 0..last_q - 1 of ``schedule`` and return the next StepResult input quarter."""
    for q in range(last_q):
        plant.step(list(schedule.get(q, ())))


IMPOSSIBLE = [
    pytest.param(at(q0=[C(Cmd.STOP), C(Cmd.LOCK)]), C(Cmd.START), id="START-while-locked"),
    pytest.param(merge(isolate(0)), C(Cmd.START), id="START-while-isolated"),
    pytest.param({}, C(Cmd.LOCK), id="LOCK-a-running-asset"),
    pytest.param(at(q0=[C(Cmd.STOP)]), C(Cmd.PROVE), id="PROVE-before-LOCK"),
    pytest.param(at(q0=[C(Cmd.STOP), C(Cmd.LOCK)]), C(Cmd.ISSUE), id="ISSUE-before-PROVE"),
    pytest.param(at(q0=[C(Cmd.STOP), C(Cmd.LOCK)]), C(Cmd.TEST), id="TEST-outside-work"),
    pytest.param(merge(isolate(0)), C(Cmd.COMPLETE), id="COMPLETE-outside-work"),
    pytest.param(
        merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)])),
        C(Cmd.COMPLETE),
        id="COMPLETE-with-a-person-inside",
    ),
    pytest.param(merge(isolate(0), at(q4=[C(Cmd.BEGIN)])), C(Cmd.CLOSE), id="CLOSE-during-work"),
    pytest.param(merge(isolate(0)), C(Cmd.UNISO), id="UNISO-with-the-permit-open"),
    pytest.param({}, C(Cmd.UNISO), id="UNISO-a-running-asset"),
    pytest.param(
        at(q0=[C(Cmd.STOP)], q1=[C(Cmd.LOCK)], q2=[C(Cmd.PROVE)]),
        C(Cmd.BEGIN),
        id="BEGIN-isolated-without-a-permit",
    ),  # fmt: skip
    pytest.param({}, C(Cmd.ENTER, P1), id="ENTER-a-pump"),
    pytest.param({}, C(Cmd.DERATE, P1), id="DERATE-without-F1D"),
    pytest.param({}, C(PlanCmd.RAISE_WO, CR, "OVERHEATING"), id="RAISE_WO-unknown-mode"),
    pytest.param({}, C(PlanCmd.RAISE_WO, P1, "LINER_WEAR"), id="RAISE_WO-a-mode-the-class-lacks"),
    pytest.param({}, C(PlanCmd.RAISE_WO, CR, None), id="RAISE_WO-without-a-mode"),
    pytest.param({}, C(PlanCmd.ORDER, P1, "SYN-NOPE-0000"), id="ORDER-unknown-part"),
    pytest.param({}, C(PlanCmd.ORDER, P1, None), id="ORDER-without-a-part"),
    pytest.param(
        at(q0=[C(PlanCmd.RAISE_WO, P1, "BEARING_DEFECT")]),
        C(PlanCmd.RAISE_WO, P1, "IMPELLER_WEAR"),
        id="RAISE_WO-twice",
    ),
]


@pytest.mark.parametrize(("setup", "cmd"), IMPOSSIBLE)
def test_impossible_commands_are_refused_and_change_nothing(setup, cmd):
    plant = fresh(approver_catch_p=0.0)
    last = max(setup, default=-1) + 1
    run_to(plant, setup, last)
    before = plant.des.by_id[cmd.asset_id].op, plant.des.by_id[cmd.asset_id].permit
    r = plant.step([cmd])
    assert cmd in r.refused and not r.executed
    s = plant.des.by_id[cmd.asset_id]
    assert (s.op, s.permit) == before


class TestPermits:
    def test_issue_needs_proven_isolation_and_gives_a_permit_until_the_end_of_the_shift(self):
        plant = fresh()
        run_to(plant, isolate(0), 4)
        obs = plant.step([])  # q = 4
        assert obs.obs.assets[CR].permit is Permit.VALID
        tr = plant.trace()
        assert tr.permit_margin_min[CR][3] == (32 - 3) * 15  # issued at 3, valid to the shift end
        assert tr.permit_margin_min[CR][0] == 1.0e6  # no permit: the "large" margin

    def test_the_permit_lapses_into_reval_at_the_handover_and_the_margin_shows_minus_one(self):
        plant = fresh()
        drive(plant, isolate(0), until=34)
        tr = plant.trace()
        assert tr.permit_margin_min[CR][30] == 2 * 15
        assert tr.permit[CR][30] == 1 and tr.permit[CR][31] == 2  # VALID, then REVAL
        assert tr.permit_margin_min[CR][31] == -1.0

    def test_a_shorter_permit_limit_ends_the_permit_inside_the_shift(self):
        plant = fresh(permit_max_h=2.0)  # 8 quarters
        drive(plant, isolate(0), until=14)
        tr = plant.trace()
        # issued at 3, limit 11: spent at the end of quarter 10. It is NONE, not REVAL: the limit is
        # reached, so there is nothing left to revalidate (0003 A12.1; it used to become REVAL)
        assert tr.permit[CR][9] == 1 and tr.permit[CR][10] == 0

    def test_revalidation_succeeds_when_the_holder_is_present_and_fails_when_absent(self):
        for p, want in ((1.0, Permit.VALID), (0.0, Permit.REVAL)):
            plant = fresh(holder_present_p=p)
            drive(plant, isolate(0), until=31)
            r = plant.step([C(Cmd.REVAL)])  # q = 32
            assert not r.refused  # an absent holder makes the command ineffective, not refused
            assert r.obs.assets[CR].permit is want

    def test_holder_presence_follows_the_probability_per_attempt(self):
        k = 0
        n = 300
        for seed in range(n):
            plant = fresh(seed, holder_present_p=0.5)
            drive(plant, isolate(0), until=31)
            k += plant.step([C(Cmd.REVAL)]).obs.assets[CR].permit is Permit.VALID
        assert binomtest(k, n, 0.5).pvalue > 1e-3, k / n

    def test_the_holder_absence_bias_of_theta_lowers_presence_at_that_handover_only(self):
        # handover into shift 1 (q = 32): present with probability 1.0 - 0.6 = 0.4; shifts 0 and 2:
        # 1.0
        theta = theta_of(c20=0.0, c21=1.0, c22=0.0)
        k0 = k1 = 0
        n = 300
        for seed in range(n):
            plant = fresh(seed, Scenario.decode(theta, "in"))
            drive(plant, isolate(0), until=31)
            k1 += (
                plant.step([C(Cmd.REVAL)]).obs.assets[CR].permit is Permit.VALID
            )  # q = 32, shift 1
            plant2 = fresh(seed, Scenario.decode(theta, "in"))
            drive(plant2, merge(isolate(0)), until=31)
            # to q = 64, the handover into shift 2 (untouched coordinate 22). The permit issued at 3
            # is spent at its 12 hour limit (quarter 51), so it is closed and a new one issued
            # at 41: the handover at 64 then has a permit to revalidate (0003 A12.1).
            for q in range(32, 64):
                plant2.step([C(Cmd.CLOSE)] if q == 40 else [C(Cmd.ISSUE)] if q == 41 else [])
            k0 += plant2.step([C(Cmd.REVAL)]).obs.assets[CR].permit is Permit.VALID
        assert binomtest(k1, n, 0.4).pvalue > 1e-3, k1 / n
        assert k0 == n  # the other handovers keep the full presence probability


class TestGas:
    def test_a_passing_test_zeroes_the_age_which_then_grows_by_15_minutes_a_quarter(self):
        plant = fresh()
        sched = merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)]))
        drive(plant, sched, until=9)
        tr = plant.trace()
        assert tr.gas_age_min[CR][4] == 1.0e6  # never tested
        assert list(tr.gas_age_min[CR][5:10]) == [0, 15, 30, 45, 60]

    def test_leaving_the_chamber_makes_the_test_stale(self):
        plant = fresh()
        sched = merge(
            isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)], q7=[C(Cmd.EXIT)])
        )
        drive(plant, sched, until=9)
        tr = plant.trace()
        assert tr.gas_age_min[CR][6] == 15 and tr.gas_age_min[CR][7] == 1.0e6

    def test_the_view_caps_the_age_and_never_means_the_cap(self):
        plant = fresh()
        assert plant.initial_observation.assets[CR].gas_age_q == 99
        sched = merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)]))
        for q in range(8):
            r = plant.step(list(sched.get(q, ())))
        assert r.obs.assets[CR].gas_age_q == 3  # tested at 5, decision at 8

    def test_an_adversary_timed_excursion_waits_for_someone_to_be_inside(self):
        plant = fresh(1, Scenario.decode(theta_of(c23=0.001), "in"))  # about quarter 8
        assert plant.des.r.gas_adv_q[0] == 8
        sched = merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q12=[C(Cmd.TEST)],
                                      q13=[C(Cmd.ENTER)]))  # fmt: skip
        for q in range(16):
            plant.step(list(sched.get(q, ())))
        tr = plant.trace()
        assert not tr.alarm[CR][:13].any()  # nobody inside at the timed quarter: nothing happens
        assert tr.alarm[CR][13]  # the first occupied quarter at or after it


class TestCrewAndSpares:
    def prepared(self, **kw):
        plant = fresh(**kw)
        sched = merge(isolate(0, CR), isolate(0, CV))
        run_to(plant, sched, 4)
        return plant

    def test_two_fitters_can_work_two_assets_but_not_three(self):
        plant = self.prepared(crew_per_shift=2)
        r = plant.step([C(Cmd.BEGIN, CR), C(Cmd.BEGIN, CV)])
        assert not r.refused and r.obs.crew_free == 0

    def test_one_fitter_works_one_asset(self):
        plant = self.prepared(crew_per_shift=1)
        r = plant.step([C(Cmd.BEGIN, CR), C(Cmd.BEGIN, CV)])
        assert [c.asset_id for c in r.executed] == [CR] and [c.asset_id for c in r.refused] == [CV]
        assert r.obs.crew_free == 0
        # finishing the work frees the fitter
        r = plant.step([C(Cmd.COMPLETE, CR)])
        assert r.obs.crew_free == 1
        assert not plant.step([C(Cmd.BEGIN, CV)]).refused

    def test_an_inspection_ties_up_a_fitter_for_its_duration(self):
        plant = self.prepared(crew_per_shift=1)
        assert plant.step([C(PlanCmd.INSPECT, P1)]).obs.crew_free == 0
        r = plant.step([C(Cmd.BEGIN, CR)])
        assert r.refused  # the only fitter is busy inspecting
        for _ in range(3):
            r = plant.step([])
        assert r.obs.crew_free == 1  # inspection started at 4, takes 4 quarters

    def test_a_part_is_taken_when_work_begins_and_a_missing_part_stops_the_job(self):
        seed = seed_with_mode(CR, "BEARING_DEFECT")
        plant = fresh(seed, approver_catch_p=0.0)
        part = "SYN-BRG-5001"
        assert PARTS[part].stock == 1
        first = open_job("BEARING_DEFECT")
        advance(plant, first, until=9)
        assert plant.step([]).obs.assets[CR].spares[part] == 0  # taken at BEGIN, not at COMPLETE
        advance(plant, first, until=40)  # the first job runs out and renews the component
        assert plant.stats().planned_jobs == 1
        second = {99: [C(PlanCmd.RAISE_WO, CR, "BEARING_DEFECT")]}
        second.update(
            {q + 96: v for q, v in isolate(4).items()}
        )  # STOP, LOCK, PROVE, ISSUE at 100-103
        r = advance(plant, second, until=103)
        job = r.obs.assets[CR].job
        assert job is not None and job.status == APPROVED and not job.part_available
        r = plant.step([C(Cmd.BEGIN)])  # q = 104: there is no part
        assert r.refused and plant.des.by_id[CR].op is Op.ISO

    def test_an_order_arrives_after_its_lead_time_times_the_scenario_multiplier(self):
        for theta, mult in ((None, 1.0), (theta_of(c29=1.0), 2.0)):
            sc = Scenario() if theta is None else Scenario.decode(theta, "in")
            plant = fresh(1, sc)
            part = "SYN-BRG-2001"  # 21 days
            arrive = int(np.ceil(21 * mult * 96))
            r = plant.step([C(PlanCmd.ORDER, P1, part)])
            assert r.obs.assets[P1].spares[part] == 4
            r = plant.step([C(PlanCmd.ORDER, P1, part)])
            assert r.refused  # one outstanding order per part
            for q in range(2, arrive):
                r = plant.step([])
                assert r.obs.assets[P1].spares[part] == (5 if q + 1 >= arrive else 4), q
            assert plant.stats().orders == 1

    def test_the_two_pumps_share_one_stock(self):
        plant = fresh()
        a = plant.initial_observation.assets
        assert a[P1].spares == a[P2].spares


class TestApprovals:
    def approval_quarter(self, plant, asset, mode, raise_at=0, horizon=400):
        for _ in range(raise_at):
            plant.step([])
        r = plant.step([C(PlanCmd.RAISE_WO, asset, mode)])
        q = raise_at
        while r.obs.assets[asset].job is not None and r.obs.assets[asset].job.status == AWAITING:
            r = plant.step([])
            q += 1
            if q > horizon:
                raise AssertionError("never approved")
        return q + 1 if r.obs.assets[asset].job is not None else None

    def test_latency_is_log_normal_with_a_median_of_two_hours(self):
        latencies = []
        for seed in range(300):
            plant = make_plant(
                PlantParams(p_fail_in_episode=0.0, approver_catch_p=0.0, p_refuse=0.0)
            )
            plant.reset(seed, Scenario())
            latencies.append(self.approval_quarter(plant, P1, "BEARING_DEFECT") * 15 / 60)
        lat = np.array(latencies)
        assert 1.6 <= np.median(lat) <= 2.6, np.median(lat)  # 2 h, rounded up to a quarter
        # log-normal with sigma 0.6: the 90th percentile is about exp(0.77) = 2.2 times the median
        assert 1.5 < np.percentile(lat, 90) / np.median(lat) < 3.0

    def test_a_repair_that_needs_a_safety_officer_waits_for_the_day_shift(self):
        seed = seed_with_mode(CR, "LINER_WEAR")
        plant = fresh(seed, approver_catch_p=0.0)
        aq = self.approval_quarter(plant, CR, "LINER_WEAR", raise_at=66)  # shift 2, 23:30
        assert aq == 96  # the first quarter of the next day shift
        plant2 = fresh(seed, approver_catch_p=0.0)
        assert self.approval_quarter(plant2, P1, "BEARING_DEFECT", raise_at=66) == 67

    def test_the_delay_multiplier_of_theta_scales_the_latency(self):
        plant = fresh(1, Scenario.decode(theta_of(c27=1.0), "in"), approver_catch_p=0.0)
        assert plant.des.r.approver_delay_mult == 3.0
        assert self.approval_quarter(plant, P1, "BEARING_DEFECT") == 3  # 0.25 h x 3 = 3 quarters

    def test_a_target_shift_holds_the_job_back_after_approval(self):
        plant = fresh(1, approver_catch_p=0.0)
        plant.step([C(PlanCmd.RAISE_WO, P1, "BEARING_DEFECT:5")])
        for _ in range(3):
            r = plant.step([])
        assert r.obs.assets[P1].job.status == APPROVED and not r.obs.assets[P1].job.ready
        while r.obs.quarter < 159:
            r = plant.step([])
        assert not r.obs.assets[P1].job.ready
        r = plant.step([])
        assert r.obs.quarter == 160 and r.obs.assets[P1].job.ready  # shift 5 starts at 5 x 32

    def test_a_wrong_diagnosis_is_caught_at_the_catch_probability_and_a_right_one_never(self):
        wrong = right = 0
        n = 300
        for seed in range(n):
            plant = make_plant(
                PlantParams(p_fail_in_episode=0.0, approver_catch_p=0.5, p_refuse=0.0)
            )
            plant.reset(seed, Scenario())
            true_mode = plant.truth.life_mode[P1]
            other = "BEARING_DEFECT" if true_mode == "IMPELLER_WEAR" else "IMPELLER_WEAR"
            plant.step([C(PlanCmd.RAISE_WO, P1, other)])
            for _ in range(60):
                r = plant.step([])
            wrong += r.obs.assets[P1].job is None
            plant2 = make_plant(
                PlantParams(p_fail_in_episode=0.0, approver_catch_p=1.0, p_refuse=0.0)
            )
            plant2.reset(seed, Scenario())
            plant2.step([C(PlanCmd.RAISE_WO, P1, plant2.truth.life_mode[P1])])
            for _ in range(60):
                r2 = plant2.step([])
            right += r2.obs.assets[P1].job is None
        assert binomtest(wrong, n, 0.5).pvalue > 1e-3, wrong / n
        assert right == 0

    def test_a_rejected_work_order_costs_a_little_and_is_counted(self):
        plant = fresh(1, approver_catch_p=1.0)
        other = (
            "BEARING_DEFECT" if plant.truth.life_mode[P1] == "IMPELLER_WEAR" else "IMPELLER_WEAR"
        )
        plant.step([C(PlanCmd.RAISE_WO, P1, other)])
        for _ in range(20):
            plant.step([])
        st = plant.stats()
        assert (st.wo_rejected, st.planned_jobs) == (1, 0)
        assert st.cost == pytest.approx(plant.params.rejected_wo_cost)

    def test_theta_overrides_the_catch_probability(self):
        plant = fresh(1, Scenario.decode(theta_of(c28=1.0), "in"), approver_catch_p=0.0)
        assert plant.des._catch_p == 1.0


class TestInspection:
    def test_result_arrives_after_inspect_q_with_a_low_noise_reading_and_costs_a_little(self):
        p = scripted_params(hi_noise_sd=0.0)
        plant = make_plant(p)
        plant.reset(2, Scenario())
        r = plant.step([C(PlanCmd.INSPECT, P1)])
        assert r.obs.assets[P1].inspection is None
        for _ in range(3):
            r = plant.step([])
        res = r.obs.assets[P1].inspection
        assert res is not None and res.day == 0
        sev = plant.des.by_id[P1].clock.severity(4)
        want = (
            p.hi_base
            + (p.hi_fail - p.hi_base) * sev
            + p.hi_load_coef * (plant._phys.load_day[P1][0] - 1)
        )
        assert res.hi == pytest.approx(want) and not res.gain_fault
        assert plant.stats().inspections == 1 and plant.stats().cost == pytest.approx(
            p.inspect_cost
        )

    def test_an_inspection_sees_through_a_sensor_gain_drift(self):
        theta = theta_of(c13=0.9, c14=0.0)  # asset 2 (PU-201): fault type 3 from day 0
        sc = Scenario.decode(theta, "out")
        plant = fresh(1, sc, hi_noise_sd=0.0)
        for _ in range(300):  # day 3: 0.005 x 3 = 1.5% of full scale, over the 1% flag
            plant.step([])
        plant.step([C(PlanCmd.INSPECT, P1), C(PlanCmd.INSPECT, P2)])
        for _ in range(4):
            r = plant.step([])
        assert r.obs.assets[P1].inspection.gain_fault and not r.obs.assets[P2].inspection.gain_fault
        # the reading itself carries no drift: it is below the drifting transmitter's reading
        drifting = r.obs.assets[P1].last_obs.hi
        assert drifting is not None
        assert r.obs.assets[P1].inspection.hi < drifting

    def test_a_second_inspection_is_refused_while_one_is_under_way(self):
        plant = fresh()
        plant.step([C(PlanCmd.INSPECT, P1)])
        assert plant.step([C(PlanCmd.INSPECT, P1)]).refused


class TestDerate:
    def test_refused_in_f1_and_executed_in_f1d(self):
        f1 = fresh()
        assert f1.step([C(Cmd.DERATE, P1)]).refused
        plant = make_plant(scripted_params(), fidelity=Fidelity.F1D)
        plant.reset(1, Scenario())
        r = plant.step([C(Cmd.DERATE, P1)])
        assert not r.refused and r.obs.assets[P1].derated

    def test_a_derated_asset_wears_slower_and_the_load_it_reports_falls(self):
        params = scripted_params(p_fail_in_episode=0.6)
        seed = 5
        fast = make_plant(params, fidelity=Fidelity.F1D)
        fast.reset(seed, Scenario())
        slow = make_plant(params, fidelity=Fidelity.F1D)
        slow.reset(seed, Scenario())
        for q in range(200):
            fast.step([])
            slow.step([C(Cmd.DERATE, P1)] if q == 0 else [])
        # 30% less load slows the wear clock by exp(0.8 x 0.3) = 1.27, so the remaining life is
        # longer
        assert slow.truth.true_rul_days[P1] > 1.15 * fast.truth.true_rul_days[P1]
        la = slow._phys.derated(P1)[0][1]
        assert la == pytest.approx(slow._phys.load_day[P1][1] * 0.7, rel=1e-9)

    def test_derating_costs_throughput(self):
        plant = make_plant(scripted_params(), fidelity=Fidelity.F1D)
        plant.reset(1, Scenario())
        plant.step([])
        plant.step([C(Cmd.DERATE, P1)])
        tr = plant.trace()
        assert tr.throughput[0] == 1.0
        assert tr.throughput[1] == pytest.approx(1.0 - plant.params.derate_throughput_loss)


class TestThroughput:
    def test_accounting_by_run_state(self):
        plant = fresh()
        sched = {
            0: [C(Cmd.STOP, P1)],  # one pump: 0.6
            1: [C(Cmd.STOP, P2)],  # no pump: 0
            2: [C(Cmd.START, P1)],  # one pump again
            3: [C(Cmd.START, P2)],  # both
            4: [C(Cmd.STOP, CR)],  # no crusher: 0
            5: [C(Cmd.START, CR), C(Cmd.STOP, CV)],  # no conveyor: 0
            6: [C(Cmd.START, CV)],
        }
        drive(plant, sched, until=7)
        assert list(plant.trace().throughput) == pytest.approx([0.6, 0, 0.6, 1, 0, 0, 1, 1])

    def test_a_failed_asset_produces_nothing(self):
        plant = make_plant(scripted_params(p_fail_in_episode=1.0))
        plant.reset(3, Scenario())
        while True:
            r = plant.step(())
            if r.done:
                break
        tr = plant.trace()
        failed = [i for i in range(tr.n_q) if tr.op[CR][i] == list(Op).index(Op.FAIL)]
        assert failed and tr.throughput[failed[0]] == 0.0
