"""P3b: the plant's own checks of P1-P6, one scripted sequence per violation.

The done criterion: a complete correct crusher job runs with zero violations, and each of the wrong
sequences below produces exactly the expected violation, at the quarter it occurs and nowhere else.
Expected quarters are worked out by hand in the comments.
"""

import pytest

from closed_loop.core.types import Cmd, Op, PlanCmd
from closed_loop.plant.des import PUMP_PAIR_ID
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.scenario import Scenario

from ._scripts import (
    CR,
    P1,
    P2,
    C,
    at,
    confined_job,
    drive,
    fresh,
    isolate,
    merge,
    scripted_params,
    seed_with_mode,
)


class TestACorrectJobIsClean:
    def test_a_complete_liner_job_has_zero_violations_and_renews_the_component(self):
        seed = seed_with_mode(CR, "LINER_WEAR")
        plant = fresh(seed)
        assert plant.truth.life_mode[CR] == "LINER_WEAR"
        violations = drive(plant, confined_job("LINER_WEAR"), until=70)
        assert violations == []
        stats = plant.stats()
        assert (stats.planned_jobs, stats.wrong_mode_jobs, stats.wo_rejected) == (1, 0, 0)
        assert plant.truth.life_id[CR] == 1  # the right repair renewed the component
        tr = plant.trace()
        assert tr.op[CR][57] == list(Op).index(Op.RUN)  # restarted at 56
        # the job crossed a handover: the permit lapsed at the end of quarter 31 and was revalidated
        assert tr.permit[CR][31] == 2  # REVAL
        assert tr.permit[CR][32] == 1  # VALID again
        assert not tr.occupied[CR][31] and tr.occupied[CR][33] and tr.occupied[CR][51]
        assert not tr.occupied[CR][52]

    def test_the_job_spent_exactly_its_work_quarters_inside_the_chamber(self):
        plant = fresh(seed_with_mode(CR, "LINER_WEAR"))
        drive(plant, confined_job("LINER_WEAR"), until=60)
        inside = int(plant.trace().occupied[CR].sum())
        assert inside == 21 + 19  # entered at 10 and 33; out at 31 and 52

    def test_a_bearing_job_needs_no_entry_and_is_clean_too(self):
        from ._scripts import open_job

        plant = fresh(seed_with_mode(CR, "BEARING_DEFECT"))
        assert drive(plant, open_job("BEARING_DEFECT"), until=60) == []
        assert plant.truth.life_id[CR] == 1
        assert not plant.trace().occupied[CR].any()


def two_pumps_isolated():
    s = {0: [C(Cmd.STOP, P1), C(Cmd.STOP, P2)], 1: [C(Cmd.LOCK, P1), C(Cmd.LOCK, P2)]}
    s[2] = [C(Cmd.PROVE, P1)]
    s[3] = [C(Cmd.PROVE, P2)]
    return s


# (name, schedule, last quarter, expected violations).
# Quarters: a command issued at q is applied in step q, and a violation is recorded at that q.
WRONG_SEQUENCES = [
    pytest.param(
        "begin_before_prove",
        at(q0=[C(Cmd.STOP)], q1=[C(Cmd.LOCK)], q2=[C(Cmd.BEGIN)]),
        6,
        [("P1", CR, 2)],  # LOCKED, never proven, work begins
        id="P1-begin-before-prove",
    ),
    pytest.param(
        "begin_on_running_asset",
        at(q0=[C(Cmd.BEGIN)]),
        4,
        [("P1", CR, 0)],
        id="P1-begin-on-running-asset",
    ),
    pytest.param(
        "start_during_work",
        merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.START)])),
        8,
        [("P1", CR, 5)],  # energising a machine with work in progress, an event
        id="P1-start-during-work",
    ),
    pytest.param(
        "enter_without_any_gas_test",
        merge(isolate(0), at(q4=[C(Cmd.ENTER)])),
        10,
        [("P3", CR, 4)],  # no test ever: the age is far past 30 minutes at once
        id="P3-enter-untested",
    ),
    pytest.param(
        "enter_after_a_stale_test",
        merge(
            isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q7=[C(Cmd.ENTER)], q8=[C(Cmd.TEST)])
        ),
        9,
        [("P3", CR, 7)],  # test at 5, entry at 7: 30 minutes old, inside the window but not fresh
        id="P3-enter-after-stale-test",
    ),
    pytest.param(
        "stop_testing_while_inside",
        merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)])),
        9,
        [("P3", CR, 8)],  # last pass at 5: 15, 30 min at ends of 6 and 7, 45 min at the end of 8
        id="P3-test-lapses-inside",
    ),
    pytest.param(
        "stay_inside_through_handover",
        merge(
            isolate(0),
            at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)]),
            {q: [C(Cmd.TEST)] for q in range(7, 32, 2)},
        ),
        33,
        [("P2", CR, 31)],  # the permit lapses at the end of quarter 31 with a person still inside
        id="P2-inside-at-handover",
    ),
    pytest.param(
        "enter_with_revalidation_pending",
        merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q32=[C(Cmd.TEST)], q33=[C(Cmd.ENTER)])),
        34,
        [("P2", CR, 33)],  # permit lapsed at 31, nobody revalidated it, a person goes in at 33
        id="P2-enter-before-revalidation",
    ),
    pytest.param(
        "both_pumps_isolated",
        two_pumps_isolated(),
        6,
        [("P4", PUMP_PAIR_ID, 1)],  # both are LOCKED at quarter 1, and a locked pump is unavailable
        id="P4-both-pumps-isolated",
    ),
]


@pytest.mark.parametrize(("name", "schedule", "until", "expected"), WRONG_SEQUENCES)
def test_each_wrong_sequence_gives_exactly_its_violation(name, schedule, until, expected):
    plant = fresh()
    got = drive(plant, schedule, until)
    assert got == expected, name
    # the trace tells the same story: the plant's list is not a separate truth
    assert plant.trace().n_q == until + 1


class TestUncontrolledFailure:
    def test_p5_is_recorded_at_the_quarter_the_crusher_fails_running(self):
        params = scripted_params(p_fail_in_episode=0.6)
        seed = next(
            s for s in range(300) if simulate_lives(params, s, Scenario()).fails_in_horizon(CR)
        )
        table = simulate_lives(params, seed, Scenario())
        fail_q = round(table.failure_day[CR] * 96)  # first quarter whose start sees the failure
        plant = fresh(seed, p_fail_in_episode=0.6)
        got = []
        while True:
            r = plant.step(())
            got.extend(v for v in r.violations if v[0] == "P5")
            if r.done:
                break
        assert got == [("P5", CR, fail_q - 1)]
        tr = plant.trace()
        assert tr.uncontrolled_failure[CR].sum() == 1 and tr.uncontrolled_failure[CR][fail_q - 1]
        assert tr.op[CR][fail_q - 1] == list(Op).index(Op.FAIL)
        assert plant.stats().unplanned_failures >= 1

    def test_only_the_crusher_failing_is_a_p5_violation(self):
        params = scripted_params(p_fail_in_episode=0.6)
        seed = next(
            s
            for s in range(300)
            if not simulate_lives(params, s, Scenario()).fails_in_horizon(CR)
            and simulate_lives(params, s, Scenario()).fails_in_horizon("CV-101")
        )
        plant = fresh(seed, p_fail_in_episode=0.6)
        got = []
        while True:
            r = plant.step(())
            got.extend(r.violations)
            if r.done:
                break
        assert plant.stats().unplanned_failures >= 1
        assert [v for v in got if v[0] == "P5"] == []

    def test_a_crusher_stopped_before_failure_does_not_violate_p5(self):
        params = scripted_params(p_fail_in_episode=0.6)
        seed = next(
            s for s in range(300) if simulate_lives(params, s, Scenario()).fails_in_horizon(CR)
        )
        plant = fresh(seed, p_fail_in_episode=0.6)
        plant.step([C(Cmd.STOP)])  # a stopped crusher does not age and cannot fail running
        got = []
        for _ in range(8639):
            r = plant.step(())
            got.extend(r.violations)
        assert [v for v in got if v[0] == "P5"] == []
        tr = plant.trace()
        assert plant.truth.life_id[CR] == 0 and not tr.uncontrolled_failure[CR].any()
        assert (tr.op[CR] == list(Op).index(Op.STOP)).all()


class TestAlarm:
    def alarm_params(self):
        return {"gas_excursion_rate": 1.0}  # an excursion starts the first quarter anyone is inside

    def entry(self):
        return merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)]))

    def test_staying_inside_after_an_alarm_is_a_p6_violation_reported_for_the_alarm_quarter(self):
        plant = fresh(**self.alarm_params())
        got = drive(plant, self.entry(), until=7)
        assert got == [("P6", CR, 6)]  # alarm at 6, still inside at 7: the 15 minutes are used up
        tr = plant.trace()
        assert tr.alarm[CR][6] and tr.alarm[CR][7] and not tr.alarm[CR][5]

    def test_leaving_at_once_is_clean(self):
        plant = fresh(**self.alarm_params())
        sched = self.entry()
        for q in range(7):
            r = plant.step(list(sched.get(q, ())))
        # the proposer sees the alarm before it has to act
        assert r.obs.assets[CR].alarm and r.violations == ()
        got = []
        for q in range(7, 13):
            got += plant.step([C(Cmd.EXIT)] if q == 7 else []).violations  # forced, executed
        assert got == []

    def test_a_test_fails_while_an_excursion_is_active(self):
        plant = fresh(**self.alarm_params())
        sched = merge(self.entry(), at(q7=[C(Cmd.TEST)], q8=[C(Cmd.EXIT)]))
        drive(plant, sched, until=8)
        tr = plant.trace()
        # last pass was at 5, the test at 7 failed (the excursion lasts 4 quarters from 6), so the
        # gas age at the end of 7 is still counted from 5
        assert tr.gas_age_min[CR][7] == pytest.approx(15 * 2)

    def test_the_alarm_clears_after_the_excursion_and_a_test_passes_again(self):
        plant = fresh(**self.alarm_params())
        sched = merge(
            isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)], q7=[C(Cmd.EXIT)])
        )
        sched[14] = [C(Cmd.TEST)]
        drive(plant, sched, until=14)
        tr = plant.trace()
        assert tr.alarm[CR][6:11].all() and not tr.alarm[CR][11:].any()  # 4 quarters from 7
        assert tr.gas_age_min[CR][14] == 0.0


def test_forced_exit_and_stop_are_the_only_commands_that_survive_total_refusal():
    plant = fresh(p_refuse=1.0)
    r = plant.step([C(Cmd.STOP, P1), C(Cmd.START, P1), C(Cmd.EXIT, P1), C(Cmd.LOCK, P1)])
    assert [c.cmd for c in r.executed] == [Cmd.STOP, Cmd.EXIT]
    assert [c.cmd for c in r.refused] == [Cmd.START, Cmd.LOCK]


def test_violation_list_of_the_plant_matches_what_the_trace_holds_for_p1_to_p3():
    """The two views of one episode agree: each recorded violation shows in the trace."""
    plant = fresh()
    sched = merge(isolate(0), at(q4=[C(Cmd.BEGIN)], q5=[C(Cmd.TEST)], q6=[C(Cmd.ENTER)]))
    got = drive(plant, sched, until=9)  # the test lapses: P3 at 8
    tr = plant.trace()
    assert got == [("P3", CR, 8)]
    assert tr.occupied[CR][8] and tr.gas_age_min[CR][8] > 30
    assert not (tr.gas_age_min[CR][6:8] > 30).any()


def test_raise_wo_is_not_a_job_command_and_does_not_by_itself_violate_anything():
    plant = fresh()
    r = plant.step([C(PlanCmd.RAISE_WO, CR, "BEARING_DEFECT")])
    assert r.violations == () and not r.refused
