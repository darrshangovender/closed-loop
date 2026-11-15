"""P3c: the requirement adapter and the job runner."""

import importlib.util
import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from closed_loop.core.types import Cmd, Op, Permit, PlanCmd
from closed_loop.plant import PlantParams
from closed_loop.plant import jobs as jobs_module
from closed_loop.plant.des import APPROVED, AWAITING, COMPLETED, JobView
from closed_loop.plant.jobs import (
    ALPHABET,
    HIGH_RISK,
    PERMIT_TYPES,
    JobRequirement,
    JobRunner,
    agent_requirement,
    matrix_source,
    requirement,
    vendored_requirement,
    work_order_command,
)
from closed_loop.plant.params import (
    ASSET_CLASS,
    ASSET_IDS,
    CONFINED_JOBS,
    MODES_BY_CLASS,
    SAFETY_OFFICER_JOBS,
)
from closed_loop.plant.plant import AssetView

from ._scripts import (
    CR,
    CV,
    P1,
    P2,
    C,
    fresh,
    permit_runs,
    run_with_runner,
    scripted_params,
    seed_with_mode,
)

AGENT = importlib.util.find_spec("maintenance_agent") is not None
needs_agent = pytest.mark.skipif(
    not AGENT,
    reason="maintenance_agent is not installed: the vendored requirements matrix cannot be "
    "compared with requirement_for here (it is exercised on its own below)",
)

CLASSES = [
    "BELT_CONVEYOR",
    "SLURRY_PUMP",
    "VENT_FAN",
    "BALL_MILL",
    "CRUSHER",
    "HV_MOTOR",
    "NO_SUCH_CLASS",
]
MODES = [
    "BEARING_DEFECT", "GEAR_WEAR", "MISALIGNMENT", "IMBALANCE", "SEAL_LEAK", "IMPELLER_WEAR",
    "WINDING_INSULATION", "LINER_WEAR", "LUBE_DEGRADATION", "OVERHEATING", None,
]  # fmt: skip


class TestRequirementAdapter:
    def test_the_vendored_matrix_answers_the_known_cases(self):
        r = vendored_requirement("CRUSHER", "LINER_WEAR", False)
        assert r == JobRequirement(
            frozenset({"LOTO", "CONFINED_SPACE", "HOT_WORK", "WORK_AT_HEIGHT"}), "PRC-CRSH-031",
            "SAFETY_OFFICER", True,
        )  # fmt: skip
        r = vendored_requirement("SLURRY_PUMP", "IMPELLER_WEAR", False)
        assert (r.permits, r.work_procedure, r.role) == (
            frozenset({"LOTO"}),
            "PRC-PUMP-021",
            "PLANNER",
        )
        r = vendored_requirement(
            "CRUSHER", "LINER_WEAR", True
        )  # an inspection first drops the mode permits
        assert r.permits == frozenset({"LOTO"}) and r.work_procedure == "PRC-INSP-060" and r.known

    def test_unknown_classes_and_modes_fail_closed(self):
        for args in (("NO_SUCH_CLASS", "LINER_WEAR", False), ("CRUSHER", "OVERHEATING", False),
                     ("CRUSHER", None, False), ("NO_SUCH_CLASS", None, True)):  # fmt: skip
            r = vendored_requirement(*args)
            assert (
                not r.known and r.permits == frozenset(PERMIT_TYPES) and r.role == "SAFETY_OFFICER"
            )
            assert r.work_procedure is None

    def test_role_follows_the_high_risk_permits(self):
        for cls, mode in itertools.product(CLASSES, MODES):
            r = vendored_requirement(cls, mode, False)
            assert (r.role == "SAFETY_OFFICER") == bool(r.permits & HIGH_RISK)

    @needs_agent
    @pytest.mark.parametrize("inspection", [False, True])
    def test_it_equals_requirement_for_on_every_class_and_mode(self, inspection):
        for cls, mode in itertools.product(CLASSES, MODES):
            mine = vendored_requirement(cls, mode, inspection)
            real = agent_requirement(cls, mode, inspection)
            assert real is not None and mine == real, (cls, mode, inspection)

    @needs_agent
    def test_the_vendored_tables_are_the_real_tables(self):
        from maintenance_agent import requirements as real

        assert jobs_module._BASE_PERMITS == frozenset(p.value for p in real.BASE_PERMITS)
        assert jobs_module._CLASS_PERMITS == {
            k: frozenset(p.value for p in v) for k, v in real.CLASS_PERMITS.items()
        }
        assert jobs_module._MODE_PERMITS == {
            k: frozenset(p.value for p in v) for k, v in real.MODE_PERMITS.items()
        }
        assert jobs_module._REPAIR_PROCEDURE == dict(real.REPAIR_PROCEDURE)
        assert jobs_module._INSPECTION_PROCEDURE == dict(real.INSPECTION_PROCEDURE)
        assert set(PERMIT_TYPES) == {p.value for p in real.PermitType}
        assert HIGH_RISK == frozenset(p.value for p in real.HIGH_RISK_PERMITS)

    @needs_agent
    def test_requirement_uses_the_real_one_when_it_is_installed(self):
        assert matrix_source() == "maintenance_agent"
        assert requirement("CRUSHER", "LINER_WEAR") == agent_requirement(
            "CRUSHER", "LINER_WEAR", False
        )

    @needs_agent
    def test_for_every_class_and_mode_in_the_plant_the_job_equals_requirement_for(self):
        from maintenance_agent.requirements import requirement_for

        for cls, modes in MODES_BY_CLASS.items():
            for m in modes:
                real = requirement_for(cls, m.mode, False)
                mine = requirement(cls, m.mode, False)
                assert mine.permits == frozenset(p.value for p in real.permits)
                assert mine.role == real.role and mine.work_procedure == real.work_procedure
                assert mine.known and real.known

    def test_without_the_sibling_it_falls_back_to_the_vendored_table(self, monkeypatch):
        monkeypatch.setattr(jobs_module, "_agent_requirement_for", lambda: None)
        assert (
            matrix_source() == "vendored"
            and agent_requirement("CRUSHER", "LINER_WEAR", False) is None
        )
        assert requirement("CRUSHER", "LINER_WEAR") == vendored_requirement(
            "CRUSHER", "LINER_WEAR", False
        )

    def test_the_plants_confined_space_and_officer_tables_match_the_matrix(self):
        for cls, modes in MODES_BY_CLASS.items():
            for m in modes:
                r = requirement(cls, m.mode, False)
                assert r.known
                assert ((cls, m.mode) in CONFINED_JOBS) == ("CONFINED_SPACE" in r.permits), (
                    cls,
                    m.mode,
                )
                assert ((cls, m.mode) in SAFETY_OFFICER_JOBS) == (r.role == "SAFETY_OFFICER"), (
                    cls,
                    m.mode,
                )

    def test_the_plants_work_orders_carry_the_matrix_answer(self):
        for aid in ASSET_IDS:
            cls = ASSET_CLASS[aid]
            for m in MODES_BY_CLASS[cls]:
                plant = fresh(1, approver_catch_p=0.0)
                r = plant.step([work_order_command(aid, m.mode)])
                job = r.obs.assets[aid].job
                assert job is not None and job.mode == m.mode
                assert job.confined == ("CONFINED_SPACE" in requirement(cls, m.mode).permits)


class TestWorkOrderCommand:
    def test_arg_format(self):
        c = work_order_command("CR-501", "LINER_WEAR", source="x")
        assert (c.asset_id, c.cmd, c.arg, c.source) == (
            "CR-501",
            PlanCmd.RAISE_WO,
            "LINER_WEAR",
            "x",
        )
        assert work_order_command("CR-501", "LINER_WEAR", 7).arg == "LINER_WEAR:7"


def view(**kw):
    base: dict = dict(
        asset_id=CR, op=Op.RUN, permit=Permit.NONE, occupied=False, gas_age_q=99, alarm=False,
        soon=False, set_lo=0, set_hi=3, rul_lo=None, rul_hi=None, prior_only=False, hi_last=1.0,
        job_open=True, spares={},
    )  # fmt: skip
    base.update(kw)
    return AssetView(**base)


def job(**kw):
    base: dict = dict(
        asset_id=CR, wo_id="WO-1", mode="LINER_WEAR", status=APPROVED, ready=True, confined=True,
        work_total_q=40, work_done_q=0, part_no="SYN-LNR-5001", part_available=True,
    )  # fmt: skip
    base.update(kw)
    return JobView(**base)


class TestRunnerChoices:
    runner = JobRunner()

    def next(self, v, j, q=None):
        c = self.runner.next_command(v, j, quarter=q)
        return None if c is None else c.cmd

    @pytest.mark.parametrize(
        ("op", "permit", "want"),
        [
            (Op.RUN, Permit.NONE, Cmd.STOP),
            (Op.STOP, Permit.NONE, Cmd.LOCK),
            (Op.FAIL, Permit.NONE, Cmd.LOCK),
            (Op.LOCKED, Permit.NONE, Cmd.PROVE),
            (Op.ISO, Permit.NONE, Cmd.ISSUE),
            (Op.ISO, Permit.REVAL, Cmd.REVAL),
            (Op.ISO, Permit.VALID, Cmd.BEGIN),
        ],
    )
    def test_the_lockout_script(self, op, permit, want):
        assert self.next(view(op=op, permit=permit), job()) == want

    def test_waits_for_approval_the_target_shift_and_the_part(self):
        v = view(op=Op.ISO, permit=Permit.VALID)
        assert self.next(v, job(status=AWAITING, ready=False)) is None
        assert self.next(view(), job(ready=False)) is None  # approved, not yet its shift
        assert self.next(v, job(part_available=False)) is None
        assert self.next(v, None) is None and self.runner.next_command(view(job=None), None) is None

    def test_a_running_asset_is_not_taken_out_of_service_for_a_part_that_is_not_here(self):
        # 0003 A17: seed 1101 of the baselines sat isolated for 59 days waiting for a spare. The
        # runner waits with the asset running, and stops it the quarter the part is available
        assert self.next(view(op=Op.RUN), job(part_available=False)) is None
        assert self.next(view(op=Op.RUN), job(part_available=True)) == Cmd.STOP

    @pytest.mark.parametrize(
        ("op", "permit", "want"),
        [
            (Op.STOP, Permit.NONE, Cmd.LOCK),
            (Op.FAIL, Permit.NONE, Cmd.LOCK),
            (Op.LOCKED, Permit.NONE, Cmd.PROVE),
            (Op.ISO, Permit.NONE, Cmd.ISSUE),
            (Op.ISO, Permit.REVAL, Cmd.REVAL),
        ],
    )
    def test_an_asset_that_is_already_down_is_made_safe_part_or_no_part(self, op, permit, want):
        assert self.next(view(op=op, permit=permit), job(part_available=False)) == want

    def test_beginning_the_work_still_waits_for_the_part(self):
        v = view(op=Op.ISO, permit=Permit.VALID)
        assert self.next(v, job(part_available=False)) is None
        assert self.next(v, job(part_available=True)) == Cmd.BEGIN

    def test_the_confined_entry_sequence(self):
        w = dict(op=Op.WORK, permit=Permit.VALID)
        assert (
            self.next(view(**w, gas_age_q=99), job(), q=10) == Cmd.TEST
        )  # no fresh test: test first
        assert (
            self.next(view(**w, gas_age_q=1), job(), q=10) == Cmd.ENTER
        )  # tested last quarter: enter
        assert (
            self.next(view(**w, gas_age_q=2), job(), q=10) == Cmd.TEST
        )  # 30 minutes old: not fresh
        assert (
            self.next(view(**w, gas_age_q=1, alarm=True), job(), q=10) is None
        )  # wait the alarm out

    def test_inside_it_retests_exits_on_alarm_and_before_the_handover(self):
        w = dict(op=Op.WORK, permit=Permit.VALID, occupied=True)
        assert self.next(view(**w, gas_age_q=0), job(), q=10) is None  # working
        assert self.next(view(**w, gas_age_q=1), job(), q=10) == Cmd.TEST  # one quarter of margin
        assert self.next(view(**w, gas_age_q=0, alarm=True), job(), q=10) == Cmd.EXIT
        assert self.next(view(**w, gas_age_q=0, soon=True), job(), q=31) == Cmd.EXIT

    def test_it_does_not_start_an_entry_near_the_end_of_the_shift(self):
        w = dict(op=Op.WORK, permit=Permit.VALID, gas_age_q=1)
        assert self.next(view(**w), job(), q=27) == Cmd.ENTER  # 5 quarters left
        assert self.next(view(**w), job(), q=29) is None  # 3 left
        assert self.next(view(**w, soon=True), job(), q=31) is None
        assert (
            self.next(view(**w, soon=True), job(), q=None) is None
        )  # without the quarter: `soon` only

    def test_a_permit_spent_at_its_limit_is_replaced_in_work_when_nobody_is_inside(self):
        spent = view(op=Op.WORK, permit=Permit.NONE, gas_age_q=99)
        assert self.next(spent, job(work_done_q=21), q=40) == Cmd.ISSUE
        # not while someone is inside (ISSUE would be refused): the way out is the same as for any
        # permit that is no longer valid
        inside = view(op=Op.WORK, permit=Permit.NONE, occupied=True)
        assert self.next(inside, job(work_done_q=21), q=40) == Cmd.EXIT
        reval_inside = view(op=Op.WORK, permit=Permit.REVAL, occupied=True)
        assert self.next(reval_inside, job(work_done_q=21), q=40) == Cmd.EXIT
        # a job with no entry has no use for a permit mid-work, and a finished one just completes
        assert self.next(spent, job(mode="BEARING_DEFECT", confined=False), q=40) is None
        assert self.next(spent, job(work_done_q=40), q=40) == Cmd.COMPLETE

    def test_the_entry_window_is_the_shorter_of_the_shift_and_the_permit(self):
        w = dict(op=Op.WORK, permit=Permit.VALID, gas_age_q=1)
        # eight quarters left in the shift: only the permit decides
        assert self.next(view(**w, permit_limit_left_q=9), job(), q=24) == Cmd.ENTER
        assert self.next(view(**w, permit_limit_left_q=4), job(), q=24) == Cmd.ENTER
        assert self.next(view(**w, permit_limit_left_q=3), job(), q=24) is None
        # plenty on the permit: the shift decides, as before
        assert self.next(view(**w, permit_limit_left_q=40), job(), q=29) is None
        assert self.next(view(**w, permit_limit_left_q=40), job(), q=27) == Cmd.ENTER
        # no quarter given: the permit alone is judged (and `soon` as ever)
        assert self.next(view(**w, permit_limit_left_q=3), job(), q=None) is None
        assert self.next(view(**w, permit_limit_left_q=30), job(), q=None) == Cmd.ENTER
        assert self.next(view(**w), job(), q=None) == Cmd.ENTER  # no permit information at all

    def test_after_the_handover_it_revalidates_before_going_back_in(self):
        v = view(op=Op.WORK, permit=Permit.REVAL, gas_age_q=99)
        assert self.next(v, job(work_done_q=21), q=32) == Cmd.REVAL

    def test_it_leaves_completes_and_closes_out(self):
        done = job(work_done_q=40)
        assert self.next(view(op=Op.WORK, permit=Permit.VALID, occupied=True), done) == Cmd.EXIT
        assert self.next(view(op=Op.WORK, permit=Permit.VALID), done) == Cmd.COMPLETE
        closing = job(status=COMPLETED, ready=False)
        assert self.next(view(op=Op.ISO, permit=Permit.VALID), closing) == Cmd.CLOSE
        assert self.next(view(op=Op.ISO, permit=Permit.REVAL), closing) == Cmd.CLOSE
        assert self.next(view(op=Op.ISO, permit=Permit.NONE), closing) == Cmd.UNISO
        assert self.next(view(op=Op.STOP), closing) == Cmd.START
        assert self.next(view(op=Op.RUN), closing) is None

    def test_a_job_without_entry_just_waits_while_the_work_is_done(self):
        crusher_bearing = job(mode="BEARING_DEFECT", confined=False, work_total_q=24)
        v = view(op=Op.WORK, permit=Permit.VALID)
        assert self.next(v, crusher_bearing) is None
        assert (
            self.next(
                v, job(mode="BEARING_DEFECT", confined=False, work_total_q=24, work_done_q=24)
            )
            == Cmd.COMPLETE
        )

    def test_parameters_are_checked(self):
        for bad in ({"retest_q": 0}, {"break_q": -1}, {"min_entry_window_q": 1}):
            with pytest.raises(ValueError):
                JobRunner(**bad)

    def test_a_team_on_the_old_procedure_tests_every_eight_quarters(self):
        old = JobRunner(retest_q=8, test_margin_q=0)
        inside = dict(op=Op.WORK, permit=Permit.VALID, occupied=True)
        assert old.next_command(view(**inside, gas_age_q=7), job(), quarter=10) is None
        c = old.next_command(view(**inside, gas_age_q=8), job(), quarter=10)
        assert c is not None and c.cmd == Cmd.TEST
        # the current procedure would have tested long before
        assert JobRunner().next_command(view(**inside, gas_age_q=2), job(), quarter=10) is not None


@settings(max_examples=400)
@given(
    op=st.sampled_from(list(Op)),
    permit=st.sampled_from(list(Permit)),
    occupied=st.booleans(),
    gas_age=st.integers(0, 99),
    alarm=st.booleans(),
    soon=st.booleans(),
    status=st.sampled_from([AWAITING, APPROVED, COMPLETED]),
    ready=st.booleans(),
    done=st.integers(0, 40),
    avail=st.booleans(),
    confined=st.booleans(),
    q=st.one_of(st.none(), st.integers(0, 200)),
)
def test_the_runner_never_emits_a_command_whose_precondition_does_not_hold(
    op, permit, occupied, gas_age, alarm, soon, status, ready, done, avail, confined, q
):
    """Whatever it is shown, the runner asks only for commands that are in order for that state."""
    mode = "LINER_WEAR" if confined else "BEARING_DEFECT"
    v = view(op=op, permit=permit, occupied=occupied, gas_age_q=gas_age, alarm=alarm, soon=soon)
    j = job(
        status=status,
        ready=ready,
        work_done_q=done,
        part_available=avail,
        mode=mode,
        confined=confined,
    )
    c = JobRunner().next_command(v, j, quarter=q)
    if c is None:
        return
    assert c.cmd in ALPHABET and c.asset_id == CR
    assert status != AWAITING
    cmd = c.cmd
    if status == COMPLETED:
        assert cmd in (Cmd.CLOSE, Cmd.UNISO, Cmd.START)
    else:
        assert status == APPROVED and ready
    if cmd == Cmd.STOP:
        assert op is Op.RUN
    if cmd == Cmd.LOCK:
        assert op in (Op.STOP, Op.FAIL)
    if cmd == Cmd.PROVE:
        assert op is Op.LOCKED
    if cmd == Cmd.ISSUE:  # in WORK only with nobody inside: a spent permit is replaced (A12.1)
        assert permit is Permit.NONE and (op is Op.ISO or (op is Op.WORK and not occupied))
    if cmd == Cmd.REVAL:
        assert permit is Permit.REVAL
    if cmd == Cmd.BEGIN:
        assert op is Op.ISO and permit is Permit.VALID and avail
    if cmd == Cmd.TEST:
        assert op is Op.WORK and done < 40 and confined
    if cmd == Cmd.ENTER:
        assert op is Op.WORK and permit is Permit.VALID and not occupied and not alarm and not soon
        assert gas_age <= 1 and confined and done < 40
    if cmd == Cmd.EXIT:
        assert occupied and op is Op.WORK
    if cmd == Cmd.COMPLETE:
        assert op is Op.WORK and not occupied and done >= 40
    if cmd == Cmd.CLOSE:
        assert op is Op.ISO and permit is not Permit.NONE and status == COMPLETED
    if cmd == Cmd.UNISO:
        assert op is Op.ISO and permit is Permit.NONE and status == COMPLETED
    if cmd == Cmd.START:
        assert op is Op.STOP and status == COMPLETED


# ------------------------------------------------------------------ the runner against the plant
def wo_for(plant, aid):
    return work_order_command(aid, plant.truth.life_mode[aid])


class TestRunnerAgainstThePlant:
    def test_each_asset_gets_its_correct_job_done_with_no_violation_and_a_renewal(self):
        for aid in (CR, CV, P1):
            seed = 3
            plant = fresh(seed, approver_catch_p=0.0)
            violations, history = run_with_runner(plant, {2: [wo_for(plant, aid)]}, until=900)
            assert violations == [], (aid, violations)
            st_ = plant.stats()
            assert (st_.planned_jobs, st_.wrong_mode_jobs) == (1, 0), aid
            assert plant.truth.life_id[aid] == 1 and plant.des.by_id[aid].op is Op.RUN
            order = [c.cmd for _, c, ok in history if ok]
            for a, b in zip(
                (Cmd.STOP, Cmd.LOCK, Cmd.PROVE, Cmd.ISSUE, Cmd.BEGIN), order, strict=False
            ):
                assert a == b
            closing = [Cmd.COMPLETE, Cmd.CLOSE, Cmd.UNISO, Cmd.START]
            if aid == CR:
                # the liner job is still in hand 12 hours after the permit was issued (6 + 48 = 54):
                # the permit is spent when the job is complete, so there is nothing to CLOSE (A12.1)
                assert plant.trace().permit[CR][history[-2][0] - 1] == 0  # NONE when UNISO is sent
                closing.remove(Cmd.CLOSE)
            assert order[-len(closing) :] == closing, aid

    @pytest.mark.parametrize("hours", [6.0, 4.0, 3.0])
    def test_a_liner_job_longer_than_the_permit_gets_new_permits_and_no_violation(self, hours):
        """A12.1: the permit is spent 12 hours (here fewer) after its issue, a handover does not
        renew it. The crew leaves on `soon`, issues a new permit in WORK, and goes back in."""
        seed = seed_with_mode(CR, "LINER_WEAR")
        plant = fresh(seed, approver_catch_p=0.0, permit_max_h=hours)
        violations, history = run_with_runner(plant, {2: [wo_for(plant, CR)]}, until=900)
        assert violations == []
        cmds = [c.cmd for _, c, ok in history if ok]
        assert plant.stats().planned_jobs == 1 and plant.truth.life_id[CR] == 1
        assert cmds.count(Cmd.ISSUE) >= 3  # the first and at least two replacements in WORK
        assert cmds.count(Cmd.ENTER) == cmds.count(Cmd.EXIT) >= 3
        runs = [n for _s, n in permit_runs(plant.trace().permit[CR])]
        assert max(runs) < int(hours * 4)  # no permit reaches its limit with anyone relying on it

    def test_the_liner_job_crosses_a_handover_and_uses_the_whole_entry_protocol(self):
        seed = seed_with_mode(CR, "LINER_WEAR")
        plant = fresh(seed, approver_catch_p=0.0)
        violations, history = run_with_runner(plant, {2: [wo_for(plant, CR)]}, until=900)
        assert violations == []
        cmds = [c.cmd for _, c, ok in history if ok]
        assert (
            cmds.count(Cmd.ENTER) >= 2 and cmds.count(Cmd.EXIT) >= 2
        )  # out for the handover, back in
        assert Cmd.REVAL in cmds
        i_reval = cmds.index(Cmd.REVAL)
        assert Cmd.EXIT in cmds[:i_reval]  # it left before the permit lapsed
        # a TEST immediately precedes every ENTER (REVAL may sit in between)
        for i, c in enumerate(cmds):
            if c == Cmd.ENTER:
                assert Cmd.TEST in cmds[max(0, i - 2) : i]
        assert int(plant.trace().occupied[CR].sum()) == 40  # exactly the work, no quarter more

    def plan(self, plant):
        return {
            2: [wo_for(plant, CR)],
            300: [wo_for(plant, P1)],
            1200: [wo_for(plant, CV)],
            2400: [wo_for(plant, P2)],
        }

    def test_with_no_random_refusals_the_runner_makes_no_violation_over_many_episodes(self):
        """Approval delays, holders (90%) and gas excursions (1% per occupied quarter)."""
        bad = done = 0
        for seed in range(80):
            plant = fresh(
                seed,
                p_refuse=0.0,
                holder_present_p=0.9,
                gas_excursion_rate=0.01,
                approver_median_h=2.0,
                approver_sigma=0.6,
                approver_catch_p=0.0,
            )
            violations, _ = run_with_runner(plant, self.plan(plant), until=4800)
            bad += len(violations)
            done += plant.stats().planned_jobs
        assert bad == 0
        assert done == 80 * 4  # every job ran to completion

    def test_with_five_percent_refusals_every_violation_is_a_run_of_refused_tests(self):
        """Refused tests let the 30-minute window lapse: P3, and only that.

        The runner re-tests every quarter, so inside the chamber three refusals in a row are needed.
        Right after an entry two are enough, because the entry quarter itself carries no test.
        """
        explained = unexplained = jobs = raised = 0
        for seed in range(80):
            plant = fresh(
                seed,
                p_refuse=0.05,
                holder_present_p=0.9,
                gas_excursion_rate=0.01,
                approver_median_h=2.0,
                approver_sigma=0.6,
                approver_catch_p=0.0,
            )
            violations, history = run_with_runner(plant, self.plan(plant), until=4800)
            jobs += plant.stats().planned_jobs
            raised += sum(s.n_wo for s in plant.des.states)  # a RAISE_WO can itself be refused
            refused_tests = {q for q, c, ok in history if c.cmd == Cmd.TEST and not ok}
            entries = {q for q, c, ok in history if c.cmd == Cmd.ENTER and ok}
            for prop, _aid, q in violations:
                run = {q - 1, q} <= refused_tests and (q - 2 in refused_tests or q - 2 in entries)
                if prop == "P3" and run:
                    explained += 1
                else:
                    unexplained += 1
        assert unexplained == 0, (explained, unexplained)
        assert (
            raised > 80 * 4 * 0.9 and jobs == raised
        )  # every accepted work order ran to completion

    def test_after_a_refusal_the_runner_simply_asks_again(self):
        plant = fresh(3, p_refuse=0.5, approver_catch_p=0.0)
        # the work order itself may be refused, so it is raised again each quarter until accepted
        plan = {q: [wo_for(plant, P1)] for q in range(2, 24)}
        violations, history = run_with_runner(plant, plan, until=1500)
        refused = [c.cmd for _, c, ok in history if not ok]
        assert refused, "with 50% refusals some commands must have been refused"
        assert plant.stats().planned_jobs == 1 and violations == []
        # the same command was re-issued at the next quarter after each refusal
        by_q = {q: c.cmd for q, c, _ in history}
        for q, c, ok in history:
            if not ok:
                # a permit that lapsed at the handover turns the next step into a revalidation
                assert by_q.get(q + 1) in (c.cmd, Cmd.REVAL), (q, c.cmd)

    def test_a_shield_override_is_just_another_refusal(self):
        """A command replaced by the shield at quarter q does not stop the job: it is re-planned."""
        plant = fresh(3, approver_catch_p=0.0)
        runner = JobRunner()
        obs = plant.step([wo_for(plant, P1)]).obs
        replaced = 0
        for _ in range(900):
            c = runner.next_command(obs.assets[P1], obs.assets[P1].job, quarter=obs.quarter)
            cmds = []
            if c is not None and c.cmd == Cmd.PROVE and replaced < 3:
                replaced += 1  # the shield turns the first three PROVEs into nothing
            elif c is not None:
                cmds = [c]
            obs = plant.step(cmds).obs
        assert replaced == 3 and plant.stats().planned_jobs == 1

    def test_a_runner_on_the_superseded_two_hour_retest_breaches_the_rev_3_clause(self):
        seed = seed_with_mode(CR, "LINER_WEAR")
        plant = fresh(seed, approver_catch_p=0.0)
        old = JobRunner(retest_q=8, test_margin_q=0)
        violations, _ = run_with_runner(plant, {2: [wo_for(plant, CR)]}, runner=old, until=900)
        assert violations and {v[0] for v in violations} == {"P3"}

    def test_the_runner_leaves_a_wrong_mode_job_on_its_own_terms(self):
        """It carries out the work order it was given, right or wrong: diagnosis is not its job."""
        seed = seed_with_mode(CR, "LINER_WEAR")
        plant = fresh(seed, approver_catch_p=0.0)
        wrong = work_order_command(CR, "BEARING_DEFECT")
        violations, _ = run_with_runner(plant, {2: [wrong]}, until=600)
        assert violations == [] and plant.stats().wrong_mode_jobs == 1
        assert plant.truth.life_id[CR] == 0

    def test_alphabet_is_the_job_commands_only(self):
        assert Cmd.NOOP not in ALPHABET and Cmd.DERATE not in ALPHABET
        assert len(ALPHABET) == 13


def test_scripted_params_still_means_what_the_runner_tests_assume():
    p = scripted_params()
    assert p.p_refuse == 0.0 and isinstance(p, PlantParams)
    assert C(Cmd.STOP).cmd == Cmd.STOP
