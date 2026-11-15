"""P11b: the simulated human approver, driving the real ApprovalGate. Nothing on the agent side
is mocked."""

from __future__ import annotations

import ast
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import closed_loop.f2 as f2_package
from closed_loop.f2.human import DAY_SHIFT, on_day_shift
from closed_loop.f2.render import sim_time

pytestmark = pytest.mark.f2


def drive(world, human, run_id: str, hours: int = 48):
    state = world.orch.get(run_id)
    for _ in range(hours):
        state = human.act(world.orch, run_id)
        if state.status.value not in ("AWAITING_APPROVAL", "BLOCKED"):
            break
        world.advance(hours=1)
    return state


def quick(params):
    """An approver who answers within minutes, so that a test controls the date of the signature."""
    return replace(params, approver_median_h=0.05, approver_sigma=0.01)


def waiting_state(i: int):
    from maintenance_agent.state import RunState, RunStatus

    return RunState(
        run_id=f"run-{i}", alert_id="a", planner="rules", status=RunStatus.AWAITING_APPROVAL
    )


class TestAttention:
    def test_latency_is_log_normal_with_the_plants_median_and_spread(
        self, fixture_world, params
    ) -> None:
        human = fixture_world.human(rng=np.random.default_rng(11))
        for i in range(800):
            assert human.review(waiting_state(i)) is None  # first sight: the latency starts
        hours = np.array([p.latency_h for p in human._pending.values()])
        assert np.median(hours) == pytest.approx(params.approver_median_h, rel=0.1)
        assert np.std(np.log(hours)) == pytest.approx(params.approver_sigma, rel=0.1)

    def test_a_run_costs_two_draws_however_often_it_is_looked_at(self, fixture_world) -> None:
        rng = np.random.default_rng(3)
        human = fixture_world.human(rng=rng)
        for _ in range(5):
            human.review(waiting_state(0))
        reference = np.random.default_rng(3)
        reference.lognormal(0.0, 1.0)
        reference.random()
        assert rng.random() == reference.random()

    def test_the_same_stream_gives_the_same_latency(self, fixture_world) -> None:
        a = fixture_world.human(rng=np.random.default_rng(5))
        b = fixture_world.human(rng=np.random.default_rng(5))
        a.review(waiting_state(0))
        b.review(waiting_state(0))
        assert a._pending["run-0"].latency_h == b._pending["run-0"].latency_h

    def test_nothing_to_review_when_the_run_is_not_waiting(self, fixture_world) -> None:
        from maintenance_agent.state import RunState, RunStatus

        done = RunState(run_id="x", alert_id="a", planner="rules", status=RunStatus.COMPLETED)
        assert fixture_world.human().review(done) is None

    def test_it_signs_only_after_the_latency_and_in_simulation_time(self, fixture_world) -> None:
        world = fixture_world
        state = world.orch.run("ALR-1001", run_id="r")
        human = world.human()
        assert human.review(state) is None
        ready = human._pending["r"].ready_at
        world.now = ready.replace(tzinfo=None) - timedelta(minutes=1)
        assert human.review(state) is None
        world.now = ready.replace(tzinfo=None) + timedelta(minutes=1)
        state = world.orch.refresh_verification("r")
        approval = human.review(state)
        assert approval is not None
        assert approval.issued_at == world.clock()
        assert approval.expires_at - approval.issued_at == timedelta(hours=8)
        assert approval.approver_id == "planner1" and approval.draft_hash == state.draft_hash
        assert approval.verification_digest == state.verification.digest

    def test_the_approval_expires_on_the_simulation_clock(self, fixture_world, params) -> None:
        """The sibling's own expiry check, run on simulated time: nine hours later it is stale."""
        world = fixture_world
        world.now = sim_time(39.0)  # 06:00, so nine hours later is still the same date
        state = world.orch.run("ALR-1001", run_id="r")
        human = world.human(params=quick(params))
        human.review(state)
        world.now = human._pending["r"].ready_at.replace(tzinfo=None) + timedelta(minutes=1)
        state = world.orch.refresh_verification("r")
        approval = human.review(state)
        world.advance(hours=9)
        state = world.orch.submit_approval("r", approval)
        refusals = [r for r in world.orch.audit("r").records() if r["type"] == "release_refused"]
        assert state.status.value == "AWAITING_APPROVAL"
        assert [r["code"] for r in refusals] == ["EXPIRED"] and world.queued == []

    def test_a_date_change_between_signing_and_release_voids_the_approval(
        self, fixture_world, params
    ) -> None:
        """The verification digest covers the day of the check; an approval does not outlive it."""
        world = fixture_world  # 18:00: six hours to midnight
        state = world.orch.run("ALR-1001", run_id="r")
        human = world.human(params=quick(params))
        human.review(state)
        world.now = human._pending["r"].ready_at.replace(tzinfo=None) + timedelta(minutes=1)
        state = world.orch.refresh_verification("r")
        approval = human.review(state)
        world.advance(hours=7)  # past midnight
        state = world.orch.submit_approval("r", approval)
        voided = [r for r in world.orch.audit("r").records() if r["type"] == "approval_voided"]
        assert state.status.value == "AWAITING_APPROVAL" and len(voided) == 1


class TestVerdicts:
    def test_a_blocked_draft_is_rejected_with_the_failed_rules(self, fixture_world) -> None:
        world = fixture_world
        world.advance(days=8)
        state = world.orch.run("ALR-1001", run_id="r")
        assert state.status.value == "BLOCKED"
        human = world.human()
        human.review(state)
        world.now = human._pending["r"].ready_at.replace(tzinfo=None) + timedelta(minutes=1)
        decision = human.review(state)
        assert decision.kind == "reject" and decision.approver_id == "planner1"
        assert human.reasons["r"] == "verification failed: V10"

    def test_the_gate_itself_refuses_to_sign_a_blocked_draft(self, fixture_world) -> None:
        from maintenance_agent.gate import GateRefused
        from maintenance_agent.models import WorkOrderDraft

        world = fixture_world
        world.advance(days=8)
        state = world.orch.run("ALR-1001", run_id="r")
        human = world.human()
        with pytest.raises(GateRefused, match="verification did not pass"):
            human._gate.approve(
                run_id="r",
                draft=WorkOrderDraft.model_validate(state.draft),
                report=state.verification,
            )

    def test_rejecting_closes_the_run_and_releases_nothing(self, fixture_world) -> None:
        world = fixture_world
        world.advance(days=8)
        world.orch.run("ALR-1001", run_id="r")
        state = drive(world, world.human(), "r")
        assert (
            state.status.value == "REJECTED" and world.queued == [] and world.mock.released() == []
        )

    def test_a_correct_draft_is_signed_and_released(self, fixture_world) -> None:
        world = fixture_world
        world.orch.run("ALR-1001", run_id="r")
        human = world.human(truth_mode=lambda asset: "BEARING_DEFECT")
        state = drive(world, human, "r")
        assert state.status.value == "COMPLETED" and len(world.queued) == 1
        entry = human.log[-1]
        assert (entry.kind, entry.wrong_mode, entry.caught) == ("approve", False, False)
        assert entry.top_mode == "BEARING_DEFECT" and entry.true_mode == "BEARING_DEFECT"


class TestWrongDiagnosis:
    """maintenance-agent's documented gap ``wrong_diagnosis_fully_compliant`` (0003 A3)."""

    def test_a_wrong_mode_draft_passes_all_sixteen_rules_and_reaches_a_signature(
        self, fixture_world, params
    ) -> None:
        world = fixture_world
        state = world.orch.run("ALR-1001", run_id="r")
        assert state.verification.passed and len(state.verification.rules_run) == 16
        human = world.human(
            params=replace(params, approver_catch_p=0.0), truth_mode=lambda asset: "GEAR_WEAR"
        )
        state = drive(world, human, "r")
        entry = human.log[-1]
        assert state.status.value == "COMPLETED"
        assert (entry.kind, entry.wrong_mode, entry.caught) == ("approve", True, False)
        assert entry.top_mode == "BEARING_DEFECT" and entry.true_mode == "GEAR_WEAR"
        assert len(world.queued) == 1  # the plant is sent to repair the wrong component

    def test_the_human_catches_it_with_the_stated_probability(self, fixture_world, params) -> None:
        world = fixture_world
        world.orch.run("ALR-1001", run_id="r")
        human = world.human(
            params=replace(params, approver_catch_p=1.0), truth_mode=lambda asset: "GEAR_WEAR"
        )
        state = drive(world, human, "r")
        entry = human.log[-1]
        assert state.status.value == "REJECTED" and world.queued == []
        assert (entry.kind, entry.wrong_mode, entry.caught) == ("reject", True, True)

    def test_the_catch_rate_over_many_runs_is_the_parameter(self, fixture_world, params) -> None:
        human = fixture_world.human(
            params=replace(params, approver_catch_p=0.5), rng=np.random.default_rng(9)
        )
        for i in range(2000):
            human.review(waiting_state(i))
        caught = np.mean([p.u_catch < 0.5 for p in human._pending.values()])
        assert caught == pytest.approx(0.5, abs=0.03)

    def test_an_inspection_first_draft_is_never_a_wrong_diagnosis(self, pump_world, params) -> None:
        world = pump_world
        alert_id = world.source.raise_alert("PU-201").alert_id
        world.orch.run(alert_id, run_id="r")
        human = world.human(
            params=replace(params, approver_catch_p=1.0), truth_mode=lambda asset: "BEARING_DEFECT"
        )
        state = drive(world, human, "r")
        assert state.status.value == "COMPLETED"
        assert human.log[-1].inspection_first and not human.log[-1].wrong_mode

    def test_with_no_truth_the_draft_is_signed(self, fixture_world) -> None:
        world = fixture_world
        world.orch.run("ALR-1001", run_id="r")
        assert drive(world, world.human(), "r").status.value == "COMPLETED"


class TestRoles:
    def test_the_day_shift_is_06_to_14(self) -> None:
        assert DAY_SHIFT == (6, 14)
        day = datetime(2026, 10, 5)
        assert [on_day_shift(day.replace(hour=h)) for h in (5, 6, 13, 14, 22)] == [
            False,
            True,
            True,
            False,
            False,
        ]

    def test_a_planner_leaves_a_confined_space_job_for_the_safety_officer(
        self, fixture_world
    ) -> None:
        world = fixture_world
        world.orch.run("ALR-1011", run_id="liner")
        planner = world.human()
        state = drive(world, planner, "liner", hours=30)
        assert state.status.value == "AWAITING_APPROVAL" and planner.deferred_role > 0
        assert planner.log == [] and world.queued == []

    def test_the_safety_officer_signs_on_the_day_shift_only(self, fixture_world) -> None:
        world = fixture_world  # 18:00: the officer is not at work
        world.orch.run("ALR-1011", run_id="liner")
        officer = world.human("officer1", roles=("SAFETY_OFFICER",), on_shift=on_day_shift)
        for _ in range(12):  # until 06:00 the next day
            state = officer.act(world.orch, "liner")
            assert state.status.value == "AWAITING_APPROVAL"
            world.advance(hours=1)
        assert officer.deferred_shift > 0 and officer.log == []
        state = drive(world, officer, "liner", hours=8)
        assert state.status.value == "COMPLETED"
        assert (
            on_day_shift(officer.log[-1].at) and officer.log[-1].required_role == "SAFETY_OFFICER"
        )
        assert state.release.approver_id == "officer1" and len(world.queued) == 1

    def test_the_role_order_is_the_siblings(self, fixture_world) -> None:
        from maintenance_agent.models import ROLE_RANK

        officer = fixture_world.human("officer1", roles=("SAFETY_OFFICER",))
        planner = fixture_world.human()
        assert officer._rank == ROLE_RANK["SAFETY_OFFICER"] > planner._rank == ROLE_RANK["PLANNER"]

    def test_a_planner_who_signs_anyway_is_refused_once_and_does_not_retry(
        self, fixture_world
    ) -> None:
        world = fixture_world
        world.orch.run("ALR-1011", run_id="liner")
        human = world.human(signs_above_rank=True)
        state = drive(world, human, "liner")
        refusals = [
            r for r in world.orch.audit("liner").records() if r["type"] == "release_refused"
        ]
        assert [r["code"] for r in refusals] == ["ROLE_INSUFFICIENT"]
        assert state.status.value == "AWAITING_APPROVAL" and len(human.log) == 1


class TestSeparation:
    def test_only_the_human_side_modules_touch_the_signing_half(self) -> None:
        """Mismatch 16: the sibling forbids agent-side modules from importing ``gate``. Here only
        ``human.py`` (the approver) and ``agent.py`` (key registration) name it."""
        root = Path(f2_package.__file__).parent
        naming = []
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and node.value == "maintenance_agent.gate":
                    naming.append(path.name)
        assert sorted(set(naming)) == ["agent.py", "human.py"]

    def test_the_approver_holds_no_orchestrator_state(self, fixture_world) -> None:
        human = fixture_world.human()
        assert not hasattr(human, "orchestrator") and not hasattr(human, "_orch")
