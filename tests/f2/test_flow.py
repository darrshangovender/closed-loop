"""P11b: one alert through the real agent, the real gate and the plant CMMS (the "done when").

An alert is produced by the real asset-health pipeline from a rendered degrading pump, read by the
real ``Orchestrator``, verified by the real sixteen rules, signed by ``SimHumanApprover`` through
the real ``ApprovalGate`` and released by ``PlantCMMS`` over the real ``MockCMMS``. Nothing is
mocked.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from closed_loop.f2.agent import run_id_for
from closed_loop.f2.render import sim_time

pytestmark = pytest.mark.f2


def raised(world) -> str:
    outcome = world.source.raise_alert("PU-201")
    assert outcome.kind == "raised", outcome
    assert outcome.alert_id is not None
    return outcome.alert_id


class TestOneAlertEndToEnd:
    def test_the_run_reaches_awaiting_approval(self, pump_world) -> None:
        alert_id = raised(pump_world)
        state = pump_world.orch.run(alert_id, run_id=run_id_for(alert_id))
        assert state.status.value == "AWAITING_APPROVAL"
        assert state.verification.passed and not state.verification.violations
        assert len(state.verification.rules_run) >= 16  # V00..V15

    def test_the_human_signs_through_the_real_gate_and_the_cmms_queues_a_job(
        self, pump_world
    ) -> None:
        world = pump_world
        alert_id = raised(world)
        run_id = run_id_for(alert_id)
        state = world.orch.run(alert_id, run_id=run_id)
        human = world.human()
        for _ in range(12):  # looks again each hour until the latency has passed
            state = human.act(world.orch, run_id)
            if state.status.value == "COMPLETED":
                break
            world.advance(hours=1)
        assert state.status.value == "COMPLETED"
        assert state.release.wo_number.startswith("WO-9")
        assert state.release.approver_id == "planner1"
        assert [wo for wo, _ in world.queued] == [state.release.wo_number]
        draft = world.queued[0][1]
        assert draft.alert_id == alert_id and draft.asset_id == "PU-201"
        assert human.log[-1].kind == "approve"
        # the approval was issued in simulation time, not wall-clock time
        assert state.approval.issued_at.year == 2026 and state.approval.issued_at.month == 11
        assert (state.approval.expires_at - state.approval.issued_at).total_seconds() == 8 * 3600

    def test_the_same_alert_eight_days_later_is_blocked_by_v10(self, pump_world) -> None:
        """Reproduces the owner's observation: a reading over 3 days old is stale (0001 s10.3)."""
        world = pump_world
        alert_id = raised(world)
        world.advance(days=8)
        state = world.orch.run(alert_id, run_id="late-run")
        assert state.status.value == "BLOCKED"
        assert state.verification.rule_ids() == {"V10"}
        messages = " ".join(v.message for v in state.verification.violations)
        assert "8 days before" in messages and "maximum 3" in messages

    def test_inside_the_window_the_same_reading_is_not_stale(self, pump_world) -> None:
        world = pump_world
        alert_id = raised(world)
        world.advance(days=2, hours=2)
        state = world.orch.run(alert_id, run_id="two-days")
        assert state.status.value == "AWAITING_APPROVAL"

    def test_a_blocked_draft_is_rejected_by_the_human_and_nothing_is_released(
        self, pump_world
    ) -> None:
        world = pump_world
        alert_id = raised(world)
        world.advance(days=8)
        run_id = run_id_for(alert_id)
        world.orch.run(alert_id, run_id=run_id)
        human = world.human()
        state = None
        for _ in range(12):
            state = human.act(world.orch, run_id)
            if state.status.value == "REJECTED":
                break
            world.advance(hours=1)
        assert state is not None and state.status.value == "REJECTED"
        assert world.queued == [] and world.mock.released() == []
        assert human.log[-1].kind == "reject" and "V10" in human.log[-1].reason

    def test_the_alerts_as_of_is_the_simulation_date(self, pump_world) -> None:
        alert = pump_world.source.get_alert(raised(pump_world))
        assert alert.rul.as_of == date(2026, 11, 9)
        assert alert.rul.as_of == sim_time(39.5).date()
        assert alert.raised_at.tzinfo is not None

    def test_the_draft_is_an_inspection_first_on_weak_evidence(self, pump_world) -> None:
        """What the agent makes of asset-health's evidence at this reading (a finding, not a bug).

        The agent's impeller-wear signature is 50% flow shortfall and 30% discharge pressure, which
        the rule layer has not yet put among its four strongest channels; the vane-pass bands map to
        overall vibration (20%). The agent therefore cannot confirm the mode and drafts a
        diagnostic inspection, naming impeller wear as the leading candidate.
        """
        world = pump_world
        alert_id = raised(world)
        state = world.orch.run(alert_id, run_id=run_id_for(alert_id))
        draft = state.draft
        assert draft["inspection_first"] is True
        assert draft["escalations"] == ["AMBIGUOUS_FAILURE_MODE"]
        assert draft["hypotheses"][0]["mode"] == "IMPELLER_WEAR"
        assert draft["hypotheses"][0]["confidence"] < 0.55
        record = world.source.record(alert_id)
        assert record.hypothesis == "impeller_wear" and record.mode == "IMPELLER_WEAR"


class TestDuplicates:
    def test_a_second_run_for_the_same_alert_is_refused_by_the_cmms(self, pump_world) -> None:
        world = pump_world
        alert_id = raised(world)
        first = world.orch.run(alert_id, run_id="first-run")
        human = world.human()
        for _ in range(12):
            state = human.act(world.orch, "first-run")
            if state.status.value == "COMPLETED":
                break
            world.advance(hours=1)
        assert state.status.value == "COMPLETED" and first.run_id == "first-run"
        second = world.orch.run(alert_id, run_id="second-run")
        human2 = world.human("planner1", rng=np.random.default_rng(5))
        for _ in range(12):
            second = human2.act(world.orch, "second-run")
            if second.status.value != "AWAITING_APPROVAL":
                break
            world.advance(hours=1)
        assert second.status.value == "ESCALATED"
        assert second.escalation == "duplicate_work_order"
        assert len(world.queued) == 1  # the plant got one job

    def test_the_alert_source_refuses_a_duplicate_id(self, pump_world) -> None:
        from closed_loop.f2.alert_source import DuplicateAlert

        world = pump_world
        alert_id = raised(world)
        with pytest.raises(DuplicateAlert, match=alert_id):
            world.source.raise_alert("PU-201", alert_id, refresh=True)
