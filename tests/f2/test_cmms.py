"""P11b: PlantCMMS over the real MockCMMS: plant stock, the sibling's checks, one job per order."""

from __future__ import annotations

import pytest

from closed_loop.f2.cmms import inventory_of
from closed_loop.f2.render import sim_time
from closed_loop.plant.params import PARTS

from .conftest import make_world

pytestmark = pytest.mark.f2


def drive(world, human, run_id: str, hours: int = 48):
    """Look at a run once an hour until it is no longer waiting for this human (or time is up)."""
    state = world.orch.get(run_id)
    for _ in range(hours):
        state = human.act(world.orch, run_id)
        if state.status.value not in ("AWAITING_APPROVAL", "BLOCKED"):
            break
        world.advance(hours=1)
    return state


class TestProtocol:
    def test_it_is_a_cmms(self, fixture_world) -> None:
        from maintenance_agent.ports import CMMS

        assert isinstance(fixture_world.cmms, CMMS)


class TestPlantStock:
    def test_the_stock_comes_from_the_plant_not_the_fixture(self, fixture_world) -> None:
        world = fixture_world
        assert world.mock.get_part("SYN-BRG-1001").stock_qty == 6  # the agent's own table
        world.stock["SYN-BRG-1001"] = 1
        part = world.cmms.get_part("SYN-BRG-1001")
        assert part.stock_qty == 1 and part.lead_time_days == 21
        assert world.cmms.plant_lookups == 1 and world.cmms.fallback_lookups == 0

    def test_a_part_the_plant_does_not_track_is_read_from_the_inner_mock_and_counted(
        self, fixture_world
    ) -> None:
        part = fixture_world.cmms.get_part("SYN-OIL-0001")
        assert part is not None and part.stock_qty == 12
        assert fixture_world.cmms.fallback_lookups == 1
        assert fixture_world.cmms.get_part("SYN-NOT-A-PART") is None

    def test_a_read_is_a_copy(self, fixture_world) -> None:
        part = fixture_world.cmms.get_part("SYN-BRG-1001")
        part.stock_qty = 99
        assert fixture_world.cmms.get_part("SYN-BRG-1001").stock_qty == 6

    def test_the_inventory_is_live(self) -> None:
        stock = {"SYN-IMP-2001": 2}
        lookup = inventory_of(lambda: stock)
        assert lookup("SYN-IMP-2001").stock_qty == 2
        stock["SYN-IMP-2001"] = 0
        assert lookup("SYN-IMP-2001").stock_qty == 0
        assert lookup("SYN-IMP-2001").lead_time_days == 42
        assert lookup("SYN-OTHER") is None
        assert PARTS["SYN-IMP-2001"].lead_days == 42.0

    def test_the_verifier_sees_the_plants_shortage(self, params) -> None:
        """V11 and V12 read stock through ``get_part``: a short plant makes the agent say so."""

        from maintenance_agent.adapters.synthetic import SyntheticAlertSource

        def run(stock: int):
            world = make_world(
                params,
                lambda clock: SyntheticAlertSource.from_fixtures(today=clock().date()),
                now=sim_time(39.5),
                stock={**{p: s.stock for p, s in PARTS.items()}, "SYN-BRG-1001": stock},
            )
            return world.orch.run("ALR-1001", run_id="r")

        enough, short = run(6), run(1)
        assert "PARTS_SHORTFALL" not in enough.draft["escalations"]
        assert "PARTS_SHORTFALL" in short.draft["escalations"]
        assert "SYN-BRG-1001 cannot arrive before" in short.draft["notes"]
        assert "cannot arrive" not in enough.draft["notes"]
        assert enough.verification.passed and short.verification.passed

    def test_the_dry_run_uses_plant_stock(self, fixture_world) -> None:
        from maintenance_agent.models import WorkOrderDraft

        world = fixture_world
        world.stock["SYN-BRG-1001"] = 1
        state = world.orch.run("ALR-1001", run_id="r")
        draft = WorkOrderDraft.model_validate(state.draft)
        receipt = world.cmms.draft_work_order(draft)
        component = receipt.payload["components"][0]
        assert component["in_stock"] is False and component["lead_time_days"] == 21
        assert receipt.ready_date is not None and (receipt.ready_date - draft.as_of).days == 21
        # the sibling's own table would have said "in stock", so the plant's stock is what spoke
        assert world.mock.draft_work_order(draft).payload["components"][0]["in_stock"] is True
        assert receipt.payload["status"] == "NOT_CREATED" and world.mock.released() == []


class TestRelease:
    def signed(self, world, alert_id: str = "ALR-1001", run_id: str = "r"):
        state = world.orch.run(alert_id, run_id=run_id)
        human = world.human()
        assert human.review(state) is None  # first sight starts the latency
        world.advance(days=1)  # well past any latency
        state = world.orch.refresh_verification(run_id)
        approval = human.review(state)
        assert approval is not None and approval.decision == "approve"
        from maintenance_agent.models import WorkOrderDraft

        return WorkOrderDraft.model_validate(state.draft), approval

    def test_a_release_without_an_approval_is_refused_by_the_sibling(self, fixture_world) -> None:
        from maintenance_agent.approval import ApprovalError

        draft, _ = self.signed(fixture_world)
        with pytest.raises(ApprovalError) as err:
            fixture_world.cmms.release_work_order(draft, None)
        assert err.value.code == "NO_APPROVAL" and fixture_world.queued == []

    def test_a_release_creates_a_work_order_and_queues_one_job(self, fixture_world) -> None:
        world = fixture_world
        draft, approval = self.signed(world)
        released = world.cmms.release_work_order(draft, approval)
        assert released.wo_number == "WO-900001" and released.approver_id == "planner1"
        assert [(wo, d.alert_id) for wo, d in world.queued] == [("WO-900001", "ALR-1001")]
        assert world.cmms.released_orders == ["WO-900001"]
        assert world.cmms.draft_of("WO-900001").hash() == draft.hash()

    def test_a_repeat_release_returns_the_same_order_and_queues_nothing(
        self, fixture_world
    ) -> None:
        world = fixture_world
        draft, approval = self.signed(world)
        first = world.cmms.release_work_order(draft, approval)
        again = world.cmms.release_work_order(draft, approval)
        assert again.wo_number == first.wo_number and len(world.queued) == 1
        assert world.cmms.find_release(approval.approval_id).wo_number == first.wo_number
        assert world.cmms.find_release_for("r", draft.hash()).wo_number == first.wo_number
        assert len(world.queued) == 1

    def test_a_second_run_for_a_released_alert_is_a_duplicate(self, fixture_world) -> None:
        from maintenance_agent.approval import ApprovalError

        world = fixture_world
        draft, approval = self.signed(world, run_id="r1")
        world.cmms.release_work_order(draft, approval)
        draft2, approval2 = self.signed(world, run_id="r2")
        with pytest.raises(ApprovalError) as err:
            world.cmms.release_work_order(draft2, approval2)
        assert err.value.code == "DUPLICATE" and len(world.queued) == 1

    def test_the_wrong_role_is_refused_at_the_last_mile(self, fixture_world) -> None:
        """A planner's signature on a confined-space liner job: the sibling's role check refuses."""
        from maintenance_agent.approval import ApprovalError, required_role
        from maintenance_agent.models import WorkOrderDraft

        world = fixture_world
        state = world.orch.run("ALR-1011", run_id="liner")  # CR-501 liner wear
        draft = WorkOrderDraft.model_validate(state.draft)
        assert required_role(draft) == "SAFETY_OFFICER" and not draft.inspection_first
        human = world.human(signs_above_rank=True)
        assert human.review(state) is None
        world.advance(days=1)
        state = world.orch.refresh_verification("liner")
        approval = human.review(state)
        assert approval is not None
        with pytest.raises(ApprovalError) as err:
            world.cmms.release_work_order(draft, approval)
        assert err.value.code == "ROLE_INSUFFICIENT" and world.queued == []
        assert world.mock.released() == []

    def test_the_orchestrator_goes_back_to_waiting_after_a_role_refusal(
        self, fixture_world
    ) -> None:
        world = fixture_world
        world.orch.run("ALR-1011", run_id="liner")
        human = world.human(signs_above_rank=True)
        state = drive(world, human, "liner")
        assert state.status.value == "AWAITING_APPROVAL" and state.approval is None
        refusals = [
            r for r in world.orch.audit("liner").records() if r["type"] == "release_refused"
        ]
        assert [r["code"] for r in refusals] == ["ROLE_INSUFFICIENT"]  # signed once, not in a loop
        assert world.queued == []

    def test_a_queue_that_fails_is_retried_when_the_release_is_next_looked_up(self, params) -> None:
        from maintenance_agent.adapters.synthetic import SyntheticAlertSource

        calls: list[str] = []
        queued: list[str] = []

        def flaky(draft, wo):
            calls.append(wo)
            if len(calls) == 1:
                raise RuntimeError("plant job queue is down")
            queued.append(wo)

        world = make_world(
            params,
            lambda clock: SyntheticAlertSource.from_fixtures(today=clock().date()),
            now=sim_time(39.5),
            queue=flaky,
        )
        world.orch.run("ALR-1001", run_id="r")
        state = drive(world, world.human(), "r")
        assert state.status.value == "FAILED" and "queue is down" in state.error
        assert len(world.mock.released()) == 1  # the order exists
        state = world.orch.resume("r")
        assert state.status.value == "COMPLETED"
        assert calls == ["WO-900001", "WO-900001"] and queued == ["WO-900001"]
