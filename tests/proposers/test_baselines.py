"""B0 to B3 (packet P6a): when each raises a work order, and that they only drive the job runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from closed_loop.causal.diagnoser import Diagnosis
from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import BIN_EDGES_DAYS, Cmd, Command, Op, PlanCmd
from closed_loop.perception.estimator import class_prior
from closed_loop.plant.jobs import ALPHABET
from closed_loop.plant.params import ASSET_CLASS, PARTS, PlantParams
from closed_loop.proposers import (
    BoundRule,
    Calendar,
    FixedLimit,
    RunToFailure,
    make_baseline,
)
from closed_loop.proposers.baselines import ORDER_RETRY_DAYS, default_mode
from tests.proposers._loop import run_episode
from tests.proposers._views import job, observation, view

PUMP_A, PUMP_B, CRUSHER, CONVEYOR = "PU-201", "PU-202", "CR-501", "CV-101"


def work_orders(commands: list[Command]) -> list[Command]:
    return [c for c in commands if c.cmd is PlanCmd.RAISE_WO]


@dataclass
class StubDiagnoser:
    """Fixed answer, counts calls. Stands in for the SCM diagnoser (still being finished)."""

    cause: str = "degradation"
    mode: str = "BEARING_DEFECT"
    name: str = "stub"
    calls: list[tuple[str, int, int]] = field(default_factory=list)
    fitted: bool = False

    def fit(self, lives: object) -> None:
        self.fitted = True

    def diagnose(self, asset_id: str, day: int, history: object) -> Diagnosis:
        self.calls.append((asset_id, day, history.n))  # type: ignore[attr-defined]
        return Diagnosis(asset_id, day, self.mode, self.cause, {})


class TestRegistry:
    def test_make_baseline_names(self) -> None:
        assert isinstance(make_baseline("B0"), RunToFailure)
        assert isinstance(make_baseline("B1"), Calendar)
        assert isinstance(make_baseline("B2"), FixedLimit)
        assert isinstance(make_baseline("B3", diagnoser=None), BoundRule)
        assert [make_baseline(n, **({"diagnoser": None} if n == "B3" else {})).name for n in
                ("B0", "B1", "B2", "B3")] == ["B0", "B1", "B2", "B3"]  # fmt: skip

    def test_keywords_reach_the_baseline(self) -> None:
        assert make_baseline("B1", fraction=0.25).fraction == 0.25  # type: ignore[attr-defined]
        assert make_baseline("B2", limit=4.0).limit == 4.0  # type: ignore[attr-defined]
        assert make_baseline("B3", k=7.0, diagnoser=None).k == 7.0  # type: ignore[attr-defined]

    def test_unknown_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="B9"):
            make_baseline("B9")

    def test_b3_defaults_to_an_scm_diagnoser_that_is_not_fitted(self) -> None:
        b3 = make_baseline("B3")
        assert b3.diagnoser.name == "scm"  # type: ignore[attr-defined]
        assert not b3.diagnoser.fitted  # type: ignore[attr-defined]


class TestB0RunToFailure:
    def test_nothing_on_a_healthy_asset_however_bad_the_forecast(self) -> None:
        b0 = RunToFailure()
        b0.reset(0)
        o = observation([view(CRUSHER, hi_last=8.9, rul_lo=0.5, rul_hi=1.0)])
        assert b0.propose(o) == []

    def test_a_work_order_only_after_the_failure(self) -> None:
        b0 = RunToFailure()
        b0.reset(0)
        out = b0.propose(observation([view(CRUSHER, op=Op.FAIL)]))
        assert [(c.asset_id, c.cmd, c.arg) for c in out] == [
            (CRUSHER, PlanCmd.RAISE_WO, "LINER_WEAR")
        ]
        assert out[0].source == "B0"

    @pytest.mark.parametrize("op", [Op.RUN, Op.STOP, Op.LOCKED, Op.ISO, Op.WORK])
    def test_only_the_failed_state_counts_as_a_failure(self, op: Op) -> None:
        b0 = RunToFailure()
        b0.reset(0)
        assert b0.propose(observation([view(CRUSHER, op=op)])) == []

    def test_the_mode_is_the_class_prior(self) -> None:
        assert default_mode(CRUSHER) == "LINER_WEAR"
        assert default_mode(PUMP_A) == "IMPELLER_WEAR"
        assert default_mode(CONVEYOR) == "BEARING_DEFECT"

    def test_one_work_order_per_cooldown(self) -> None:
        b0 = RunToFailure(cooldown_q=DAY_Q)
        b0.reset(0)
        failed = [view(CRUSHER, op=Op.FAIL)]
        assert b0.propose(observation(failed, quarter=10))
        assert b0.propose(observation(failed, quarter=11)) == []
        assert b0.propose(observation(failed, quarter=10 + DAY_Q - 1)) == []
        assert b0.propose(observation(failed, quarter=10 + DAY_Q))

    def test_no_second_order_while_one_is_open(self) -> None:
        b0 = RunToFailure()
        b0.reset(0)
        o = observation([view(CRUSHER, op=Op.FAIL, job_view=job(CRUSHER, "AWAITING_APPROVAL"))])
        assert work_orders(b0.propose(o)) == []


def orders(commands: list[Command]) -> list[Command]:
    return [c for c in commands if c.cmd is PlanCmd.ORDER]


def rule_baselines() -> list[Any]:
    return [
        RunToFailure(),
        Calendar(fraction=0.5),
        FixedLimit(limit=5.0),
        BoundRule(k=14.0, diagnoser=None),
    ]


class TestAFailedAssetGetsAWorkOrder:
    """0003 A17: before this a calendar baseline left a failed asset broken until its date, and a
    bound rule could argue itself out of the repair."""

    def test_b1_raises_on_a_failed_asset_long_before_its_calendar_date(self) -> None:
        b1 = Calendar(fraction=0.5)
        b1.reset(0)
        assert b1.propose(observation(quarter=0)) == []  # day 0: the clock starts
        out = work_orders(b1.propose(observation([view(CRUSHER, op=Op.FAIL)], quarter=2 * DAY_Q)))
        assert [(c.asset_id, c.arg, c.source) for c in out] == [(CRUSHER, "LINER_WEAR", "B1")]
        assert 2 < b1.interval_days[CRUSHER]  # it was not the calendar

    def test_b2_raises_on_a_failed_asset_whose_reading_is_low(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        out = work_orders(b2.propose(observation([view(PUMP_A, op=Op.FAIL, hi_last=1.0)])))
        assert [c.asset_id for c in out] == [PUMP_A]

    def test_b3_raises_on_a_failed_asset_whose_bound_is_high(self) -> None:
        b3 = BoundRule(k=14.0, diagnoser=None)
        b3.reset(0)
        v = view(CRUSHER, op=Op.FAIL, rul_lo=400.0)  # the forecast has not noticed
        assert [c.asset_id for c in work_orders(b3.propose(observation([v])))] == [CRUSHER]
        b4 = BoundRule(k=14.0, diagnoser=None)
        b4.reset(0)
        no_interval = view(CRUSHER, op=Op.FAIL, rul_lo=None)
        assert [c.asset_id for c in work_orders(b4.propose(observation([no_interval])))] == [
            CRUSHER
        ]

    @pytest.mark.parametrize("cause", ["sensor_fault", "process_upset"])
    def test_b3_raises_on_a_failed_asset_whose_diagnosis_is_not_degradation(
        self, cause: str
    ) -> None:
        diag = StubDiagnoser(cause=cause, mode="BEARING_DEFECT")
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        v = view(CRUSHER, op=Op.FAIL, hi_last=6.0, rul_lo=5.0)
        out = work_orders(b3.propose(observation([v])))
        # with the class's most likely mode: the diagnoser's mode is not trusted for a cause that
        # is not degradation
        assert [(c.asset_id, c.arg) for c in out] == [(CRUSHER, "LINER_WEAR")]
        assert diag.calls, "the diagnoser was asked"

    def test_b3_takes_the_diagnosed_mode_on_a_failed_asset_when_the_cause_is_degradation(
        self,
    ) -> None:
        diag = StubDiagnoser(cause="degradation", mode="BEARING_DEFECT")
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        out = work_orders(b3.propose(observation([view(CRUSHER, op=Op.FAIL, rul_lo=400.0)])))
        assert [(c.asset_id, c.arg) for c in out] == [(CRUSHER, "BEARING_DEFECT")]

    def test_a_healthy_asset_whose_diagnosis_is_not_degradation_still_gets_none(self) -> None:
        # the clause is for a failed asset only
        diag = StubDiagnoser(cause="sensor_fault")
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        assert b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0)])) == []

    @pytest.mark.parametrize("make", [RunToFailure, Calendar, FixedLimit])
    def test_the_cooldown_the_open_job_and_the_pump_serialisation_still_hold(self, make) -> None:
        p = make()
        p.reset(0)
        both = [view(PUMP_A, op=Op.FAIL), view(PUMP_B, op=Op.FAIL)]
        assert [c.asset_id for c in work_orders(p.propose(observation(both, quarter=10)))] == [
            PUMP_A
        ]
        # the same quarter after: cooldown for A, and B waits for A's order (not yet open: the
        # plant has not accepted it, so B is held by the cooldown on A only through "raised")
        again = work_orders(p.propose(observation(both, quarter=11)))
        assert [c.asset_id for c in again if c.asset_id == PUMP_A] == []
        open_job = [
            view(PUMP_A, op=Op.FAIL, job_view=job(PUMP_A, "AWAITING_APPROVAL", ready=False)),
            view(PUMP_B, op=Op.FAIL),
        ]
        assert work_orders(p.propose(observation(open_job, quarter=10 + 2 * DAY_Q))) == []
        crusher = view(CRUSHER, op=Op.FAIL, job_view=job(CRUSHER, "AWAITING_APPROVAL", ready=False))
        assert work_orders(p.propose(observation([crusher], quarter=3 * DAY_Q))) == []

    def test_the_day_a_job_completes_is_still_a_blackout_for_a_failed_asset(self) -> None:
        b1 = Calendar()
        b1.reset(0)
        b1.propose(observation([view(CRUSHER, jobs_done=0, day=3)], quarter=3 * DAY_Q))
        stale = observation([view(CRUSHER, op=Op.FAIL, jobs_done=1, day=3)], quarter=3 * DAY_Q + 5)
        assert work_orders(b1.propose(stale)) == []


class TestSparesAreReordered:
    """0003 A17: a reorder point of zero, the plainest stores rule there is."""

    PART = "SYN-LNR-5001"
    OTHER = "SYN-BRG-5001"

    def obs(self, quarter: int, stock: int, part: str | None = None) -> Any:
        part = part or self.PART
        return observation([view(CRUSHER, spares={part: stock})], quarter=quarter)

    @pytest.mark.parametrize("which", [0, 1, 2, 3], ids=["B0", "B1", "B2", "B3"])
    def test_every_rule_baseline_orders_a_part_that_shows_no_stock(self, which: int) -> None:
        p = rule_baselines()[which]
        p.reset(0)
        out = orders(p.propose(self.obs(10, 0)))
        assert [(c.asset_id, c.arg, c.source) for c in out] == [(CRUSHER, self.PART, p.name)]

    def test_nothing_is_ordered_while_there_is_stock(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        assert orders(b2.propose(self.obs(10, 1))) == []

    def test_one_order_when_the_stock_hits_zero_and_none_repeated(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        assert len(orders(b2.propose(self.obs(10, 0)))) == 1
        for q in range(11, 11 + 20 * DAY_Q, 7):  # every few quarters for twenty days
            assert orders(b2.propose(self.obs(q, 0))) == []

    def test_a_second_order_after_the_delivery_and_a_second_stock_out(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        assert len(orders(b2.propose(self.obs(10, 0)))) == 1
        assert orders(b2.propose(self.obs(5 * DAY_Q, 0))) == []
        assert orders(b2.propose(self.obs(6 * DAY_Q, 1))) == []  # delivered
        assert orders(b2.propose(self.obs(7 * DAY_Q, 1))) == []
        again = orders(b2.propose(self.obs(8 * DAY_Q, 0)))  # used up again
        assert [(c.asset_id, c.arg) for c in again] == [(CRUSHER, self.PART)]
        assert orders(b2.propose(self.obs(8 * DAY_Q + 1, 0))) == []

    def test_an_order_the_plant_may_have_refused_is_repeated_after_the_retry_days(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        assert len(orders(b2.propose(self.obs(0, 0)))) == 1
        assert ORDER_RETRY_DAYS == 30  # the planner's own default
        assert orders(b2.propose(self.obs((ORDER_RETRY_DAYS - 1) * DAY_Q, 0))) == []
        assert len(orders(b2.propose(self.obs(ORDER_RETRY_DAYS * DAY_Q, 0)))) == 1

    def test_the_retry_counts_from_the_day_of_the_order_not_from_the_start(self) -> None:
        # an order on day 100 is not repeated on day 101 to 129, and is on day 130
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        assert len(orders(b2.propose(self.obs(100 * DAY_Q, 0)))) == 1
        for day in (101, 115, 129):
            assert orders(b2.propose(self.obs(day * DAY_Q, 0))) == [], day
        assert len(orders(b2.propose(self.obs((100 + ORDER_RETRY_DAYS) * DAY_Q, 0)))) == 1

    def test_each_part_is_ordered_once_and_a_part_shared_by_two_assets_once(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        two = observation(
            [
                view(CRUSHER, spares={self.PART: 0, self.OTHER: 0}),
                view(PUMP_A, spares={self.OTHER: 0}),
            ]
        )
        out = orders(b2.propose(two))
        assert sorted(c.arg for c in out) == sorted([self.PART, self.OTHER])

    def test_one_part_in_stock_and_one_not(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        o = observation([view(CRUSHER, spares={self.PART: 1, self.OTHER: 0})])
        assert [c.arg for c in orders(b2.propose(o))] == [self.OTHER]

    def test_reset_forgets_what_was_ordered(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        assert len(orders(b2.propose(self.obs(10, 0)))) == 1
        b2.reset(1)
        assert len(orders(b2.propose(self.obs(10, 0)))) == 1

    def test_an_order_goes_with_a_work_order_command_in_one_proposal(self) -> None:
        b0 = RunToFailure()
        b0.reset(0)
        o = observation([view(CRUSHER, op=Op.FAIL, spares={self.PART: 0})])
        out = b0.propose(o)
        assert [c.cmd for c in out] == [PlanCmd.RAISE_WO, PlanCmd.ORDER]

    def test_the_hostile_wrappers_pass_an_order_through_or_ignore_it(self) -> None:
        from closed_loop.proposers.hostile import Careless, Gullible

        o = self.obs(10, 0)
        careless = Careless(FixedLimit(limit=5.0), p_skip=1.0)
        careless.reset(0)
        assert [c.cmd for c in careless.propose(o)] == [PlanCmd.ORDER]
        gullible = Gullible(FixedLimit(limit=5.0), screen=False, scan=lambda text: None)
        gullible.reset(0)
        assert [c.cmd for c in gullible.propose(o)] == [PlanCmd.ORDER]


class TestB1Calendar:
    def test_the_interval_is_a_fraction_of_the_characteristic_life(self) -> None:
        params = PlantParams()
        b1 = Calendar(fraction=0.5, params=params)
        for aid, cls in ASSET_CLASS.items():
            scale = class_prior(cls, params.renewal_onset_days).scale
            assert b1.interval_days[aid] == pytest.approx(0.5 * scale)
        assert Calendar(fraction=1.0).interval_days[CRUSHER] == pytest.approx(
            2 * b1.interval_days[CRUSHER]
        )

    def test_the_interval_itself_is_due(self) -> None:
        b1 = Calendar(fraction=0.5)
        b1.reset(0)
        b1.interval_days[CONVEYOR] = 10.0
        b1.propose(observation(quarter=0))

        def due(day: int) -> bool:
            out = b1.propose(observation(quarter=day * DAY_Q))
            return any(c.asset_id == CONVEYOR for c in work_orders(out))

        assert not due(9)
        assert due(10)

    def test_not_before_the_interval_and_then_yes(self) -> None:
        b1 = Calendar(fraction=0.5)
        b1.reset(0)
        t = b1.interval_days[PUMP_A]
        assert b1.propose(observation(quarter=0)) == []
        before = int(t) * DAY_Q - 1
        assert [c for c in b1.propose(observation(quarter=before)) if c.asset_id == PUMP_A] == []
        late = b1.propose(observation(quarter=int(t + 1) * DAY_Q))
        assert [c for c in work_orders(late) if c.asset_id == PUMP_A], "due once old enough"

    def test_a_completed_job_restarts_the_clock(self) -> None:
        b1 = Calendar(fraction=0.5)
        b1.reset(0)
        t = b1.interval_days[CONVEYOR]
        b1.propose(observation(quarter=0))
        day = int(t) + 5
        done = observation([view(CONVEYOR, jobs_done=1, day=day)], quarter=day * DAY_Q)
        got = b1.propose(done)  # the job just completed: no new order for the conveyor
        assert [c for c in work_orders(got) if c.asset_id == CONVEYOR] == []
        next_day = observation(
            [view(CONVEYOR, jobs_done=1, day=day + 1)], quarter=(day + 1) * DAY_Q
        )
        assert [c for c in work_orders(b1.propose(next_day)) if c.asset_id == CONVEYOR] == []
        due = int(day + t + 1)
        again = observation([view(CONVEYOR, jobs_done=1, day=due)], quarter=due * DAY_Q)
        assert [c for c in work_orders(b1.propose(again)) if c.asset_id == CONVEYOR]

    def test_reset_forgets_the_age(self) -> None:
        b1 = Calendar(fraction=0.5)
        b1.reset(0)
        b1.propose(observation(quarter=0))
        b1.reset(1)
        # as if the first call of a new episode: the clock starts at the day it is first seen
        t = b1.interval_days[CONVEYOR]
        q = int(t + 2) * DAY_Q
        assert b1.propose(observation(quarter=q)) == []  # first seen now, so not yet old


class TestB2FixedLimit:
    def test_below_the_limit_nothing_at_it_a_work_order(self) -> None:
        b2 = FixedLimit(limit=5.5)
        b2.reset(0)
        assert b2.propose(observation([view(CRUSHER, hi_last=5.49)])) == []
        out = work_orders(b2.propose(observation([view(CRUSHER, hi_last=5.5)], quarter=1)))
        assert [(c.asset_id, c.arg) for c in out] == [(CRUSHER, "LINER_WEAR")]

    def test_a_masked_day_is_no_reading(self) -> None:
        b2 = FixedLimit()
        b2.reset(0)
        assert b2.propose(observation([view(CRUSHER, hi_last=None)])) == []

    def test_the_default_limit_is_the_midpoint_of_alert_and_failure_levels(self) -> None:
        p = PlantParams()
        assert FixedLimit().limit == 0.5 * (p.hi_alert + p.hi_fail) == 5.5


class TestB3BoundRule:
    def test_raises_when_the_lower_bound_is_under_k_and_not_before(self) -> None:
        b3 = BoundRule(k=14.0, diagnoser=None)
        b3.reset(0)
        assert b3.propose(observation([view(CRUSHER, rul_lo=14.0, rul_hi=30.0)])) == []
        assert b3.propose(observation([view(CRUSHER, rul_lo=40.0, rul_hi=60.0)], quarter=1)) == []
        out = work_orders(b3.propose(observation([view(CRUSHER, rul_lo=13.9)], quarter=2)))
        assert [(c.asset_id, c.arg, c.source) for c in out] == [(CRUSHER, "LINER_WEAR", "B3")]

    def test_no_interval_no_action(self) -> None:
        b3 = BoundRule(diagnoser=None)
        b3.reset(0)
        assert b3.propose(observation([view(CRUSHER, rul_lo=None)])) == []

    def test_k_defaults_to_the_second_bin_edge(self) -> None:
        assert BoundRule(diagnoser=None).k == BIN_EDGES_DAYS[1] == 10.0

    @pytest.mark.parametrize(("bin_limit", "lo", "fires"), [
        (2, 10.5, False), (2, 9.9, True), (3, 3.5, False), (3, 2.9, True), (1, 29.0, True),
    ])  # fmt: skip
    def test_the_rule_can_be_stated_in_health_bins(
        self, bin_limit: int, lo: float, fires: bool
    ) -> None:
        b3 = BoundRule(bin_limit=bin_limit, diagnoser=None)
        b3.reset(0)
        out = work_orders(b3.propose(observation([view(CRUSHER, rul_lo=lo)])))
        assert bool(out) is fires

    def test_k_and_a_bin_limit_together_are_refused(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            BoundRule(k=5.0, bin_limit=2, diagnoser=None)
        with pytest.raises(ValueError, match="bin_limit"):
            BoundRule(bin_limit=0, diagnoser=None)
        with pytest.raises(ValueError, match="bin_limit"):
            BoundRule(bin_limit=4, diagnoser=None)

    def test_the_diagnosed_mode_goes_on_the_work_order(self) -> None:
        diag = StubDiagnoser(mode="BEARING_DEFECT")
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        out = work_orders(b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0)])))
        assert [(c.asset_id, c.arg) for c in out] == [(CRUSHER, "BEARING_DEFECT")]
        assert diag.calls and diag.calls[0][0] == CRUSHER

    @pytest.mark.parametrize("cause", ["sensor_fault", "process_upset"])
    def test_a_diagnosis_that_is_not_degradation_raises_no_work_order(self, cause: str) -> None:
        diag = StubDiagnoser(cause=cause)
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        o = observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0)])
        assert b3.propose(o) == []
        assert diag.calls, "the diagnoser was asked"

    def test_one_diagnosis_per_asset_per_day(self) -> None:
        diag = StubDiagnoser(cause="sensor_fault")
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        for q in range(0, 10):
            b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0)], quarter=q))
        assert len(diag.calls) == 1
        b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0, day=1)], quarter=DAY_Q))
        assert len(diag.calls) == 2

    def test_the_history_is_rebuilt_once_per_asset_per_day_not_once_per_quarter(self) -> None:
        """P12a F6: a diagnosis of "not degradation" raises no order, so the rule asks again in
        every quarter; the cache must be looked at before the history is rebuilt."""
        diag = StubDiagnoser(cause="sensor_fault")
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0)], quarter=0))
        recorder = b3._recorder
        assert recorder is not None
        built: list[str] = []
        real = recorder.history

        def counting(asset_id: str) -> Any:
            built.append(asset_id)
            return real(asset_id)

        recorder.history = counting  # type: ignore[method-assign]
        for q in range(1, DAY_Q):
            b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0)], quarter=q))
        assert built == []  # today's diagnosis is cached: no rebuild in the other 95 quarters
        b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0, day=1)], quarter=DAY_Q))
        assert built == [CRUSHER]  # a new day: one rebuild, one diagnosis
        assert len(diag.calls) == 2

    def test_the_diagnoser_sees_the_readings_of_the_life_so_far(self) -> None:
        diag = StubDiagnoser(cause="sensor_fault")
        b3 = BoundRule(k=1.0, diagnoser=diag)  # k below the interval: nothing raised, only recorded
        b3.reset(0)
        for d in range(4):
            b3.propose(
                observation([view(CRUSHER, hi_last=3.0 + d, rul_lo=50.0, day=d)], quarter=d * DAY_Q)
            )
        b3.k = 100.0
        b3.propose(observation([view(CRUSHER, hi_last=7.0, rul_lo=50.0, day=4)], quarter=4 * DAY_Q))
        assert diag.calls == [(CRUSHER, 4, 5)]

    def test_the_unfitted_default_diagnoser_fails_loudly(self) -> None:
        b3 = BoundRule(k=14.0)
        b3.reset(0)
        with pytest.raises(RuntimeError, match="fit"):
            b3.propose(observation([view(CRUSHER, hi_last=6.0, rul_lo=5.0)]))

    def test_fit_diagnoser_reaches_the_diagnoser(self) -> None:
        diag = StubDiagnoser()
        BoundRule(diagnoser=diag).fit_diagnoser([])
        assert diag.fitted
        with pytest.raises(ValueError, match="no diagnoser"):
            BoundRule(diagnoser=None).fit_diagnoser([])

    def test_a_reading_from_before_the_renewal_is_kept_out_of_the_new_history(self) -> None:
        diag = StubDiagnoser(cause="sensor_fault")
        b3 = BoundRule(k=14.0, diagnoser=diag)
        b3.reset(0)
        for d in range(3):
            b3.propose(
                observation([view(CRUSHER, hi_last=6.0, rul_lo=90.0, day=d)], quarter=d * DAY_Q)
            )
        # the work order completes on day 3: jobs_done moves, the day's reading is the old part's
        b3.propose(
            observation(
                [view(CRUSHER, hi_last=8.0, rul_lo=90.0, day=3, jobs_done=1)], quarter=3 * DAY_Q
            )
        )
        # a day later, the first reading of the new component
        b3.k = 100.0
        b3.propose(
            observation(
                [view(CRUSHER, hi_last=1.0, rul_lo=90.0, day=4, jobs_done=1)], quarter=4 * DAY_Q
            )
        )
        assert diag.calls == [(CRUSHER, 4, 1)]


class TestSharedBehaviour:
    def test_the_day_a_job_completes_is_a_blackout(self) -> None:
        b2 = FixedLimit(limit=5.5, cooldown_q=0)  # so that only the blackout can hold it back
        b2.reset(0)
        b2.propose(observation([view(CRUSHER, hi_last=6.0, jobs_done=0, day=3)], quarter=3 * DAY_Q))
        # the job completed; the daily reading is still the old part's, one day old
        stale = observation([view(CRUSHER, hi_last=6.0, jobs_done=1, day=3)], quarter=3 * DAY_Q + 5)
        assert work_orders(b2.propose(stale)) == []
        later = observation(
            [view(CRUSHER, hi_last=6.0, jobs_done=1, day=3)], quarter=3 * DAY_Q + 60
        )
        assert work_orders(b2.propose(later)) == []
        tomorrow = observation([view(CRUSHER, hi_last=6.0, jobs_done=1, day=4)], quarter=4 * DAY_Q)
        assert work_orders(b2.propose(tomorrow))

    def test_the_two_pumps_are_never_ordered_together(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        both = [view(PUMP_A, hi_last=7.0), view(PUMP_B, hi_last=7.0)]
        out = work_orders(b2.propose(observation(both)))
        assert [c.asset_id for c in out] == [PUMP_A]

    def test_a_pump_waits_for_the_other_pumps_open_order(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        o = observation(
            [view(PUMP_A, hi_last=7.0, job_view=job(PUMP_A, "AWAITING_APPROVAL")),
             view(PUMP_B, hi_last=7.0)]
        )  # fmt: skip
        assert work_orders(b2.propose(o)) == []

    def test_serialising_can_be_switched_off(self) -> None:
        b2 = FixedLimit(limit=5.0, serialise_pumps=False)
        b2.reset(0)
        both = [view(PUMP_A, hi_last=7.0), view(PUMP_B, hi_last=7.0)]
        assert {c.asset_id for c in work_orders(b2.propose(observation(both)))} == {PUMP_A, PUMP_B}

    def test_an_open_job_is_driven_by_the_job_runner_and_nothing_else(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        v = view(PUMP_A, op=Op.RUN, hi_last=7.0, job_view=job(PUMP_A))
        out = b2.propose(observation([v], quarter=40))
        assert [(c.asset_id, c.cmd) for c in out] == [(PUMP_A, Cmd.STOP)]
        assert out[0].source == "B2"

    def test_a_job_that_is_not_released_gives_no_command(self) -> None:
        b2 = FixedLimit(limit=5.0)
        b2.reset(0)
        v = view(PUMP_A, hi_last=7.0, job_view=job(PUMP_A, "AWAITING_APPROVAL", ready=False))
        assert b2.propose(observation([v])) == []

    def test_cooldown_and_serialising_validate_their_arguments(self) -> None:
        with pytest.raises(ValueError, match="cooldown_q"):
            FixedLimit(cooldown_q=-1)
        with pytest.raises(ValueError, match="fraction"):
            Calendar(fraction=0.0)


class TestResetAndDeterminism:
    @pytest.mark.parametrize("name", ["B0", "B2", "B3"])
    def test_reset_clears_the_state_the_baseline_keeps(self, name: str) -> None:
        kw = {"diagnoser": None} if name == "B3" else {}
        p = make_baseline(name, **kw)
        failed = observation(
            [view(CRUSHER, op=Op.FAIL, hi_last=9.0, rul_lo=0.1), view(PUMP_A, jobs_done=2)],
            quarter=7,
        )
        p.reset(1)
        first = p.propose(failed)
        assert first, f"{name} should order the crusher"
        assert p.propose(observation([view(CRUSHER, op=Op.FAIL, hi_last=9.0, rul_lo=0.1)],
                                     quarter=8)) == []  # fmt: skip
        p.reset(2)
        assert p.propose(failed) == first

    @pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3"])
    def test_two_runs_with_one_seed_issue_the_same_commands(self, name: str) -> None:
        kw = {"diagnoser": None} if name == "B3" else {}
        a = run_episode(
            make_baseline(name, **kw), 5, perception=name == "B3", horizon_q=3 * DAY_Q * 10
        )
        b = run_episode(
            make_baseline(name, **kw), 5, perception=name == "B3", horizon_q=3 * DAY_Q * 10
        )
        assert a.issued == b.issued


class TestWholeEpisodes:
    @pytest.mark.parametrize("name", ["B0", "B1", "B2", "B3"])
    def test_each_baseline_completes_an_f1_episode_with_commands_from_the_alphabet(
        self, name: str
    ) -> None:
        kw = {"diagnoser": None} if name == "B3" else {}
        ep = run_episode(make_baseline(name, **kw), 3, perception=name == "B3")
        assert ep.trace is not None and ep.trace.n_q == 8640
        for _q, c in ep.issued:
            # 0003 A17: a work order, a reorder of a spare (a part number, nothing else), or a
            # command of the job runner's alphabet
            assert c.cmd in (PlanCmd.RAISE_WO, PlanCmd.ORDER) or c.cmd in ALPHABET, c
            if c.cmd is PlanCmd.ORDER:
                assert c.arg in PARTS, c
            assert c.source == name

    def test_the_baselines_together_raise_some_work_orders(self) -> None:
        raised = 0
        for name in ("B1", "B2", "B3"):
            kw = {"diagnoser": None} if name == "B3" else {}
            for seed in (1, 2):
                ep = run_episode(make_baseline(name, **kw), seed, perception=name == "B3")
                raised += sum(1 for _q, c in ep.issued if c.cmd is PlanCmd.RAISE_WO)
        assert raised > 0

    def test_b0_orders_only_after_a_failure_in_a_real_episode(self) -> None:
        for seed in (1, 2, 3):
            ep = run_episode(make_baseline("B0"), seed)
            assert ep.plant is not None and ep.trace is not None
            first_order: dict[str, int] = {}
            for q, c in ep.issued:
                if c.cmd is PlanCmd.RAISE_WO:
                    first_order.setdefault(c.asset_id, q)
            for aid, q in first_order.items():
                failed_by_then = bool(ep.trace.uncontrolled_failure[aid][: q + 1].any())
                assert failed_by_then, (seed, aid, q)
