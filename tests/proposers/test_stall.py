"""0003 A18, R4 and R5: the daily restart offer, the stall watch, and a job for a stalled asset.

The unit tests use hand-built observations (``_views.py``); the episode tests run whole episodes on
the SYNTHETIC plant with the small prepared context, on development seeds only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from closed_loop.core.timegrid import DAY_Q
from closed_loop.core.types import Cmd, Command, Op, PlanCmd
from closed_loop.proposers import BoundRule, Calendar, FixedLimit, RunToFailure
from closed_loop.proposers.restart import OPERATOR, Restarting
from closed_loop.proposers.stall import STALL_Q, StallWatch
from tests.proposers._views import job, observation, view

CRUSHER, PUMP_A = "CR-501", "PU-201"
FIRST = DAY_Q * 10 + 1  # the quarter whose view holds the day's update


@dataclass
class Scripted:
    commands: list[Command] = field(default_factory=list)
    name: str = "scripted"
    resets: list[int] = field(default_factory=list)

    def reset(self, seed: int) -> None:
        self.resets.append(seed)

    def propose(self, obs: object) -> list[Command]:
        return list(self.commands)


def starts(out: list[Command]) -> list[str]:
    return [c.asset_id for c in out if c.cmd is Cmd.START]


def work_orders(out: list[Command]) -> list[Command]:
    return [c for c in out if c.cmd is PlanCmd.RAISE_WO]


class TestTheStallDefinition:
    def test_the_threshold_is_a_day_and_two_quarters(self) -> None:
        assert STALL_Q == DAY_Q + 2

    def test_stalled_exactly_at_the_threshold_and_not_a_quarter_before(self) -> None:
        w = StallWatch()
        w.observe(observation([view(CRUSHER, op=Op.STOP)], quarter=100))
        before = observation([view(CRUSHER, op=Op.STOP)], quarter=100 + STALL_Q - 1)
        w.observe(before)
        assert not w.stalled(CRUSHER, before) and w.stalled_assets(before) == frozenset()
        at = observation([view(CRUSHER, op=Op.STOP)], quarter=100 + STALL_Q)
        w.observe(at)
        assert w.stalled(CRUSHER, at) and w.stalled_assets(at) == frozenset({CRUSHER})

    @pytest.mark.parametrize("op", [Op.RUN, Op.FAIL, Op.LOCKED, Op.ISO, Op.WORK])
    def test_only_a_stopped_asset_stalls(self, op: Op) -> None:
        w = StallWatch()
        for q in (100, 100 + STALL_Q + 5):
            o = observation([view(CRUSHER, op=op)], quarter=q)
            w.observe(o)
        assert not w.stalled(CRUSHER, o)

    def test_an_open_work_order_ends_the_stall(self) -> None:
        w = StallWatch()
        w.observe(observation([view(CRUSHER, op=Op.STOP)], quarter=100))
        o = observation([view(CRUSHER, op=Op.STOP, job_view=job(CRUSHER))], quarter=100 + STALL_Q)
        w.observe(o)
        assert not w.stalled(CRUSHER, o)

    def test_running_again_forgets_the_clock(self) -> None:
        w = StallWatch()
        w.observe(observation([view(CRUSHER, op=Op.STOP)], quarter=100))
        w.observe(observation([view(CRUSHER, op=Op.RUN)], quarter=101))
        o = observation([view(CRUSHER, op=Op.STOP)], quarter=100 + STALL_Q)
        w.observe(o)
        assert not w.stalled(CRUSHER, o), "the clock restarted at the new stop"

    def test_reset_forgets_everything(self) -> None:
        w = StallWatch()
        w.observe(observation([view(CRUSHER, op=Op.STOP)], quarter=100))
        w.reset()
        o = observation([view(CRUSHER, op=Op.STOP)], quarter=100 + STALL_Q)
        assert not w.stalled(CRUSHER, o)


class TestTheDailyRestartOffer:
    def stopped(self, *ids: str, quarter: int = FIRST) -> Any:
        return observation([view(a, op=Op.STOP) for a in ids], quarter=quarter)

    def test_a_stopped_asset_is_offered_a_start_once_a_day(self) -> None:
        r = Restarting(Scripted())
        out = r.propose(self.stopped(CRUSHER))
        assert out == [Command(CRUSHER, Cmd.START, None, OPERATOR)] and r.offers == 1
        for q in (FIRST + 1, FIRST + 5, FIRST + DAY_Q - 1, FIRST - 1):
            assert r.propose(self.stopped(CRUSHER, quarter=q)) == [], q
        assert starts(r.propose(self.stopped(CRUSHER, quarter=FIRST + DAY_Q))) == [CRUSHER]
        assert r.offers == 2

    def test_every_stopped_asset_with_no_job_and_no_inner_command_is_offered(self) -> None:
        spoken = Command(PUMP_A, PlanCmd.INSPECT, None, "inner")
        r = Restarting(Scripted([spoken]))
        o = observation(
            [
                view(CRUSHER, op=Op.STOP),
                view(PUMP_A, op=Op.STOP),
                view("PU-202", op=Op.STOP, job_view=job("PU-202")),
                view("CV-101", op=Op.RUN),
            ],
            quarter=FIRST,
        )
        out = r.propose(o)
        assert starts(out) == [CRUSHER], "not the pump the inner spoke for, nor one with a job"
        assert r.offers == 1 and spoken in out

    @pytest.mark.parametrize("op", [Op.RUN, Op.FAIL, Op.LOCKED, Op.ISO, Op.WORK])
    def test_only_a_stopped_asset_is_offered_one(self, op: Op) -> None:
        r = Restarting(Scripted())
        assert r.propose(observation([view(CRUSHER, op=op)], quarter=FIRST)) == []

    def test_the_seam_is_the_inners(self) -> None:
        inner = Scripted()
        r = Restarting(inner)
        assert r.name == "scripted"
        r.propose(self.stopped(CRUSHER))
        r.reset(7)
        assert inner.resets == [7] and r.offers == 0


PUMP_B = "PU-202"


class TestThePairStop:
    """0003 A18, R8. A shielded pump job asks for LOCK on the reserve; the proof refuses it while
    the other pump runs, so the operator stops the other pump beside it (the allowed set holds
    LOCK|STOP)."""

    def lock(self, aid: str = PUMP_A) -> Command:
        return Command(aid, Cmd.LOCK, None, "inner")

    def obs(self, other: Any = None, quarter: int = FIRST + 5, **kw: Any) -> Any:
        other = other or view(PUMP_B, op=Op.RUN, set_hi=3)
        return observation(
            [view(PUMP_A, op=Op.STOP, job_view=job(PUMP_A)), other], quarter=quarter, **kw
        )

    def test_a_lock_on_one_pump_stops_the_running_other_at_the_top_bin(self) -> None:
        r = Restarting(Scripted([self.lock()]), pair_stop=True)
        out = r.propose(self.obs())
        assert out == [self.lock(), Command(PUMP_B, Cmd.STOP, None, OPERATOR)]
        assert r.pair_stops == 1 and r.offers == 0

    def test_it_works_the_other_way_round(self) -> None:
        r = Restarting(Scripted([self.lock(PUMP_B)]), pair_stop=True)
        o = observation(
            [view(PUMP_B, op=Op.STOP, job_view=job(PUMP_B)), view(PUMP_A, op=Op.RUN, set_hi=3)],
            quarter=FIRST + 5,
        )
        assert Command(PUMP_A, Cmd.STOP, None, OPERATOR) in r.propose(o)

    def test_it_happens_in_any_quarter_not_only_the_first_of_the_day(self) -> None:
        r = Restarting(Scripted([self.lock()]), pair_stop=True)
        for q in (FIRST - 1, FIRST, FIRST + 7, FIRST + DAY_Q - 1):
            assert starts(r.propose(self.obs(quarter=q))) == []
        assert r.pair_stops == 4

    def test_it_waits_for_the_other_pump_to_reach_the_top_bin_or_a_monitor_to_fire(self) -> None:
        r = Restarting(Scripted([self.lock()]), pair_stop=True)
        low = view(PUMP_B, op=Op.RUN, set_hi=2)
        assert r.propose(self.obs(low)) == [self.lock()] and r.pair_stops == 0
        out = r.propose(self.obs(low, fired=True))
        assert Command(PUMP_B, Cmd.STOP, None, OPERATOR) in out and r.pair_stops == 1

    @pytest.mark.parametrize("op", [Op.STOP, Op.FAIL, Op.WORK, Op.LOCKED, Op.ISO])
    def test_the_other_pump_must_be_running(self, op: Op) -> None:
        r = Restarting(Scripted([self.lock()]), pair_stop=True)
        assert r.propose(self.obs(view(PUMP_B, op=op, set_hi=3))) == [self.lock()]

    def test_the_other_pump_must_have_no_job(self) -> None:
        r = Restarting(Scripted([self.lock()]), pair_stop=True)
        busy = view(PUMP_B, op=Op.RUN, set_hi=3, job_view=job(PUMP_B))
        assert r.propose(self.obs(busy)) == [self.lock()]

    def test_it_yields_to_a_command_the_inner_proposer_gave_the_other_pump(self) -> None:
        said = Command(PUMP_B, PlanCmd.INSPECT, None, "inner")
        r = Restarting(Scripted([self.lock(), said]), pair_stop=True)
        assert r.propose(self.obs()) == [self.lock(), said] and r.pair_stops == 0

    @pytest.mark.parametrize("cmd", [Cmd.STOP, Cmd.START, Cmd.BEGIN])
    def test_only_a_lock_triggers_it(self, cmd: Cmd) -> None:
        r = Restarting(Scripted([Command(PUMP_A, cmd, None, "inner")]), pair_stop=True)
        assert all(c.source != OPERATOR for c in r.propose(self.obs()))

    def test_a_lock_on_a_non_pump_does_not(self) -> None:
        r = Restarting(Scripted([Command("CV-101", Cmd.LOCK, None, "inner")]), pair_stop=True)
        assert all(c.source != OPERATOR for c in r.propose(self.obs()))

    def test_off_by_default_and_when_the_switch_is_off(self) -> None:
        for r in (
            Restarting(Scripted([self.lock()])),
            Restarting(Scripted([self.lock()]), pair_stop=False),
        ):
            assert r.propose(self.obs()) == [self.lock()] and r.pair_stops == 0

    def test_reset_clears_the_count(self) -> None:
        r = Restarting(Scripted([self.lock()]), pair_stop=True)
        r.propose(self.obs())
        r.reset(3)
        assert r.pair_stops == 0


class TestAStalledAssetGetsAJob:
    """R5: the rule baselines treat a stalled asset as they treat a failed one."""

    def stalled_obs(self, quarter: int) -> Any:
        return observation([view(CRUSHER, op=Op.STOP, hi_last=0.5, rul_lo=400.0)], quarter=quarter)

    @pytest.mark.parametrize(
        "make",
        [
            RunToFailure,
            lambda: Calendar(fraction=0.5),
            lambda: FixedLimit(limit=5.0),
            lambda: BoundRule(k=14.0, diagnoser=None),
        ],
        ids=["B0", "B1", "B2", "B3"],
    )
    def test_a_rule_baseline_raises_a_work_order_once_the_stop_has_stood_long_enough(
        self, make: Any
    ) -> None:
        p = make()
        p.reset(0)
        t0 = 3 * DAY_Q + 10
        assert work_orders(p.propose(self.stalled_obs(t0))) == []
        assert work_orders(p.propose(self.stalled_obs(t0 + STALL_Q - 1))) == []
        out = work_orders(p.propose(self.stalled_obs(t0 + STALL_Q)))
        assert [(c.asset_id, c.cmd) for c in out] == [(CRUSHER, PlanCmd.RAISE_WO)]

    @pytest.mark.parametrize("op", [Op.RUN, Op.WORK, Op.LOCKED])
    def test_an_asset_that_is_not_stopped_is_not_made_a_job(self, op: Op) -> None:
        p = RunToFailure()
        p.reset(0)
        for q in (3 * DAY_Q, 3 * DAY_Q + STALL_Q + 3):
            out = p.propose(observation([view(CRUSHER, op=op)], quarter=q))
        assert work_orders(out) == []

    def test_a_stopped_asset_with_a_job_open_is_left_to_the_runner(self) -> None:
        p = RunToFailure()
        p.reset(0)
        v = view(CRUSHER, op=Op.STOP, job_view=job(CRUSHER, "AWAITING_APPROVAL", ready=False))
        for q in (3 * DAY_Q, 3 * DAY_Q + STALL_Q + 3):
            assert work_orders(p.propose(observation([v], quarter=q))) == []
