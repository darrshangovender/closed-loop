"""The hand-written interlock (ADR 0003 A4): each rule, the seam, the online monitors, the lags.

The rules are tested on hand-built observations (``tests/proposers/_views.py``); the claim that the
online monitors agree with the offline scorer is tested on whole plant episodes.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np
import pytest

from closed_loop.core.types import Cmd, Command, Op, Permit, PlanCmd
from closed_loop.eval import interlock as interlock_module
from closed_loop.eval.interlock import (
    R_BEGIN,
    R_ENTER_GAS,
    R_ENTER_PERMIT,
    R_EXIT_ALARM,
    R_EXIT_GAS,
    R_PUMP_LOCK,
    R_STOP,
    RULES,
    InterlockedProposer,
    default_point_rul,
)
from closed_loop.plant.params import PUMPS
from closed_loop.proposers import Careless, make_baseline
from closed_loop.shield.runtime import ShieldedProposer
from closed_loop.spec.properties import SpecParams, property_formulas, signals_from_trace
from closed_loop.spec.stl import Always, robustness
from tests.proposers._loop import run_episode
from tests.proposers._views import observation, view

CR, CV, PA, PB = "CR-501", "CV-101", "PU-201", "PU-202"


@dataclass
class Scripted:
    commands: list[Command] = field(default_factory=list)
    name: str = "scripted"
    resets: list[int] = field(default_factory=list)

    def reset(self, seed: int) -> None:
        self.resets.append(seed)

    def propose(self, obs: object) -> list[Command]:
        return list(self.commands)


def cmd(asset: str, c: Cmd | PlanCmd, arg: str | None = None) -> Command:
    return Command(asset, c, arg, "inner")


def run_once(
    commands: list[Command], *views, quarter: int = 40, **kw
) -> tuple[InterlockedProposer, list[Command]]:  # type: ignore[no-untyped-def]
    """A fresh interlock around a scripted inner proposer, one decision."""
    il = InterlockedProposer(Scripted(commands), **kw)
    return il, il.propose(observation(list(views), quarter=quarter))


def kinds(out: list[Command]) -> list[tuple[str, object]]:
    return [(c.asset_id, c.cmd) for c in out]


class TestRule1BeginNeedsProvenIsolation:
    @pytest.mark.parametrize("op", [Op.RUN, Op.STOP, Op.LOCKED, Op.FAIL])
    def test_refused_on_an_asset_whose_isolation_is_not_proven(self, op: Op) -> None:
        il, out = run_once([cmd(CR, Cmd.BEGIN)], view(CR, op=op, permit=Permit.VALID))
        assert out == []
        assert il.by_rule[R_BEGIN] == 1 and il.refused == 1 and il.overrides == 1

    def test_admitted_once_the_isolation_is_proven(self) -> None:
        il, out = run_once([cmd(CR, Cmd.BEGIN)], view(CR, op=Op.ISO, permit=Permit.VALID))
        assert kinds(out) == [(CR, Cmd.BEGIN)]
        assert il.overrides == 0 and il.admitted == 1

    def test_applies_to_every_asset(self) -> None:
        for aid in (CV, PA, PB):
            _, out = run_once([cmd(aid, Cmd.BEGIN)], view(aid, op=Op.STOP))
            assert out == [], aid

    def test_an_asset_already_at_work_keeps_the_proof_it_began_with(self) -> None:
        il = InterlockedProposer(Scripted([cmd(CR, Cmd.BEGIN)]))
        il.propose(observation([view(CR, op=Op.ISO)], quarter=10))
        out = il.propose(observation([view(CR, op=Op.WORK)], quarter=11))
        assert kinds(out) == [(CR, Cmd.BEGIN)]

    def test_work_that_began_on_an_unproven_asset_never_counts_as_proven(self) -> None:
        il = InterlockedProposer(Scripted([cmd(CR, Cmd.BEGIN)]))
        il.propose(observation([view(CR, op=Op.LOCKED)], quarter=10))
        out = il.propose(observation([view(CR, op=Op.WORK)], quarter=11))
        assert out == []


class TestACommandOfNeitherKind:
    """P12g, G6: the interlock does not wave through a command it cannot classify."""

    def forged(self, asset: str = CR) -> Command:
        c = Command(asset, Cmd.BEGIN, source="forged")
        object.__setattr__(c, "cmd", "BEGIN")  # round the constructor, which refuses it
        return c

    def test_a_plain_string_command_stops_the_episode_instead_of_passing(self) -> None:
        with pytest.raises(TypeError, match="neither a Cmd nor a PlanCmd"):
            run_once([self.forged()], view(CR, op=Op.RUN, permit=Permit.VALID))

    def test_it_is_refused_on_every_asset_not_only_the_crusher(self) -> None:
        for asset in (CV, PA, PB):
            with pytest.raises(TypeError):
                run_once([self.forged(asset)], view(asset, op=Op.RUN))

    def test_planning_commands_still_pass_untouched(self) -> None:
        il, out = run_once([cmd(CR, PlanCmd.RAISE_WO, "LINER_WEAR")], view(CR, op=Op.RUN))
        assert kinds(out) == [(CR, PlanCmd.RAISE_WO)] and il.overrides == 0

    def test_the_rules_are_the_same_seven(self) -> None:
        assert len(RULES) == 7  # a refusal under a new rule would have changed every by_rule


class TestRule2EnterNeedsAValidPermit:
    @pytest.mark.parametrize("permit", [Permit.NONE, Permit.REVAL])
    def test_refused_without_a_valid_permit(self, permit: Permit) -> None:
        il, out = run_once([cmd(CR, Cmd.ENTER)], view(CR, op=Op.WORK, permit=permit, gas_age_q=1))
        assert out == [] and il.by_rule[R_ENTER_PERMIT] == 1

    def test_admitted_with_a_valid_permit_and_a_fresh_test(self) -> None:
        il, out = run_once(
            [cmd(CR, Cmd.ENTER)], view(CR, op=Op.WORK, permit=Permit.VALID, gas_age_q=1)
        )
        assert kinds(out) == [(CR, Cmd.ENTER)] and il.overrides == 0


class TestRule3EnterNeedsAFreshTest:
    @pytest.mark.parametrize(
        ("gas_age_q", "admitted"), [(1, True), (2, True), (3, False), (4, False), (99, False)]
    )
    def test_the_test_must_be_inside_the_period_when_the_entry_happens(
        self, gas_age_q: int, admitted: bool
    ) -> None:
        il, out = run_once(
            [cmd(CR, Cmd.ENTER)],
            view(CR, op=Op.WORK, permit=Permit.VALID, gas_age_q=gas_age_q),
        )
        assert bool(out) is admitted
        assert il.by_rule[R_ENTER_GAS] == int(not admitted)

    def test_the_break_window_is_not_one_of_the_six_rules(self) -> None:
        """A test two quarters old passes the interlock though the plant's P3 break rule flags it:
        the interlock is exactly the six rules of 0003 A4, not the clause."""
        _, out = run_once(
            [cmd(CR, Cmd.ENTER)], view(CR, op=Op.WORK, permit=Permit.VALID, gas_age_q=2)
        )
        assert out

    def test_a_longer_period_in_the_spec_moves_the_limit(self) -> None:
        spec = SpecParams(retest_q=8, break_q=8)
        for age, admitted in ((8, True), (9, False)):
            _, out = run_once(
                [cmd(CR, Cmd.ENTER)],
                view(CR, op=Op.WORK, permit=Permit.VALID, gas_age_q=age),
                spec=spec,
            )
            assert bool(out) is admitted


class TestRule4ExitIsForced:
    def test_when_the_gas_margin_is_a_quarter_or_less(self) -> None:
        for age, forced in ((1, False), (2, True), (3, True), (99, True)):
            il, out = run_once(
                [cmd(CR, Cmd.TEST)],
                view(CR, op=Op.WORK, permit=Permit.VALID, occupied=True, gas_age_q=age),
            )
            assert (kinds(out) == [(CR, Cmd.EXIT)]) is forced, age
            assert il.by_rule[R_EXIT_GAS] == int(forced)

    def test_when_the_alarm_is_on(self) -> None:
        il, out = run_once(
            [cmd(CR, Cmd.TEST)],
            view(CR, op=Op.WORK, permit=Permit.VALID, occupied=True, gas_age_q=1, alarm=True),
        )
        assert kinds(out) == [(CR, Cmd.EXIT)]
        assert il.by_rule[R_EXIT_ALARM] == 1 and il.forced == 1 and il.overrides == 1

    def test_nobody_inside_nothing_to_force(self) -> None:
        il, out = run_once(
            [cmd(CR, Cmd.TEST)],
            view(CR, op=Op.WORK, permit=Permit.VALID, occupied=False, gas_age_q=99, alarm=True),
        )
        assert kinds(out) == [(CR, Cmd.TEST)] and il.overrides == 0

    def test_the_forced_command_replaces_the_inner_commands_for_that_asset_only(self) -> None:
        il, out = run_once(
            [cmd(CR, Cmd.TEST), cmd(PA, Cmd.STOP), cmd(CR, PlanCmd.INSPECT)],
            view(CR, op=Op.WORK, permit=Permit.VALID, occupied=True, gas_age_q=2),
        )
        assert kinds(out) == [(PA, Cmd.STOP), (CR, PlanCmd.INSPECT), (CR, Cmd.EXIT)]
        assert out[-1].source == il.name

    def test_an_exit_the_inner_proposer_already_issued_is_not_an_override(self) -> None:
        il, out = run_once(
            [cmd(CR, Cmd.EXIT)],
            view(CR, op=Op.WORK, permit=Permit.VALID, occupied=True, gas_age_q=3, alarm=True),
        )
        assert out == [cmd(CR, Cmd.EXIT)]
        assert il.overrides == 0 and il.forced == 0 and il.admitted == 1

    def test_the_alarm_is_named_before_the_margin(self) -> None:
        il, _ = run_once(
            [],
            view(CR, op=Op.WORK, permit=Permit.VALID, occupied=True, gas_age_q=3, alarm=True),
        )
        assert il.by_rule[R_EXIT_ALARM] == 1 and il.by_rule[R_EXIT_GAS] == 0


class TestRule5PumpLock:
    @pytest.mark.parametrize("other", [Op.STOP, Op.ISO, Op.WORK, Op.FAIL, Op.LOCKED])
    def test_refused_while_the_other_pump_is_not_running(self, other: Op) -> None:
        il, out = run_once([cmd(PA, Cmd.LOCK)], view(PA, op=Op.STOP), view(PB, op=other))
        assert out == [] and il.by_rule[R_PUMP_LOCK] == 1

    def test_admitted_while_the_other_pump_runs(self) -> None:
        il, out = run_once([cmd(PB, Cmd.LOCK)], view(PA, op=Op.RUN), view(PB, op=Op.STOP))
        assert kinds(out) == [(PB, Cmd.LOCK)] and il.overrides == 0

    def test_the_crusher_and_the_conveyor_can_be_locked_freely(self) -> None:
        for aid in (CR, CV):
            _, out = run_once([cmd(aid, Cmd.LOCK)], view(aid, op=Op.STOP), view(PA, op=Op.STOP))
            assert kinds(out) == [(aid, Cmd.LOCK)]


class TestRule6StopOnLowRul:
    def test_a_point_estimate_under_three_days_forces_a_stop(self) -> None:
        il, out = run_once([], view(CR, op=Op.RUN, rul_lo=0.5, rul_hi=4.0))  # midpoint 2.25
        assert kinds(out) == [(CR, Cmd.STOP)]
        assert il.by_rule[R_STOP] == 1 and il.forced == 1

    @pytest.mark.parametrize(
        ("lo", "hi", "stops"),
        [(2.0, 4.0, False), (1.0, 5.0, False), (2.9, 2.9, True), (0.0, 5.9, True)],
    )
    def test_the_midpoint_against_three_days(self, lo: float, hi: float, stops: bool) -> None:
        _, out = run_once([], view(CR, op=Op.RUN, rul_lo=lo, rul_hi=hi))
        assert bool(out) is stops

    def test_no_estimate_no_stop(self) -> None:
        _, out = run_once([], view(CR, op=Op.RUN, rul_lo=None))
        assert out == []
        _, out = run_once([], view(CR, op=Op.RUN, rul_lo=0.0, rul_hi=None))
        assert out == []

    def test_only_a_running_crusher_is_stopped(self) -> None:
        for op in (Op.STOP, Op.LOCKED, Op.ISO, Op.WORK, Op.FAIL):
            _, out = run_once([], view(CR, op=op, rul_lo=0.0, rul_hi=1.0))
            assert out == [], op

    def test_the_other_assets_are_not_stopped(self) -> None:
        _, out = run_once([], view(PA, op=Op.RUN, rul_lo=0.0, rul_hi=1.0))
        assert out == []

    def test_the_point_estimate_can_be_injected(self) -> None:
        il, out = run_once(
            [], view(CR, op=Op.RUN, rul_lo=50.0, rul_hi=60.0), point_rul=lambda v: 1.0
        )
        assert kinds(out) == [(CR, Cmd.STOP)]
        _, out = run_once([], view(CR, op=Op.RUN, rul_lo=0.0, rul_hi=1.0), point_rul=lambda v: None)
        assert out == []

    def test_the_default_point_is_the_interval_midpoint(self) -> None:
        assert default_point_rul(view(CR, rul_lo=2.0, rul_hi=6.0)) == 4.0
        assert default_point_rul(view(CR, rul_lo=None, rul_hi=6.0)) is None

    def test_the_day_a_work_order_completes_is_a_hold_off(self) -> None:
        il = InterlockedProposer(Scripted())
        low = dict(op=Op.RUN, rul_lo=0.0, rul_hi=1.0)
        assert il.propose(observation([view(CR, jobs_done=0, **low)], quarter=96 * 3 + 10))
        out = il.propose(observation([view(CR, jobs_done=1, **low)], quarter=96 * 3 + 11))
        assert out == [], "the interval is the old component's until tomorrow"
        out = il.propose(observation([view(CR, jobs_done=1, **low)], quarter=96 * 3 + 80))
        assert out == []
        out = il.propose(observation([view(CR, jobs_done=1, **low)], quarter=96 * 4))
        assert kinds(out) == [(CR, Cmd.STOP)], "tomorrow the estimate has caught up"


class TestRule6RefusesTheRestartOfAnAssetItWouldStopAgain:
    """0003 A18 R4: a stopped crusher is offered a restart once a day; while the estimate is still
    under the bound rule 6 would stop it on the next row, so the restart is refused and counted
    once, not admitted and then overridden every day."""

    low = dict(op=Op.STOP, rul_lo=0.5, rul_hi=4.0)  # midpoint 2.25 < 3

    def test_a_start_of_the_crusher_under_the_bound_is_refused_as_rule_6(self) -> None:
        il, out = run_once([cmd(CR, Cmd.START)], view(CR, **self.low))
        assert out == []
        assert il.by_rule[R_STOP] == 1 and il.refused == 1 and il.forced == 0

    def test_a_start_with_the_estimate_over_the_bound_is_admitted(self) -> None:
        il, out = run_once([cmd(CR, Cmd.START)], view(CR, op=Op.STOP, rul_lo=5.0, rul_hi=9.0))
        assert kinds(out) == [(CR, Cmd.START)] and il.overrides == 0

    def test_a_start_with_no_estimate_is_admitted(self) -> None:
        _, out = run_once([cmd(CR, Cmd.START)], view(CR, op=Op.STOP, rul_lo=None))
        assert kinds(out) == [(CR, Cmd.START)]

    def test_the_day_a_work_order_completes_the_start_is_admitted(self) -> None:
        """The estimate on the view is the old part's until tomorrow; the hold-off of rule 6
        covers the restart that closes a job as well."""
        il = InterlockedProposer(Scripted([cmd(CR, Cmd.START)]))
        il.propose(observation([view(CR, jobs_done=0, **self.low)], quarter=96 * 3 + 10))
        out = il.propose(observation([view(CR, jobs_done=1, **self.low)], quarter=96 * 3 + 11))
        assert kinds(out) == [(CR, Cmd.START)]

    def test_only_the_crusher_is_held_by_its_estimate(self) -> None:
        # the crusher's own estimate is under the bound; a pump restart is not its business
        _, out = run_once([cmd(PA, Cmd.START)], view(CR, **self.low), view(PA, op=Op.STOP))
        assert kinds(out) == [(PA, Cmd.START)]
        _, out = run_once([cmd(PA, Cmd.START)], view(PA, **self.low))
        assert kinds(out) == [(PA, Cmd.START)]

    def test_a_refusal_is_one_override_not_one_per_row(self) -> None:
        il = InterlockedProposer(Scripted([cmd(CR, Cmd.START)]))
        for q in (96 * 5 + 1, 96 * 6 + 1):
            assert il.propose(observation([view(CR, **self.low)], quarter=q)) == []
        assert il.by_rule[R_STOP] == 2 and il.forced == 0  # one refusal a day, never a forced stop


class TestNothingElseIsTouched:
    @pytest.mark.parametrize(
        "c",
        [Cmd.NOOP, Cmd.START, Cmd.STOP, Cmd.LOCK, Cmd.PROVE, Cmd.ISSUE, Cmd.REVAL, Cmd.TEST,
         Cmd.EXIT, Cmd.COMPLETE, Cmd.CLOSE, Cmd.UNISO, Cmd.DERATE],
    )  # fmt: skip
    def test_other_commands_pass_unchanged(self, c: Cmd) -> None:
        il, out = run_once([cmd(CR, c)], view(CR, op=Op.STOP, rul_lo=50.0, rul_hi=60.0))
        assert out == [cmd(CR, c)] and il.overrides == 0

    def test_planning_commands_pass(self) -> None:
        planned = [cmd(CR, PlanCmd.RAISE_WO, "LINER_WEAR"), cmd(PA, PlanCmd.INSPECT)]
        il, out = run_once(planned, view(CR, op=Op.RUN))
        assert out == planned and il.overrides == 0

    def test_a_command_for_an_asset_the_observation_lacks_passes(self) -> None:
        il = InterlockedProposer(Scripted([cmd("XX-000", Cmd.BEGIN)]))
        assert il.propose(observation([], quarter=3)) == [cmd("XX-000", Cmd.BEGIN)]

    def test_the_six_rules_are_the_whole_list(self) -> None:
        assert len(RULES) == 7 and len(set(RULES)) == 7  # six rules; rule 4 has two triggers


class TestTheSeamIsTheShields:
    def test_same_surface_as_the_shielded_proposer(self) -> None:
        il = InterlockedProposer(Scripted())
        for name in ("propose", "reset", "overrides", "admitted", "name", "inner"):
            assert hasattr(il, name), name
        shield_names = {n for n in dir(ShieldedProposer) if not n.startswith("_")}
        assert {"propose", "reset"} <= shield_names
        assert list(inspect.signature(il.propose).parameters) == ["obs"]
        assert list(inspect.signature(ShieldedProposer.propose).parameters) == ["self", "obs"]

    def test_reset_clears_everything_and_forwards_the_seed(self) -> None:
        inner = Scripted([cmd(CR, Cmd.BEGIN)])
        il = InterlockedProposer(inner)
        il.propose(observation([view(CR, op=Op.RUN, rul_lo=0.0, rul_hi=1.0)], quarter=7))
        assert il.overrides > 0 and il.robustness("P1")
        il.reset(5)
        assert inner.resets == [5]
        assert (il.overrides, il.admitted, il.refused, il.forced) == (0, 0, 0, 0)
        assert not any(il.by_rule.values()) and il.robustness("P1") == []
        il.reset()
        assert inner.resets == [5]

    def test_the_name_says_what_it_wraps(self) -> None:
        assert InterlockedProposer(Scripted(name="B3")).name == "interlock(B3)"

    def test_it_reads_no_game_no_perception_set_and_no_truth(self) -> None:
        source = inspect.getsource(interlock_module)
        code = "\n".join(
            line.split("#")[0] for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        body = code.split('"""', 2)[-1]  # after the module docstring
        for word in ("set_lo", "set_hi", "closed_loop.shield", "solve(", "ShieldTable",
                     "TruthView", "health_bin", "true_rul", ".truth"):  # fmt: skip
            assert word not in body, word


class TestOnlineMonitors:
    def feed(self, il: InterlockedProposer, rows: list[dict]) -> None:  # type: ignore[type-arg]
        for q, kw in enumerate(rows):
            il.propose(observation([view(CR, **kw)], quarter=q))

    def test_one_push_per_decision_and_the_row_is_the_quarter_before(self) -> None:
        il = InterlockedProposer(Scripted())
        self.feed(il, [dict(op=Op.RUN)] * 5)
        assert len(il.robustness("P2")) == 5

    def test_p3_goes_negative_at_the_row_where_the_test_is_too_old(self) -> None:
        il = InterlockedProposer(Scripted())
        inside = dict(op=Op.WORK, permit=Permit.VALID, occupied=True)
        ages = [1, 2, 3, 4]  # a test 0, 15, 30 and 45 minutes old at the row the view describes
        self.feed(il, [dict(inside, gas_age_q=a) for a in ages])
        # 30 minutes is exactly the period and still compliant; 45 is not (the specification's
        # margin is the period less the age)
        assert [x < 0 for x in il.robustness("P3")] == [False, False, False, True]

    def test_the_gas_margin_monitor_is_the_specifications_atom(self) -> None:
        il = InterlockedProposer(Scripted())
        il.propose(observation([view(CR, gas_age_q=1)], quarter=1))
        assert il._gas.last == 30.0  # retest 30 min, a test from the previous step
        il.propose(observation([view(CR, gas_age_q=2)], quarter=2))
        assert il._gas.last == 15.0

    def test_the_rul_margin_monitor_is_the_point_estimate_less_three_days(self) -> None:
        il = InterlockedProposer(Scripted())
        il.propose(observation([view(CR, rul_lo=4.0, rul_hi=10.0)], quarter=1))
        assert il._rul.last == pytest.approx(7.0 - 3.0)

    def test_the_inner_formulas_are_monitored_without_the_outer_g(self) -> None:
        il = InterlockedProposer(Scripted())
        for pid, f in property_formulas(il.spec).items():
            assert isinstance(f, Always)
            assert il._monitors[pid].formula == f.arg

    def test_a_violation_that_ends_lets_the_robustness_recover(self) -> None:
        il = InterlockedProposer(Scripted())
        bad = dict(op=Op.WORK, permit=Permit.NONE, occupied=True, gas_age_q=1)
        ok = dict(op=Op.RUN)
        self.feed(il, [ok, bad, ok, ok])
        assert [x < 0 for x in il.robustness("P2")] == [False, True, False, False]


class TestWhatTheBreakItPassFound:
    """One regression per mutant of the break-it pass that survived the first tests."""

    def feed(self, il: InterlockedProposer, rows: list[dict]) -> None:  # type: ignore[type-arg]
        for q, kw in enumerate(rows):
            il.propose(observation([view(CR, **kw)], quarter=q))

    def test_a_reentry_on_a_test_older_than_the_break_is_a_p3_violation_in_the_second_clause(
        self,
    ) -> None:
        # a 2-hour period (so the first clause is loose) and a break of more than one quarter
        il = InterlockedProposer(Scripted(), SpecParams(retest_q=8, break_q=1))
        inside = dict(op=Op.WORK, permit=Permit.VALID, occupied=True)
        out = dict(op=Op.WORK, permit=Permit.VALID, occupied=False)
        rows = [dict(inside, gas_age_q=1)] * 11  # tests in every quarter, rows 0 to 10
        rows += [dict(out, gas_age_q=2), dict(out, gas_age_q=3)]  # rows 11 and 12: empty
        rows += [dict(inside, gas_age_q=4)]  # row 13: back in on the test of row 10
        self.feed(il, rows)
        negative = [x < 0 for x in il.robustness("P3")]
        assert negative[13] and not any(negative[:13])
        # empty for 30 minutes, the test 45 minutes old: it does not cover the break

    def test_the_same_reentry_with_a_test_taken_after_the_break_is_fine(self) -> None:
        il = InterlockedProposer(Scripted(), SpecParams(retest_q=8, break_q=1))
        inside = dict(op=Op.WORK, permit=Permit.VALID, occupied=True)
        out = dict(op=Op.WORK, permit=Permit.VALID, occupied=False)
        rows = [dict(inside, gas_age_q=1)] * 11
        rows += [dict(out, gas_age_q=1), dict(out, gas_age_q=1)]  # tested while empty
        rows += [dict(inside, gas_age_q=1)]
        self.feed(il, rows)
        assert not any(x < 0 for x in il.robustness("P3"))

    def test_an_uncontrolled_failure_is_seen_once_in_the_quarter_it_shows(self) -> None:
        il = InterlockedProposer(Scripted())
        self.feed(il, [dict(op=Op.RUN)] * 3 + [dict(op=Op.FAIL)] * 3)
        assert [x < 0 for x in il.robustness("P5")] == [False] * 3 + [True, False, False]

    def test_a_stopped_pump_is_available_and_an_isolated_one_is_not(self) -> None:
        def p4(a: Op, b: Op) -> bool:
            il = InterlockedProposer(Scripted())
            il.propose(observation([view(PA, op=a), view(PB, op=b)], quarter=1))
            return il.robustness("P4")[0] < 0

        assert not p4(Op.STOP, Op.STOP)
        assert not p4(Op.RUN, Op.ISO)
        assert not p4(Op.STOP, Op.WORK)
        assert p4(Op.ISO, Op.WORK)
        assert p4(Op.FAIL, Op.ISO)

    def test_rule_3_is_about_the_crusher_s_gas_test_only(self) -> None:
        """A pump has no chamber to enter; the crusher's stale test must not be pinned on it."""
        _, out = run_once(
            [cmd(PA, Cmd.ENTER)],
            view(CR, op=Op.WORK, permit=Permit.VALID, gas_age_q=99),
            view(PA, op=Op.WORK, permit=Permit.VALID),
        )
        assert kinds(out) == [(PA, Cmd.ENTER)]


class TestDetectionLags:
    def run(self, rows: list[dict], violations: list[tuple[str, str, int]]):  # type: ignore[no-untyped-def]
        il = InterlockedProposer(Scripted())
        for q, kw in enumerate(rows):
            il.propose(observation([view(CR, **kw)], quarter=q))
        return il.episode_record(violations)

    def test_work_on_an_unproven_asset_is_seen_at_once(self) -> None:
        rows = [dict(op=Op.RUN)] * 3 + [dict(op=Op.WORK)] * 3  # row r is observed at quarter r + 1
        ep = self.run(rows, [("P1", CR, 2)])  # work began at row 2
        (v,) = ep.violations
        assert (v.prop, v.asset_id, v.quarter, v.first_negative, v.lag_q) == ("P1", CR, 2, 2, 0)

    def test_work_on_a_proven_asset_is_no_violation_of_the_monitor(self) -> None:
        rows = [dict(op=Op.RUN), dict(op=Op.ISO), dict(op=Op.WORK), dict(op=Op.WORK)]
        ep = self.run(rows, [("P1", CR, 1)])
        assert ep.violations[0].first_negative is None and ep.undetected == 1

    def test_p6_is_seen_one_quarter_after_the_alarm(self) -> None:
        inside = dict(op=Op.WORK, permit=Permit.VALID, occupied=True, gas_age_q=1)
        rows = [dict(op=Op.RUN)] * 2 + [dict(inside, alarm=True)] * 2 + [dict(op=Op.RUN)]
        ep = self.run(rows, [("P6", CR, 1)])  # the alarm row; the plant dates P6 by it
        assert ep.violations[0].first_negative == 2 and ep.violations[0].lag_q == 1

    def test_the_lag_is_counted_from_the_occurrence_not_from_the_first_ever_violation(self) -> None:
        bad = dict(op=Op.WORK, permit=Permit.NONE, occupied=True, gas_age_q=1)
        ok = dict(op=Op.RUN)
        rows = [ok, bad, ok, ok, ok, bad, ok]
        ep = self.run(rows, [("P2", CR, 0), ("P2", CR, 4)])
        assert [v.lag_q for v in ep.violations] == [0, 0]

    def test_a_property_that_stays_positive_is_undetected(self) -> None:
        ep = self.run([dict(op=Op.RUN)] * 6, [("P5", CR, 2)])
        assert ep.violations[0].lag_q is None

    def test_an_unknown_property_is_undetected_not_an_error(self) -> None:
        ep = self.run([dict(op=Op.RUN)] * 3, [("P9", CR, 1)])
        assert ep.violations[0].lag_q is None

    def test_violations_come_back_in_time_order_with_the_counts(self) -> None:
        il = InterlockedProposer(Scripted([cmd(CR, Cmd.BEGIN)]))
        for q in range(4):
            il.propose(observation([view(CR, op=Op.RUN)], quarter=q))
        ep = il.episode_record([("P2", CV, 9), ("P1", CR, 2), ("P1", CR, 1)])
        assert [(v.quarter, v.prop) for v in ep.violations] == [(1, "P1"), (2, "P1"), (9, "P2")]
        assert ep.refused == 4 and ep.admitted == 0 and ep.by_rule[R_BEGIN] == 4


def run_with_interlock(seed: int, p_skip: float, *, interlock: bool = True):  # type: ignore[no-untyped-def]
    holder: dict[str, InterlockedProposer] = {}

    def wrap(plant, proposer):  # type: ignore[no-untyped-def]
        if not interlock:
            return proposer
        holder["il"] = InterlockedProposer(proposer)
        return holder["il"]

    ep = run_episode(Careless(make_baseline("B2"), p_skip), seed, wrap=wrap)
    return ep, holder.get("il")


class TestOnlineAgreesWithOfflineOnWholeEpisodes:
    @pytest.mark.parametrize("seed", [1, 2, 3, 4])
    def test_the_verdicts_of_the_online_monitors_are_those_of_the_offline_scorer(
        self, seed: int
    ) -> None:
        ep, il = run_with_interlock(seed, 0.3, interlock=True)
        assert il is not None and ep.trace is not None
        spec = il.spec
        signals = signals_from_trace(ep.trace, spec, CR, PUMPS)
        n = ep.trace.n_q
        for pid, f in property_formulas(spec).items():
            assert isinstance(f, Always)
            offline = robustness(f.arg, signals)  # row r
            online = il.robustness(pid)  # rows -1, 0, ..., n - 2
            assert len(online) == n
            got = np.array(online[1:]) < 0  # rows 0 .. n - 2
            want = offline[: n - 1] < 0
            assert np.array_equal(got, want), (seed, pid, np.flatnonzero(got != want)[:5])

    def test_the_episode_has_violations_to_time_and_the_record_is_consistent(self) -> None:
        total = lagged = 0
        for seed in range(1, 9):
            ep, il = run_with_interlock(seed, 0.3, interlock=True)
            assert il is not None
            rec = il.episode_record(ep.violations)
            assert len(rec.violations) == len(ep.violations)
            assert rec.refused + rec.forced == il.overrides
            total += len(rec.violations)
            lagged += sum(1 for v in rec.violations if v.lag_q is not None)
            for v in rec.violations:
                assert v.lag_q is None or v.lag_q >= 0
        assert total > 0, "the careless fitter gets some through the six rules"


class TestTheInterlockRemovesWhatItIsWrittenFor:
    def test_it_refuses_begin_before_prove_and_the_p1_count_falls(self) -> None:
        with_il = without = 0
        refused = 0
        for seed in range(1, 9):
            a, il = run_with_interlock(seed, 0.3, interlock=True)
            b, _ = run_with_interlock(seed, 0.3, interlock=False)
            assert il is not None
            with_il += a.violated("P1")
            without += b.violated("P1")
            refused += il.by_rule[R_BEGIN]
        assert refused > 0
        assert with_il < without

    def test_an_honest_job_runner_never_trips_the_three_refusals(self) -> None:
        for seed in (1, 2, 3, 4):
            holder: dict[str, InterlockedProposer] = {}

            def wrap(plant, proposer, holder=holder):  # type: ignore[no-untyped-def]
                holder["il"] = InterlockedProposer(proposer)
                return holder["il"]

            run_episode(make_baseline("B2"), seed, wrap=wrap)
            il = holder["il"]
            assert il.by_rule[R_BEGIN] == 0, seed
            assert il.by_rule[R_ENTER_PERMIT] == 0 and il.by_rule[R_ENTER_GAS] == 0, seed
