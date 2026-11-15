"""ShieldedProposer: filtering, substitution by priority, counters, widening and a hostile loop.

Since ADR 0003 A12 the shield gates three games: the crusher job, the pump pair and the conveyor
CV-101, and a pump takes LOCK and PROVE under their own names (the abstract ISOLATE is retired).
"""

from __future__ import annotations

import pytest

from closed_loop.core.rng import rng_for
from closed_loop.core.types import Cmd, Command, PlanCmd
from closed_loop.shield.abstraction import ConveyorModel, CrusherJobModel, PumpPairModel
from closed_loop.shield.game import Winning, solve
from closed_loop.shield.runtime import CONVEYOR_ID, CRUSHER_ID, PUMP_IDS, ShieldedProposer
from closed_loop.shield.table import ShieldTable, build_shield
from tests.shield.mapping_stub import (
    AbstractPlant,
    RandomProposer,
    ScriptedProposer,
    StubMapper,
    StubView,
    observation,
    quiet_pump,
)

A, B = PUMP_IDS
CV = CONVEYOR_ID


def _conveyor_table() -> ShieldTable:
    model = ConveyorModel()
    return build_shield(model, solve(model), 4)


CONVEYOR_TABLE = _conveyor_table()


def shielded(
    crusher_table: ShieldTable,
    pump_table: ShieldTable,
    commands: list[Command] | None = None,
    **kw: object,
) -> tuple[ShieldedProposer, ScriptedProposer]:
    inner = ScriptedProposer(commands or [])
    kw.setdefault("conveyor", CONVEYOR_TABLE)
    return ShieldedProposer(inner, crusher_table, pump_table, StubMapper(), **kw), inner  # type: ignore[arg-type]


def running(lo: int, hi: int) -> StubView:
    return StubView(CRUSHER_ID, "RUN", set_lo=lo, set_hi=hi)


class TestCrusher:
    def test_a_safe_silence_changes_nothing(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == [] and (s.overrides, s.admitted) == (0, 0)

    def test_silence_at_a_possibly_critical_bin_is_replaced_by_stop(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        out = s.propose(observation(running(2, 3), quiet_pump(A), quiet_pump(B)))
        assert out == [Command(CRUSHER_ID, Cmd.STOP, source="shield")]
        assert (s.overrides, s.admitted, s.uncovered) == (1, 0, 0)

    def test_an_allowed_command_is_returned_as_it_was(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        start = Command(
            CRUSHER_ID, Cmd.START, arg="x", source="planner"
        )  # no-op on a running asset
        s, _ = shielded(crusher_tables[4], pump_tables[4], [start])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == [start] and out[0] is start
        assert (s.overrides, s.admitted) == (0, 1)

    def test_an_unsafe_command_is_replaced_by_the_first_command_in_priority_order(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        begin = Command(CRUSHER_ID, Cmd.BEGIN, source="careless")  # work on an energised asset
        s, _ = shielded(crusher_tables[4], pump_tables[4], [begin])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == [Command(CRUSHER_ID, Cmd.STOP, source="shield")]  # STOP outranks NOOP
        assert (s.overrides, s.admitted) == (1, 0)

    def test_a_substituted_noop_is_the_absence_of_a_command(
        self, crusher: CrusherJobModel, crusher_win: Winning, pump_tables: dict[int, ShieldTable]
    ) -> None:
        # regression: with the default priority STOP outranks NOOP, so this branch needs a table
        # that puts NOOP first; the shield must then send nothing rather than a NOOP command
        order = ("NOOP", *(c for c in crusher.commands() if c != "NOOP"))
        table = build_shield(crusher, crusher_win, 4, priority=order)
        s, _ = shielded(table, pump_tables[4], [Command(CRUSHER_ID, Cmd.BEGIN)])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == [] and (s.overrides, s.admitted) == (1, 0)

    def test_someone_inside_with_the_alarm_on_can_only_be_got_out(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        view = StubView(
            CRUSHER_ID, "WORK", "VALID", occupied=True, gas_age_q=0, alarm=True, set_lo=1, set_hi=1
        )
        test = Command(CRUSHER_ID, Cmd.TEST)
        s, _ = shielded(crusher_tables[4], pump_tables[4], [test])
        out = s.propose(observation(view, quiet_pump(A), quiet_pump(B)))
        assert out == [Command(CRUSHER_ID, Cmd.EXIT, source="shield")]

    def test_a_clean_entry_is_admitted_and_a_late_one_is_not(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        enter = Command(CRUSHER_ID, Cmd.ENTER)
        ready = StubView(CRUSHER_ID, "WORK", "VALID", gas_age_q=0, set_lo=1, set_hi=2)
        stale = StubView(CRUSHER_ID, "WORK", "VALID", gas_age_q=2, set_lo=1, set_hi=2)
        s, _ = shielded(crusher_tables[4], pump_tables[4], [enter])
        assert s.propose(observation(ready, quiet_pump(A), quiet_pump(B))) == [enter]
        assert s.propose(observation(stale, quiet_pump(A), quiet_pump(B))) != [enter]
        assert (s.admitted, s.overrides) == (1, 1)

    def test_surplus_commands_to_one_component_are_dropped_and_counted(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        start, stop = Command(CRUSHER_ID, Cmd.START), Command(CRUSHER_ID, Cmd.STOP)
        s, _ = shielded(crusher_tables[4], pump_tables[4], [start, stop])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == [start]
        assert (s.overrides, s.admitted) == (1, 1)

    def test_other_assets_and_planning_commands_pass_through_uncounted(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        keep = [
            Command("BM-301", Cmd.START),  # an asset the shield has no game for
            Command(CRUSHER_ID, PlanCmd.RAISE_WO, arg="LINER_WEAR"),
            Command(A, PlanCmd.ORDER, arg="P-1"),
            Command(CV, PlanCmd.INSPECT),
        ]
        s, _ = shielded(crusher_tables[4], pump_tables[4], keep)
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == keep and (s.overrides, s.admitted) == (0, 0)

    def test_a_command_outside_the_abstract_alphabet_is_never_admitted(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4], [Command(CRUSHER_ID, Cmd.DERATE)])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert Cmd.DERATE not in [c.cmd for c in out] and s.overrides == 1

    @pytest.mark.parametrize("asset", [CRUSHER_ID, A, B, CV])
    def test_a_plain_string_command_on_a_gated_asset_is_dropped_and_counted(
        self,
        crusher_tables: dict[int, ShieldTable],
        pump_tables: dict[int, ShieldTable],
        asset: str,
    ) -> None:
        """P12g, G6: the gate must not rest on ``Command.__post_init__``. A command whose ``cmd`` is
        a plain string (which compares equal to the StrEnum member, so the plant would run it) is
        built here by getting round the constructor; before the fix it went through unchanged."""
        forged = Command(asset, Cmd.BEGIN, source="forged")
        object.__setattr__(forged, "cmd", "BEGIN")
        s, _ = shielded(crusher_tables[4], pump_tables[4], [forged])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert forged not in out and all(c.cmd != "BEGIN" for c in out)
        assert s.overrides >= 1 and s.admitted == 0

    def test_a_plain_string_command_on_an_asset_outside_the_games_is_not_this_gates_business(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        # the constructor refuses it everywhere (tests/core/test_types.py); the gate speaks only
        # for the assets it has a game for
        forged = Command("BM-301", Cmd.START)
        object.__setattr__(forged, "cmd", "START")
        s, _ = shielded(crusher_tables[4], pump_tables[4], [forged])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == [forged] and s.overrides == 0

    def test_a_plain_string_command_cannot_be_built_at_all(self) -> None:
        with pytest.raises(TypeError, match="Cmd or a PlanCmd"):
            Command(CRUSHER_ID, "BEGIN")  # type: ignore[arg-type]

    def test_a_missing_view_is_an_error_not_a_pass(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        obs = observation(running(0, 0), quiet_pump(A), quiet_pump(B))
        with pytest.raises(ValueError, match=CRUSHER_ID):
            s.propose(type(obs)({A: obs.assets[A], B: obs.assets[B]}))
        with pytest.raises(ValueError, match=A):
            s.propose(type(obs)({CRUSHER_ID: obs.assets[CRUSHER_ID], B: obs.assets[B]}))
        with pytest.raises(ValueError, match=CV):
            s.propose(
                type(obs)({k: v for k, v in obs.assets.items() if k != CV})  # type: ignore[call-arg]
            )


class TestPerceptionSets:
    def test_a_wider_set_costs_throughput_not_safety(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        # a crusher that is in fact at bin 0 is stopped only when the set admits bin 3
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        narrow = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        wide = s.propose(observation(running(0, 3), quiet_pump(A), quiet_pump(B)))
        assert narrow == [] and wide == [Command(CRUSHER_ID, Cmd.STOP, source="shield")]

    def test_a_fired_monitor_widens_every_set(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        fired = False
        s, _ = shielded(crusher_tables[4], pump_tables[4], monitor_fired=lambda: fired)
        obs = observation(running(0, 0), quiet_pump(A), quiet_pump(B))
        assert s.propose(obs) == []
        fired = True
        out = s.propose(obs)
        assert Command(CRUSHER_ID, Cmd.STOP, source="shield") in out
        # both pumps may now be critical at once, so one of them is stopped as well
        assert [c.cmd for c in out if c.asset_id in PUMP_IDS] == [Cmd.STOP]

    def test_without_a_callable_the_observation_flag_decides(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        obs = observation(running(0, 0), quiet_pump(A), quiet_pump(B), fired=True)
        out = s.propose(obs)
        assert Command(CRUSHER_ID, Cmd.STOP, source="shield") in out

    @pytest.mark.parametrize("bad", [(3, 1), (-1, 2), (0, 4)])
    def test_a_malformed_set_is_read_as_the_whole_range(
        self,
        bad: tuple[int, int],
        crusher_tables: dict[int, ShieldTable],
        pump_tables: dict[int, ShieldTable],
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        out = s.propose(observation(running(*bad), quiet_pump(A), quiet_pump(B)))
        assert out == [Command(CRUSHER_ID, Cmd.STOP, source="shield")]

    def test_an_empty_allowed_set_falls_back_to_the_first_priority_and_is_counted(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        # a running crusher with someone inside at bin 3 is outside the winning region
        view = StubView(CRUSHER_ID, "RUN", "NONE", occupied=True, gas_age_q=2, set_lo=3, set_hi=3)
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        out = s.propose(observation(view, quiet_pump(A), quiet_pump(B)))
        assert out == [Command(CRUSHER_ID, Cmd.STOP, source="shield")]
        assert (s.overrides, s.uncovered) == (1, 1)


class TestPumps:
    def test_a_pump_under_maintenance_pins_the_other_when_it_may_fail(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        out = s.propose(
            observation(running(0, 0), quiet_pump(A, "WORK"), quiet_pump(B, "RUN", 3, 3))
        )
        # only the pump that must change is touched: A is left alone (NOOP), B is stopped
        assert out == [Command(B, Cmd.STOP, source="shield")]
        assert (s.overrides, s.admitted) == (1, 0)

    def test_the_pump_command_that_was_fine_survives_the_substitution(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        stop_a = Command(A, Cmd.STOP, source="planner")  # right: A may fail
        begin_b = Command(B, Cmd.BEGIN, source="careless")  # wrong: B is running
        s, _ = shielded(crusher_tables[4], pump_tables[4], [stop_a, begin_b])
        out = s.propose(
            observation(running(0, 0), quiet_pump(A, "RUN", 3, 3), quiet_pump(B, "RUN", 0, 0))
        )
        assert out[0] is stop_a and out[1:] == [Command(B, Cmd.STOP, source="shield")]
        assert s.overrides == 1

    def test_locking_a_stopped_pump_is_downgraded_to_a_stop_while_the_other_may_fail(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        # locked, A would no longer count as available (ADR 0003 A12.2); stopped, it still does
        lock = Command(A, Cmd.LOCK, source="planner")
        s, _ = shielded(crusher_tables[4], pump_tables[4], [lock])
        out = s.propose(
            observation(running(0, 0), quiet_pump(A, "STOP"), quiet_pump(B, "RUN", 3, 3))
        )
        assert out == [Command(A, Cmd.STOP, source="shield")]
        assert s.overrides == 1

    def test_a_pump_may_be_locked_while_the_other_cannot_fail(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        lock = Command(A, Cmd.LOCK)
        s, _ = shielded(crusher_tables[4], pump_tables[4], [lock])
        out = s.propose(observation(running(0, 0), quiet_pump(A, "STOP"), quiet_pump(B)))
        assert out == [lock] and s.admitted == 1 and s.overrides == 0

    def test_locking_the_second_pump_is_refused_whatever_the_first_one_is_doing(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        lock = Command(B, Cmd.LOCK)
        for first in ("ISO", "LOCKED", "WORK", "FAIL"):  # none of them is available
            s, _ = shielded(crusher_tables[4], pump_tables[4], [lock])
            out = s.propose(observation(running(0, 0), quiet_pump(A, first), quiet_pump(B, "STOP")))
            assert lock not in out, first
            assert s.overrides == 1, first

    def test_a_locked_pump_cannot_be_started_and_its_neighbour_is_pinned(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        """Seed 1167 in miniature: PU-201 locked, PU-202 at the top bin may fail next."""
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        out = s.propose(
            observation(running(0, 0), quiet_pump(A, "LOCKED"), quiet_pump(B, "RUN", 3, 3))
        )
        assert out == [Command(B, Cmd.STOP, source="shield")]
        start = Command(A, Cmd.START)  # energising a locked pump is a violation of its own
        s2, _ = shielded(crusher_tables[4], pump_tables[4], [start])
        assert start not in s2.propose(
            observation(running(0, 0), quiet_pump(A, "LOCKED"), quiet_pump(B, "STOP"))
        )

    def test_a_substituted_lock_or_prove_goes_out_under_its_own_name(
        self, crusher_tables: dict[int, ShieldTable], pumps: PumpPairModel, pumps_win: Winning
    ) -> None:
        first = ("LOCK|NOOP", *(c for c in pumps.commands() if c != "LOCK|NOOP"))
        table = build_shield(pumps, pumps_win, 4, priority=first)
        s, _ = shielded(
            crusher_tables[4], table, [Command(A, Cmd.BEGIN)]
        )  # BEGIN on a stopped pump
        out = s.propose(observation(running(0, 0), quiet_pump(A, "STOP"), quiet_pump(B, "STOP")))
        assert out == [Command(A, Cmd.LOCK, source="shield")]
        first = ("PROVE|NOOP", *(c for c in pumps.commands() if c != "PROVE|NOOP"))
        table = build_shield(pumps, pumps_win, 4, priority=first)
        s, _ = shielded(crusher_tables[4], table, [Command(A, Cmd.BEGIN)])  # BEGIN on a locked one
        out = s.propose(observation(running(0, 0), quiet_pump(A, "LOCKED"), quiet_pump(B, "STOP")))
        assert out == [Command(A, Cmd.PROVE, source="shield")]

    def test_two_commands_to_one_pump_keep_the_first(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        first, second = Command(A, Cmd.STOP), Command(A, Cmd.START)
        s, _ = shielded(crusher_tables[4], pump_tables[4], [first, second])
        out = s.propose(observation(running(0, 0), quiet_pump(A), quiet_pump(B)))
        assert out == [first] and s.overrides == 1


def belt(op: str, lo: int = 0, hi: int = 0) -> StubView:
    return StubView(CV, op, set_lo=lo, set_hi=hi)


class TestConveyor:
    """ADR 0003 A12.3: CV-101 is gated by its own table (P12a F3: 41 plant P1 events in 200
    episodes behind the shield, a careless BEGIN or START on a belt the shield did not look at)."""

    def propose(
        self,
        crusher_tables: dict[int, ShieldTable],
        pump_tables: dict[int, ShieldTable],
        op: str,
        commands: list[Command],
        lo: int = 0,
        hi: int = 0,
    ) -> tuple[list[Command], ShieldedProposer]:
        s, _ = shielded(crusher_tables[4], pump_tables[4], commands)
        obs = observation(running(0, 0), quiet_pump(A), quiet_pump(B), conveyor=belt(op, lo, hi))
        return s.propose(obs), s

    def test_a_careless_begin_on_a_running_belt_is_replaced(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        begin = Command(CV, Cmd.BEGIN, source="careless")
        out, s = self.propose(crusher_tables, pump_tables, "RUN", [begin])
        assert out == [Command(CV, Cmd.STOP, source="shield")]
        assert (s.overrides, s.admitted) == (1, 0)

    @pytest.mark.parametrize("op", ["RUN", "STOP", "LOCKED", "FAIL"])
    def test_begin_is_refused_wherever_the_isolation_is_not_proven(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable], op: str
    ) -> None:
        begin = Command(CV, Cmd.BEGIN)
        out, s = self.propose(crusher_tables, pump_tables, op, [begin])
        assert begin not in out and s.overrides == 1

    @pytest.mark.parametrize("op", ["LOCKED", "ISO", "WORK"])
    def test_a_careless_start_under_a_lock_or_with_work_open_is_refused(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable], op: str
    ) -> None:
        start = Command(CV, Cmd.START, source="careless")
        out, s = self.propose(crusher_tables, pump_tables, op, [start])
        assert start not in out and s.overrides == 1

    def test_what_is_in_order_passes_untouched_and_is_counted_as_admitted(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        for op, cmd in (
            ("STOP", Cmd.START),
            ("RUN", Cmd.STOP),
            ("STOP", Cmd.LOCK),
            ("LOCKED", Cmd.PROVE),
            ("ISO", Cmd.BEGIN),
            ("WORK", Cmd.COMPLETE),
            ("ISO", Cmd.UNISO),
            ("RUN", Cmd.ISSUE),  # a plant command the game has no name for: the abstract NOOP
        ):
            c = Command(CV, cmd, source="planner")
            out, s = self.propose(crusher_tables, pump_tables, op, [c])
            assert out == [c] and (s.overrides, s.admitted) == (0, 1), (op, cmd)

    def test_silence_is_fine_and_a_belt_command_never_touches_the_other_assets(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        out, s = self.propose(crusher_tables, pump_tables, "RUN", [])
        assert out == [] and (s.overrides, s.admitted) == (0, 0)
        both = [Command(CV, Cmd.BEGIN), Command(CRUSHER_ID, Cmd.BEGIN)]
        out, s = self.propose(crusher_tables, pump_tables, "RUN", both)
        assert {c.asset_id for c in out} == {CV, CRUSHER_ID} and s.overrides == 2

    def test_the_belts_health_set_does_not_change_what_is_allowed(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        """Its failure is no violation: a wide set costs nothing, a narrow one buys nothing."""
        begin = Command(CV, Cmd.BEGIN)
        results = {
            tuple(self.propose(crusher_tables, pump_tables, "RUN", [begin], lo, hi)[0])
            for lo, hi in ((0, 0), (3, 3), (0, 3), (1, 2))
        }
        assert results == {(Command(CV, Cmd.STOP, source="shield"),)}

    def test_a_substitution_follows_the_conveyor_tables_own_priority(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        """Regression (break-it pass R02): the crusher's priority order ranks the shared commands
        the same way, so a belt gated by the wrong table's priority looked right. A table that puts
        LOCK first must win over STOP."""
        model = ConveyorModel()
        order = ("LOCK", *(c for c in model.commands() if c != "LOCK"))
        table = build_shield(model, solve(model), 4, priority=order)
        s, _ = shielded(crusher_tables[4], pump_tables[4], [Command(CV, Cmd.BEGIN)], conveyor=table)
        obs = observation(running(0, 0), quiet_pump(A), quiet_pump(B), conveyor=belt("RUN"))
        assert s.propose(obs) == [Command(CV, Cmd.LOCK, source="shield")]

    def test_two_commands_to_the_belt_keep_the_first(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        first, second = Command(CV, Cmd.STOP), Command(CV, Cmd.START)
        out, s = self.propose(crusher_tables, pump_tables, "RUN", [first, second])
        assert out == [first] and s.overrides == 1

    def test_the_conveyor_table_is_required_and_comes_with_its_model(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        with pytest.raises(TypeError, match="conveyor"):
            ShieldedProposer(  # type: ignore[call-arg]
                ScriptedProposer([]), crusher_tables[4], pump_tables[4], StubMapper()
            )
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        assert s.conveyor is CONVEYOR_TABLE and s.conveyor.model_parameters() == {
            "kind": "conveyor"
        }


class TestLifecycle:
    def test_reset_clears_the_counters_and_forwards_the_seed(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, inner = shielded(crusher_tables[4], pump_tables[4])
        s.propose(observation(running(2, 3), quiet_pump(A), quiet_pump(B)))
        assert s.overrides == 1
        s.reset()
        assert (s.overrides, s.admitted, s.uncovered) == (0, 0, 0) and inner.resets == []
        s.reset(7)
        assert inner.resets == [7]

    def test_the_name_says_what_it_wraps(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        s, _ = shielded(crusher_tables[4], pump_tables[4])
        assert s.name == "shield(scripted)"

    def test_the_models_are_rebuilt_from_the_tables(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        # a table for another retest period: its gas range is longer, and the runtime follows it
        slow = CrusherJobModel(retest_q=8)
        table = build_shield(slow, solve(slow), 4)
        inner = ScriptedProposer([])
        s = ShieldedProposer(
            inner, table, pump_tables[4], StubMapper(retest_q=8), conveyor=CONVEYOR_TABLE
        )
        view = StubView(CRUSHER_ID, "WORK", "VALID", occupied=True, gas_age_q=4, set_lo=0, set_hi=0)
        # 4 quarters since the test: fine under an 8-quarter period, so silence is allowed
        assert s.propose(observation(view, quiet_pump(A), quiet_pump(B))) == []
        assert s.overrides == 0


class TestHostileLoop:
    """The runtime wrapped round the abstract models, against a random proposer."""

    def test_no_violation_with_the_shield_and_many_without(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        plant = AbstractPlant(rng_for(1, "plant"))
        s = ShieldedProposer(
            RandomProposer(rng_for(1, "proposer")),
            crusher_tables[4],
            pump_tables[4],
            plant.mapper,
            conveyor=CONVEYOR_TABLE,
        )
        for _ in range(15_000):
            plant.apply(s.propose(plant.observe()))
        assert plant.violations == 0
        assert s.overrides > 3_000 and s.admitted > 1_000 and s.uncovered == 0

        bare = AbstractPlant(rng_for(1, "plant"))
        proposer = RandomProposer(rng_for(1, "proposer"))
        for _ in range(5_000):
            bare.apply(proposer.propose(bare.observe()))
        assert bare.violations > 500

    def test_the_run_is_deterministic(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        def run() -> tuple[int, int, tuple[object, object, object]]:
            plant = AbstractPlant(rng_for(5, "plant"))
            s = ShieldedProposer(
                RandomProposer(rng_for(5, "proposer")),
                crusher_tables[2],
                pump_tables[2],
                plant.mapper,
                conveyor=CONVEYOR_TABLE,
            )
            for _ in range(2_000):
                plant.apply(s.propose(plant.observe()))
            return s.overrides, s.admitted, (plant.crusher, plant.pumps, plant.conveyor)

        assert run() == run()

    def test_the_belt_alone_is_what_the_hostile_loop_loses_without_its_table(
        self, crusher_tables: dict[int, ShieldTable], pump_tables: dict[int, ShieldTable]
    ) -> None:
        """P12a F3 as a sabotage: leave CV-101 outside the shield (a table that allows
        everything) and the stand-in plant violates P1 on the belt alone."""
        model = ConveyorModel()
        everything = ShieldTable(
            model.name,
            {(o, hs): frozenset(model.commands()) for (o, hs) in CONVEYOR_TABLE.entries},
            CONVEYOR_TABLE.priority,
            CONVEYOR_TABLE.meta,
        )
        plant = AbstractPlant(rng_for(2, "plant"))
        s = ShieldedProposer(
            RandomProposer(rng_for(2, "proposer")),
            crusher_tables[4],
            pump_tables[4],
            plant.mapper,
            conveyor=everything,
        )
        for _ in range(8_000):
            plant.apply(s.propose(plant.observe()))
        assert plant.violations > 100  # the belt's own, with the crusher and pumps shielded
