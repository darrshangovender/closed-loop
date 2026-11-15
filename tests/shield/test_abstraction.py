"""The three abstract games: sizes, hand-written rule cases, and invariants over every transition.

ADR 0003 A12 changed the pump pair (a locked pump is a state and is not available, eight commands
a pump), added the conveyor game, and gave the crusher a permit that is spent at 12 hours from its
issue; the cases that pinned the older relation say so where they were rewritten.
"""

from __future__ import annotations

import importlib.util
import sys
from itertools import product
from pathlib import Path

import pytest

import closed_loop.shield.abstraction as abstraction
from closed_loop.shield.abstraction import (
    ABSTRACTION_VERSION,
    CRUSHER_COMMANDS,
    PUMP_COMMANDS,
    PUMP_OPS,
    VIOL,
    ConveyorModel,
    CrusherJobModel,
    PumpPairModel,
    model_from_parameters,
)
from closed_loop.shield.game import Winning, solve

# (op, h, permit, occ, gas, alarm, soon), retest_q = 2 so gas runs 1..3 and 3 is stale
SpikeState = tuple[str, int, str, int, int, int, int]

RUN_H2: SpikeState = ("RUN", 2, "NONE", 0, 3, 0, 0)
WORK_READY: SpikeState = ("WORK", 1, "VALID", 0, 1, 0, 0)  # ENTER is legal here


def outcomes(model: CrusherJobModel, s: SpikeState, c: str) -> frozenset:
    return model.step(s, c)


class TestSizes:
    def test_crusher_counts(self, crusher: CrusherJobModel) -> None:
        states = list(crusher.states())
        assert len(states) == 1728 == len(set(states))
        assert tuple(crusher.commands()) == CRUSHER_COMMANDS
        assert len(CRUSHER_COMMANDS) == 14
        assert CRUSHER_COMMANDS[:3] == ("NOOP", "START", "STOP")
        assert "DERATE" not in CRUSHER_COMMANDS
        assert crusher.initial() in set(states)

    def test_pump_counts(self, pumps: PumpPairModel) -> None:
        """ADR 0003 A12.2: six ops a pump (LOCKED joins), 24 states a pump, 576 pair states, eight
        commands a pump, 64 pair commands (it was 400, 49 and seven, with ISOLATE)."""
        states = list(pumps.states())
        assert len(states) == 576 == len(set(states))
        assert len(pumps.commands()) == 64 == len(set(pumps.commands()))
        assert PUMP_OPS == ("RUN", "STOP", "LOCKED", "ISO", "WORK", "FAIL")
        assert PUMP_COMMANDS == (
            "NOOP", "START", "STOP", "LOCK", "PROVE", "BEGIN", "COMPLETE", "UNISO",
        )  # fmt: skip
        assert "ISOLATE" not in PUMP_COMMANDS
        assert pumps.initial() == ("RUN", 0, "RUN", 0)
        assert "NOOP|NOOP" in pumps.commands() and "LOCK|PROVE" in pumps.commands()

    def test_conveyor_counts(self, conveyor: ConveyorModel) -> None:
        states = list(conveyor.states())
        assert len(states) == 24 == len(set(states))
        assert tuple(conveyor.commands()) == PUMP_COMMANDS
        assert conveyor.initial() == ("RUN", 0) and conveyor.initial() in set(states)
        assert {s[0] for s in states} == set(PUMP_OPS)

    def test_the_version_names_the_relation_a12_changed(self) -> None:
        assert ABSTRACTION_VERSION == "2"

    def test_retest_q_sets_the_gas_range(self) -> None:
        stale = CrusherJobModel(retest_q=8)
        states = list(stale.states())
        assert len(states) == 6 * 4 * 3 * 2 * 9 * 2 * 2
        assert stale.initial() == ("RUN", 0, "NONE", 0, 9, 0, 0)
        assert max(s[4] for s in states) == 9  # type: ignore[index]

    @pytest.mark.parametrize("bad", [0, -1, 2.0, "2", True, None])
    def test_retest_q_is_validated(self, bad: object) -> None:
        with pytest.raises(ValueError, match="retest_q"):
            CrusherJobModel(bad)  # type: ignore[arg-type]

    def test_observable_drops_only_the_hidden_bin(
        self, crusher: CrusherJobModel, pumps: PumpPairModel
    ) -> None:
        assert crusher.observable(("WORK", 3, "VALID", 1, 2, 1, 0)) == ("WORK", "VALID", 1, 2, 1, 0)
        assert crusher.health(("WORK", 3, "VALID", 1, 2, 1, 0)) == 3
        assert pumps.observable(("RUN", 1, "WORK", 3)) == ("RUN", "WORK")
        assert pumps.observable(("LOCKED", 1, "FAIL", 3)) == ("LOCKED", "FAIL")
        assert pumps.health(("RUN", 1, "WORK", 3)) == (1, 3)
        # each observable state has exactly the four values of the hidden bin behind it
        counts: dict[object, int] = {}
        for s in crusher.states():
            counts[crusher.observable(s)] = counts.get(crusher.observable(s), 0) + 1
        assert set(counts.values()) == {4}
        assert len(counts) == 432

    def test_unknown_commands_are_refused(
        self, crusher: CrusherJobModel, pumps: PumpPairModel, conveyor: ConveyorModel
    ) -> None:
        with pytest.raises(ValueError, match="DERATE"):
            crusher.step(RUN_H2, "DERATE")
        with pytest.raises(ValueError, match="ISOLATE"):
            pumps.step(
                pumps.initial(), "ISOLATE"
            )  # a joint command is needed, and ISOLATE is retired
        with pytest.raises(ValueError, match="ISOLATE"):
            pumps.step(pumps.initial(), "ISOLATE|NOOP")
        for bad in ("ISOLATE", "ENTER", "ISSUE", "NOOP|NOOP"):
            with pytest.raises(ValueError, match="conveyor"):
                conveyor.step(conveyor.initial(), bad)

    def test_observable_of_the_conveyor_drops_the_bin(self, conveyor: ConveyorModel) -> None:
        assert conveyor.observable(("ISO", 2)) == ("ISO",) and conveyor.health(("ISO", 2)) == 2
        assert len({conveyor.observable(s) for s in conveyor.states()}) == 6


class TestCrusherRules:
    """Each case is one sentence of the English rules in the module docstring."""

    def test_stop_is_executed_and_stops_the_failure_clock(self, crusher: CrusherJobModel) -> None:
        # A_exec + ordering: the stopped crusher cannot fail or degrade this tick.
        assert outcomes(crusher, ("RUN", 3, "NONE", 0, 3, 0, 0), "STOP") == {
            ("STOP", 3, "NONE", 0, 3, 0, 0),
            ("STOP", 3, "NONE", 0, 3, 0, 1),
        }

    def test_running_at_bin_2_may_rise_one_bin_and_the_announcement_may_rise(
        self, crusher: CrusherJobModel
    ) -> None:
        assert outcomes(crusher, RUN_H2, "NOOP") == {
            ("RUN", h, "NONE", 0, 3, 0, soon) for h in (2, 3) for soon in (0, 1)
        }

    def test_running_at_bin_3_may_fail_and_that_is_a_violation(
        self, crusher: CrusherJobModel
    ) -> None:
        out = outcomes(crusher, ("RUN", 3, "NONE", 0, 3, 0, 0), "NOOP")
        assert VIOL in out
        assert ("RUN", 3, "NONE", 0, 3, 0, 0) in out
        assert all(not (isinstance(t, tuple) and t[1] == 4) for t in out)

    def test_a_stopped_component_does_not_degrade(self, crusher: CrusherJobModel) -> None:
        for op in ("STOP", "LOCKED", "ISO", "WORK", "FAIL"):
            s = (op, 3, "NONE", 0, 3, 0, 0)
            assert all(t == VIOL or t[1] == 3 for t in outcomes(crusher, s, "NOOP"))

    def test_a_non_forced_command_may_be_refused_into_noop(self, crusher: CrusherJobModel) -> None:
        stopped = ("STOP", 1, "NONE", 0, 3, 0, 0)
        out = outcomes(crusher, stopped, "START")
        assert ("STOP", 1, "NONE", 0, 3, 0, 0) in out  # refused
        assert ("RUN", 1, "NONE", 0, 3, 0, 0) in out  # executed
        assert ("RUN", 2, "NONE", 0, 3, 0, 0) in out  # executed, then a bin up

    @pytest.mark.parametrize("op", ["LOCKED", "ISO", "WORK"])
    def test_start_under_lock_is_a_violation(self, crusher: CrusherJobModel, op: str) -> None:
        assert VIOL in outcomes(crusher, (op, 0, "NONE", 0, 3, 0, 0), "START")

    @pytest.mark.parametrize("op", ["RUN", "STOP", "LOCKED", "FAIL"])
    def test_begin_on_an_unproven_asset_is_a_violation(
        self, crusher: CrusherJobModel, op: str
    ) -> None:
        assert VIOL in outcomes(crusher, (op, 0, "VALID", 0, 3, 0, 0), "BEGIN")

    @pytest.mark.parametrize("permit", ["NONE", "REVAL"])
    def test_begin_without_a_valid_permit_is_a_violation(
        self, crusher: CrusherJobModel, permit: str
    ) -> None:
        assert VIOL in outcomes(crusher, ("ISO", 0, permit, 0, 3, 0, 0), "BEGIN")

    def test_begin_with_proof_and_permit_starts_work_or_is_refused(
        self, crusher: CrusherJobModel
    ) -> None:
        out = outcomes(crusher, ("ISO", 0, "VALID", 0, 3, 0, 0), "BEGIN")
        assert VIOL not in out
        assert {t[0] for t in out} == {"WORK", "ISO"}  # type: ignore[index]

    def test_lock_prove_issue_chain(self, crusher: CrusherJobModel) -> None:
        assert ("LOCKED", 0, "NONE", 0, 3, 0, 0) in outcomes(
            crusher, ("STOP", 0, "NONE", 0, 3, 0, 0), "LOCK"
        )
        assert ("ISO", 0, "NONE", 0, 3, 0, 0) in outcomes(
            crusher, ("LOCKED", 0, "NONE", 0, 3, 0, 0), "PROVE"
        )
        assert ("ISO", 0, "VALID", 0, 3, 0, 0) in outcomes(
            crusher, ("ISO", 0, "NONE", 0, 3, 0, 0), "ISSUE"
        )
        # lock on a running asset does nothing
        assert {t[0] for t in outcomes(crusher, RUN_H2, "LOCK")} == {"RUN"}  # type: ignore[index]

    def test_enter_when_every_guard_holds_is_safe_and_the_alarm_may_come_on_after(
        self, crusher: CrusherJobModel
    ) -> None:
        out = outcomes(crusher, WORK_READY, "ENTER")
        assert VIOL not in out
        inside = {t for t in out if t[3] == 1}  # type: ignore[index]
        assert ("WORK", 1, "VALID", 1, 2, 1, 0) in inside  # alarm on: next command must be EXIT
        assert ("WORK", 1, "VALID", 1, 2, 0, 1) in inside  # handover announced

    @pytest.mark.parametrize(
        "state",
        [
            ("ISO", 1, "VALID", 0, 1, 0, 0),  # not in WORK
            ("WORK", 1, "REVAL", 0, 1, 0, 0),  # permit awaiting revalidation
            ("WORK", 1, "NONE", 0, 1, 0, 0),  # no permit
            ("WORK", 1, "VALID", 0, 2, 0, 0),  # test older than the previous tick
            ("WORK", 1, "VALID", 0, 3, 0, 0),  # test stale
            ("WORK", 1, "VALID", 0, 1, 1, 0),  # alarm on
            ("WORK", 1, "VALID", 0, 1, 0, 1),  # handover this tick: permit is REVAL inside
        ],
    )
    def test_enter_with_any_guard_broken_is_a_violation(
        self, crusher: CrusherJobModel, state: SpikeState
    ) -> None:
        assert VIOL in outcomes(crusher, state, "ENTER")

    @pytest.mark.parametrize("retest_q", [3, 8])
    def test_entry_needs_a_test_from_the_previous_tick_even_when_the_period_is_longer(
        self, retest_q: int
    ) -> None:
        # regression: with retest_q = 2 an older test is caught by the stale check anyway, so the
        # explicit "gas == 1" guard only shows when the period is longer
        model = CrusherJobModel(retest_q=retest_q)
        for gas in range(2, retest_q + 1):
            assert VIOL in model.step(("WORK", 1, "VALID", 0, gas, 0, 0), "ENTER"), gas
        assert VIOL not in model.step(("WORK", 1, "VALID", 0, 1, 0, 0), "ENTER")

    def test_enter_with_someone_inside_changes_nothing(self, crusher: CrusherJobModel) -> None:
        inside: SpikeState = ("WORK", 1, "VALID", 1, 1, 0, 0)
        assert outcomes(crusher, inside, "ENTER") >= outcomes(crusher, inside, "NOOP")
        assert VIOL not in outcomes(crusher, inside, "ENTER")

    def test_exit_is_always_executed_and_breaks_the_test_chain(
        self, crusher: CrusherJobModel
    ) -> None:
        inside_with_alarm: SpikeState = ("WORK", 1, "VALID", 1, 2, 1, 1)
        out = outcomes(crusher, inside_with_alarm, "EXIT")
        assert VIOL not in out
        assert all(t[3] == 0 and t[4] == 3 for t in out)  # type: ignore[index]
        assert all(t[5] == 1 for t in out)  # the alarm stays on until a clean test

    def test_staying_inside_with_the_alarm_on_is_a_violation(
        self, crusher: CrusherJobModel
    ) -> None:
        inside_with_alarm: SpikeState = ("WORK", 1, "VALID", 1, 1, 1, 0)
        assert outcomes(crusher, inside_with_alarm, "NOOP") == {VIOL}
        assert outcomes(crusher, inside_with_alarm, "TEST") == {VIOL}

    def test_missing_the_retest_inside_is_a_violation(self, crusher: CrusherJobModel) -> None:
        assert outcomes(crusher, ("WORK", 1, "VALID", 1, 2, 0, 0), "NOOP") == {VIOL}
        # TEST may be refused like any non-forced command, and a refusal at gas 2 is a miss
        late = outcomes(crusher, ("WORK", 1, "VALID", 1, 2, 0, 0), "TEST")
        assert VIOL in late
        # with a fresh test behind it, the refusal is survivable and the test resets the count
        out = outcomes(crusher, ("WORK", 1, "VALID", 1, 1, 0, 0), "TEST")
        assert VIOL not in out
        assert {t[4] for t in out} == {1, 2}  # type: ignore[index]
        executed = {t for t in out if t[4] == 1}  # type: ignore[index]
        assert {t[5] for t in executed} == {0, 1}  # either result  # type: ignore[index]

    def test_a_handover_while_inside_voids_the_permit(self, crusher: CrusherJobModel) -> None:
        assert outcomes(crusher, ("WORK", 1, "VALID", 1, 1, 0, 1), "NOOP") == {VIOL}

    def test_a_handover_outside_asks_for_revalidation_or_the_permit_is_spent(
        self, crusher: CrusherJobModel
    ) -> None:
        """ADR 0003 A12.1: an announced step ends a permit, as REVAL or as NONE (both outcomes), and
        the next quarter may announce again (``soon`` 0 or 1, both outcomes)."""
        both = {
            ("ISO", 0, permit, 0, 3, 0, soon) for permit in ("REVAL", "NONE") for soon in (0, 1)
        }
        assert outcomes(crusher, ("ISO", 0, "VALID", 0, 3, 0, 1), "NOOP") == both
        assert outcomes(crusher, ("ISO", 0, "REVAL", 0, 3, 0, 1), "NOOP") == both
        none = outcomes(crusher, ("ISO", 0, "NONE", 0, 3, 0, 1), "NOOP")
        assert none == {("ISO", 0, "NONE", 0, 3, 0, 0), ("ISO", 0, "NONE", 0, 3, 0, 1)}

    def test_without_an_announcement_a_permit_stays_what_it_is(
        self, crusher: CrusherJobModel
    ) -> None:
        for permit in ("VALID", "REVAL", "NONE"):
            out = outcomes(crusher, ("ISO", 0, permit, 0, 3, 0, 0), "NOOP")
            assert {t[2] for t in out} == {permit}, permit  # type: ignore[index]
            assert {t[6] for t in out} == {0, 1}, permit  # type: ignore[index]

    def test_a_revalidation_that_succeeds_can_still_be_ended_by_the_announced_step(
        self, crusher: CrusherJobModel
    ) -> None:
        out = outcomes(crusher, ("ISO", 0, "REVAL", 0, 3, 0, 1), "REVAL")
        assert {t[2] for t in out} == {"REVAL", "NONE"}  # type: ignore[index]  # never VALID at the end

    def test_issue_replaces_a_spent_permit_in_work_as_well_as_in_iso(
        self, crusher: CrusherJobModel
    ) -> None:
        """0003 A12.1: ISSUE works with permit NONE in ISO or WORK, and in no other op."""
        for op in ("ISO", "WORK"):
            out = outcomes(crusher, (op, 1, "NONE", 0, 3, 0, 0), "ISSUE")
            assert (op, 1, "VALID", 0, 3, 0, 0) in out, op
        for op in ("RUN", "STOP", "LOCKED", "FAIL"):
            out = outcomes(crusher, (op, 1, "NONE", 0, 3, 0, 0), "ISSUE")
            assert all(t[2] == "NONE" for t in out), op  # type: ignore[index]

    def test_issue_with_someone_inside_changes_nothing(self, crusher: CrusherJobModel) -> None:
        """0003 A12.1, the guard: with permit NONE the permit becomes VALID when ``op`` is ISO, or
        when ``op`` is WORK and nobody is inside. Otherwise nothing changes: an executed ISSUE is a
        NOOP, and an occupied space with no valid permit is the sink (P2)."""
        for s in crusher.states():
            if s[0] == "WORK" and s[2] == "NONE" and s[3] == 1:  # type: ignore[index]
                assert outcomes(crusher, s, "ISSUE") == outcomes(crusher, s, "NOOP"), s
                assert outcomes(crusher, s, "ISSUE") == {VIOL}, s
        # and the guard is the only difference: nobody inside, the same state, ISSUE works
        assert ("WORK", 1, "VALID", 0, 3, 0, 0) in outcomes(
            crusher, ("WORK", 1, "NONE", 0, 3, 0, 0), "ISSUE"
        )
        # a held permit is not replaced and ISSUE leaves REVAL alone
        for permit in ("VALID", "REVAL"):
            out = outcomes(crusher, ("WORK", 1, permit, 0, 3, 0, 0), "ISSUE")
            assert {t[2] for t in out} == {permit}, permit  # type: ignore[index]

    def test_a_work_state_whose_permit_was_spent_is_not_a_dead_end(
        self, crusher_win: Winning
    ) -> None:
        spent = ("WORK", 1, "NONE", 0, 3, 0, 0)
        assert spent in crusher_win.states and "ISSUE" in crusher_win.safe[spent]
        # and from there the crew can test and go back in
        assert ("WORK", 1, "VALID", 0, 3, 0, 0) in crusher_win.states
        assert "TEST" in crusher_win.safe[("WORK", 1, "VALID", 0, 3, 0, 0)]
        assert "ENTER" in crusher_win.safe[("WORK", 1, "VALID", 0, 1, 0, 0)]

    def test_an_announced_step_with_someone_inside_is_a_violation_whatever_the_permit_does(
        self, crusher: CrusherJobModel
    ) -> None:
        for permit in ("VALID", "REVAL"):
            assert outcomes(crusher, ("WORK", 1, permit, 1, 1, 0, 1), "TEST") >= {VIOL}
        assert outcomes(crusher, ("WORK", 1, "VALID", 1, 1, 0, 1), "EXIT") != {VIOL}

    def test_the_docstring_no_longer_says_the_shift_covers_the_twelve_hour_limit(self) -> None:
        doc = abstraction.__doc__ or ""
        assert "covered because a shift is 8 hours" not in doc
        assert "every handover forces REVAL" not in doc
        assert "12 hours from" in doc and "without renewing" in doc

    def test_revalidation_depends_on_the_holder(self, crusher: CrusherJobModel) -> None:
        out = outcomes(crusher, ("ISO", 0, "REVAL", 0, 3, 0, 0), "REVAL")
        assert {t[2] for t in out} == {"VALID", "REVAL"}  # type: ignore[index]

    def test_the_alarm_stays_on_without_a_test(self, crusher: CrusherJobModel) -> None:
        out = outcomes(crusher, ("WORK", 1, "VALID", 0, 3, 1, 0), "NOOP")
        assert {t[5] for t in out} == {1}  # type: ignore[index]

    def test_complete_may_renew_the_component_and_is_a_violation_with_someone_inside(
        self, crusher: CrusherJobModel
    ) -> None:
        out = outcomes(crusher, ("WORK", 3, "VALID", 0, 3, 0, 0), "COMPLETE")
        assert ("ISO", 0, "VALID", 0, 3, 0, 0) in out  # renewed
        assert VIOL not in out
        assert VIOL in outcomes(crusher, ("WORK", 3, "VALID", 1, 1, 0, 0), "COMPLETE")

    def test_a_wrong_mode_completion_is_a_transition_of_the_model(
        self, crusher: CrusherJobModel
    ) -> None:
        """0003 A11 regression: the job finishes and the worn part stays (A3), at every bin."""
        for h in range(4):
            out = outcomes(crusher, ("WORK", h, "VALID", 0, 3, 0, 0), "COMPLETE")
            assert ("ISO", h, "VALID", 0, 3, 0, 0) in out, h  # not renewed
            assert ("ISO", 0, "VALID", 0, 3, 0, 0) in out, h  # renewed
            assert {t[1] for t in out if t != VIOL if t[0] == "ISO"} == {0, h}  # type: ignore[index]

    def test_the_refused_completion_and_the_two_completions_are_all_outcomes(
        self, crusher: CrusherJobModel
    ) -> None:
        out = outcomes(crusher, ("WORK", 2, "VALID", 0, 3, 0, 0), "COMPLETE")
        assert {t[0] for t in out} == {"WORK", "ISO"}  # type: ignore[index]
        assert {(t[0], t[1]) for t in out} == {("WORK", 2), ("ISO", 2), ("ISO", 0)}  # type: ignore[index]

    def test_completing_outside_work_changes_no_health(self, crusher: CrusherJobModel) -> None:
        for op in ("STOP", "LOCKED", "ISO", "FAIL"):
            out = outcomes(crusher, (op, 3, "NONE", 0, 3, 0, 0), "COMPLETE")
            assert {t[1] for t in out} == {3}, op  # type: ignore[index]

    def test_close_and_uniso_guards(self, crusher: CrusherJobModel) -> None:
        assert VIOL in outcomes(crusher, ("WORK", 0, "VALID", 0, 3, 0, 0), "CLOSE")
        assert ("ISO", 0, "NONE", 0, 3, 0, 0) in outcomes(
            crusher, ("ISO", 0, "VALID", 0, 3, 0, 0), "CLOSE"
        )
        assert VIOL in outcomes(crusher, ("ISO", 0, "VALID", 0, 3, 0, 0), "UNISO")
        assert ("STOP", 0, "NONE", 0, 3, 0, 0) in outcomes(
            crusher, ("ISO", 0, "NONE", 0, 3, 0, 0), "UNISO"
        )

    def test_a_longer_retest_period_changes_the_stale_value(self) -> None:
        slow = CrusherJobModel(retest_q=8)
        inside: SpikeState = ("WORK", 1, "VALID", 1, 8, 0, 0)  # 7 quarters since the test
        assert VIOL in slow.step(inside, "NOOP")  # next tick would be stale (9)
        assert VIOL not in slow.step(("WORK", 1, "VALID", 1, 7, 0, 0), "NOOP")
        assert all(t[4] == 9 for t in slow.step(("WORK", 1, "VALID", 1, 8, 0, 1), "EXIT"))  # type: ignore[index]


class TestSpikeSemantics:
    def test_the_spike_reading_makes_an_alarm_onset_a_violation(self) -> None:
        legacy = CrusherJobModel(spike_semantics=True)
        out = legacy.step(WORK_READY, "ENTER")
        assert VIOL in out  # the environment can always raise the alarm the moment someone is in

    def test_the_spike_reading_leaves_every_entry_unsafe(self) -> None:
        legacy = CrusherJobModel(spike_semantics=True)
        win = solve(legacy)
        assert not any("ENTER" in safe for safe in win.safe.values())

    def test_the_production_reading_admits_an_entry(self, crusher_win: Winning) -> None:
        assert "ENTER" in crusher_win.safe[WORK_READY]

    def test_the_spike_reading_keeps_issue_in_iso_and_a_handover_that_only_revalidates(
        self,
    ) -> None:
        legacy = CrusherJobModel(spike_semantics=True)
        out = legacy.step(("WORK", 1, "NONE", 0, 3, 0, 0), "ISSUE")
        assert all(t[2] == "NONE" for t in out)  # type: ignore[index]
        assert legacy.step(("ISO", 0, "VALID", 0, 3, 0, 1), "NOOP") == {
            ("ISO", 0, "REVAL", 0, 3, 0, 0)
        }


class TestASecondAnnouncementInARow:
    """A12.1 and the plant: ``soon`` is a handover next OR the permit's limit next, and the two can
    fall on consecutive quarters (a permit issued 15 quarters into a shift is spent on the last
    quarter but one of a later shift). The plant then shows ``soon`` twice in a row. P5d-a first
    implemented the text of A12 as it stood ("``soon`` becomes 0"), measured 36 unmatched
    ``crusher.soon`` ticks in 200 episodes, and the owner ruled that the game contains the
    transition: after an announced step ``soon`` becomes 0 or 1. It is the model's normal behaviour
    now; the older reading is kept below as a test double to show it never mattered to the proof."""

    class Once(CrusherJobModel):
        """The reading before the ruling: after an announced step ``soon`` becomes 0."""

        def _execute(self, s, e):  # type: ignore[no-untyped-def]
            out = super()._execute(s, e)
            if s[6] == 1:
                out = {o for o in out if o == VIOL or o[6] == 0}
            return out

    def test_every_outcome_comes_with_its_soon_flipped_twin(self, crusher: CrusherJobModel) -> None:
        """The announcement is the environment's free choice in every state, announced or not."""
        for s in crusher.states():
            for c in crusher.commands():
                out = crusher.step(s, c)
                for o in out:
                    if o != VIOL:
                        assert (*o[:6], 1 - o[6]) in out, (s, c, o)  # type: ignore[index]

    def test_the_transition_the_plant_showed_is_an_outcome(self, crusher: CrusherJobModel) -> None:
        """The first row of the 36: a VALID permit, an announced step, the plant ends it and the
        next quarter announces again."""
        got = outcomes(crusher, ("ISO", 0, "VALID", 0, 3, 0, 1), "NOOP")
        assert ("ISO", 0, "REVAL", 0, 3, 0, 1) in got and ("ISO", 0, "NONE", 0, 3, 0, 1) in got
        exit_out = outcomes(crusher, ("WORK", 0, "VALID", 1, 1, 0, 1), "EXIT")
        assert ("WORK", 0, "NONE", 0, 3, 0, 1) in exit_out

    def test_the_states_the_plant_lands_in_are_winning_ones(self, crusher_win: Winning) -> None:
        model = CrusherJobModel()
        landed = 0
        for s in crusher_win.states:
            if s[6] != 1:  # type: ignore[index]
                continue
            for c in crusher_win.safe[s]:
                for o in model.step(s, c):
                    if o != VIOL and o[6] == 1:  # type: ignore[index]
                        assert o in crusher_win.states, (s, c, o)
                        landed += 1
        assert landed > 1_000

    def test_the_old_reading_gives_the_same_region_and_safe_sets(
        self, crusher_win: Winning
    ) -> None:
        """Why the ruling cost the shield nothing: the region and every safe set are those of the
        game in which ``soon`` is 0 after an announced step. Only the relation (so the table hash)
        differs."""
        once = solve(self.Once())
        assert once.states == crusher_win.states and len(crusher_win.states) == 1693
        assert dict(once.safe) == dict(crusher_win.safe)
        assert once.relation_sha256 != crusher_win.relation_sha256

    def test_the_spike_reading_keeps_soon_at_zero_after_an_announced_step(self) -> None:
        legacy = CrusherJobModel(spike_semantics=True)
        for s in legacy.states():
            if s[6] == 1:  # type: ignore[index]
                for c in legacy.commands():
                    assert all(o == VIOL or o[6] == 0 for o in legacy.step(s, c)), (s, c)  # type: ignore[index]


class TestInvariantsOverEveryTransition:
    def test_outcomes_are_states_or_the_sink(self, crusher: CrusherJobModel) -> None:
        states = set(crusher.states())
        for s in states:
            for c in crusher.commands():
                out = crusher.step(s, c)
                assert out
                assert all(t == VIOL or t in states for t in out), (s, c)

    def test_a_phys_health_moves_at_most_one_bin_and_only_while_running(
        self, crusher: CrusherJobModel
    ) -> None:
        for s in crusher.states():
            for c in crusher.commands():
                for t in crusher.step(s, c):
                    if t == VIOL:
                        continue
                    if t[1] == s[1]:  # type: ignore[index]
                        continue
                    if t[1] == 0 and c == "COMPLETE":  # type: ignore[index]
                        assert s[0] == "WORK"  # type: ignore[index]
                        continue
                    # a bin up needs a component that is running after the command (START
                    # turns STOP into RUN before the environment moves)
                    started = c == "START" and s[0] == "STOP"  # type: ignore[index]
                    running = s[0] == "RUN" or started  # type: ignore[index]
                    assert t[1] == s[1] + 1 and running and t[0] == "RUN", (s, c, t)  # type: ignore[index]

    def test_a_failed_crusher_only_comes_from_a_failed_crusher(
        self, crusher: CrusherJobModel
    ) -> None:
        # an uncontrolled failure is the sink (P5), never a state the controller can use
        for s in crusher.states():
            for c in crusher.commands():
                if s[0] != "FAIL":  # type: ignore[index]
                    assert all(t == VIOL or t[0] != "FAIL" for t in crusher.step(s, c))  # type: ignore[index]

    def test_forced_commands_are_never_refused(self, crusher: CrusherJobModel) -> None:
        for s in crusher.states():
            for t in crusher.step(s, "EXIT"):
                assert t == VIOL or (t[3] == 0 and t[4] == 3)  # type: ignore[index]
            if s[0] == "RUN":  # type: ignore[index]
                assert all(t == VIOL or t[0] == "STOP" for t in crusher.step(s, "STOP"))  # type: ignore[index]

    def test_every_other_command_includes_the_refusal_branch(
        self, crusher: CrusherJobModel
    ) -> None:
        for s in crusher.states():
            noop = crusher.step(s, "NOOP")
            for c in crusher.commands():
                if c not in ("STOP", "EXIT", "NOOP"):
                    assert crusher.step(s, c) >= noop, (s, c)

    def test_pump_pair_health_moves_at_most_one_bin(self, pumps: PumpPairModel) -> None:
        for s in pumps.states():
            for c in pumps.commands():
                for t in pumps.step(s, c):
                    if t == VIOL:
                        continue
                    for i in (1, 3):
                        assert t[i] in (s[i], s[i] + 1, 0), (s, c, t)  # type: ignore[index]
                        if t[i] == s[i] + 1:  # type: ignore[index]
                            assert t[i - 1] == "RUN"  # type: ignore[index]  # running after the command


class TestPumpPairRules:
    def test_stopping_both_is_safe_and_deterministic(self, pumps: PumpPairModel) -> None:
        assert pumps.step(("RUN", 3, "RUN", 3), "STOP|STOP") == {("STOP", 3, "STOP", 3)}

    def test_two_running_pumps_at_bin_3_can_both_fail(self, pumps: PumpPairModel) -> None:
        out = pumps.step(("RUN", 3, "RUN", 3), "NOOP|NOOP")
        assert VIOL in out  # P4: no pump left
        assert ("RUN", 3, "RUN", 3) in out
        assert ("FAIL", 3, "RUN", 3) in out

    def test_a_pump_under_maintenance_makes_the_other_one_critical(
        self, pumps: PumpPairModel
    ) -> None:
        assert VIOL in pumps.step(("WORK", 0, "RUN", 3), "NOOP|NOOP")
        assert pumps.step(("WORK", 0, "RUN", 3), "NOOP|STOP") == {("WORK", 0, "STOP", 3)}

    def test_a_stopped_pump_counts_as_available(self, pumps: PumpPairModel) -> None:
        assert VIOL not in pumps.step(("STOP", 0, "STOP", 0), "NOOP|NOOP")

    def test_locking_both_pumps_is_a_violation(self, pumps: PumpPairModel) -> None:
        assert VIOL in pumps.step(("STOP", 0, "STOP", 0), "LOCK|LOCK")

    def test_a_locked_pump_is_not_available(self, pumps: PumpPairModel) -> None:
        """0003 A12.2: it cannot be started until the lock is removed (the plant used to count it,
        even a failed one, and the closed loop found a P4 the specification read and it did not)."""
        assert pumps.step(("LOCKED", 0, "LOCKED", 0), "NOOP|NOOP") == {VIOL}
        assert pumps.step(("LOCKED", 2, "FAIL", 3), "NOOP|NOOP") == {VIOL}
        assert pumps.step(("LOCKED", 0, "ISO", 0), "NOOP|NOOP") == {VIOL}
        assert VIOL not in pumps.step(("LOCKED", 0, "STOP", 0), "NOOP|NOOP")
        # the other pump is then critical, as when one is under maintenance
        assert VIOL in pumps.step(("LOCKED", 0, "RUN", 3), "NOOP|NOOP")
        assert pumps.step(("LOCKED", 0, "RUN", 3), "NOOP|STOP") == {("LOCKED", 0, "STOP", 3)}

    def test_each_pump_may_refuse_on_its_own(self, pumps: PumpPairModel) -> None:
        out = pumps.step(("STOP", 0, "STOP", 0), "LOCK|NOOP")
        assert out == {("LOCKED", 0, "STOP", 0), ("STOP", 0, "STOP", 0)}

    def test_lock_prove_uniso_chain(self, pumps: PumpPairModel) -> None:
        run = ("RUN", 0)
        assert ("LOCKED", 1, "RUN", 0) in pumps.step(("STOP", 1, *run), "LOCK|NOOP")
        assert ("LOCKED", 3, "RUN", 0) in pumps.step(("FAIL", 3, *run), "LOCK|NOOP")
        assert ("ISO", 1, "RUN", 0) in pumps.step(("LOCKED", 1, *run), "PROVE|NOOP")
        assert ("STOP", 1, "RUN", 0) in pumps.step(("ISO", 1, *run), "UNISO|NOOP")
        # each does nothing elsewhere
        for op in ("RUN", "LOCKED", "ISO", "WORK"):
            assert {t[0] for t in pumps.step((op, 0, *run), "LOCK|NOOP")} == {op}, op
        for op in ("RUN", "STOP", "ISO", "WORK", "FAIL"):
            assert {t[0] for t in pumps.step((op, 0, *run), "PROVE|NOOP")} == {op}, op

    @pytest.mark.parametrize(
        ("state", "cmd"),
        [
            (("LOCKED", 0, "RUN", 0), "START|NOOP"),
            (("ISO", 0, "RUN", 0), "START|NOOP"),
            (("WORK", 0, "RUN", 0), "START|NOOP"),
            (("RUN", 0, "RUN", 0), "BEGIN|NOOP"),
            (("STOP", 0, "RUN", 0), "BEGIN|NOOP"),
            (("LOCKED", 0, "RUN", 0), "BEGIN|NOOP"),
            (("FAIL", 0, "RUN", 0), "BEGIN|NOOP"),
        ],
    )
    def test_energising_under_lock_or_working_unproven_is_a_violation(
        self, pumps: PumpPairModel, state: tuple[str, int, str, int], cmd: str
    ) -> None:
        assert VIOL in pumps.step(state, cmd)

    def test_start_in_fail_and_begin_in_iso_are_not_violations_by_themselves(
        self, pumps: PumpPairModel
    ) -> None:
        assert VIOL not in pumps.step(("FAIL", 0, "RUN", 0), "START|STOP")
        assert ("WORK", 0, "STOP", 0) in pumps.step(("ISO", 0, "RUN", 0), "BEGIN|STOP")

    def test_the_old_isolate_command_is_retired(self, pumps: PumpPairModel) -> None:
        for cmd in ("ISOLATE|NOOP", "NOOP|ISOLATE", "ISOLATE|ISOLATE"):
            with pytest.raises(ValueError, match="ISOLATE"):
                pumps.step(("STOP", 0, "STOP", 0), cmd)

    def test_complete_may_renew_the_pump(self, pumps: PumpPairModel) -> None:
        assert ("ISO", 0, "RUN", 1) in pumps.step(("WORK", 3, "RUN", 0), "COMPLETE|NOOP")

    def test_a_wrong_mode_completion_leaves_the_pump_as_worn_as_it_was(
        self, pumps: PumpPairModel
    ) -> None:
        """0003 A11 regression, pump side: both outcomes exist, at every bin."""
        for h in range(4):
            out = pumps.step(("WORK", h, "RUN", 0), "COMPLETE|NOOP")
            assert ("ISO", h, "RUN", 0) in out, h  # not renewed
            assert ("ISO", 0, "RUN", 0) in out, h  # renewed
            assert ("WORK", h, "RUN", 0) in out, h  # refused

    def test_the_pump_completion_outcomes_are_refused_unrenewed_and_renewed(
        self, pumps: PumpPairModel
    ) -> None:
        out = pumps.step(("WORK", 2, "STOP", 3), "COMPLETE|NOOP")
        assert {t[:2] for t in out} == {("WORK", 2), ("ISO", 2), ("ISO", 0)}  # type: ignore[index]
        # an isolated pump is unavailable whether or not the job renewed it (P4)
        assert pumps.step(("WORK", 2, "WORK", 3), "COMPLETE|COMPLETE") == {VIOL}


class TestConveyorRules:
    """0003 A12.3: the pump's per-asset rules, no pair rule, and two VIOL rules only."""

    def test_the_only_violations_are_start_under_lock_and_work_without_proof(
        self, conveyor: ConveyorModel
    ) -> None:
        for s in conveyor.states():
            op = s[0]  # type: ignore[index]
            for c in conveyor.commands():
                # the refused branch of a command is a NOOP and never a violation; only the executed
                # branch can be one, and a running conveyor failing is no violation at all
                expected = (c == "START" and op in ("LOCKED", "ISO", "WORK")) or (
                    c == "BEGIN" and op in ("RUN", "STOP", "LOCKED", "FAIL")
                )
                assert (VIOL in conveyor.step(s, c)) == expected, (s, c)

    def test_a_failing_conveyor_is_not_a_violation(self, conveyor: ConveyorModel) -> None:
        out = conveyor.step(("RUN", 3), "NOOP")
        assert out == {("RUN", 3), ("FAIL", 3)}
        assert conveyor.step(("RUN", 2), "NOOP") == {("RUN", 2), ("RUN", 3)}

    def test_stop_and_noop_are_never_refused_and_every_other_command_may_be(
        self, conveyor: ConveyorModel
    ) -> None:
        assert conveyor.step(("RUN", 1), "STOP") == {("STOP", 1)}
        for s in conveyor.states():
            noop = conveyor.step(s, "NOOP")
            assert conveyor.step(s, "STOP") != noop or s[0] != "RUN"  # type: ignore[index]
            for c in conveyor.commands():
                if c not in ("STOP", "NOOP"):
                    assert conveyor.step(s, c) >= noop, (s, c)

    def test_the_lock_chain_and_the_unproven_work(self, conveyor: ConveyorModel) -> None:
        assert ("LOCKED", 2) in conveyor.step(("STOP", 2), "LOCK")
        assert ("LOCKED", 3) in conveyor.step(("FAIL", 3), "LOCK")
        assert ("ISO", 2) in conveyor.step(("LOCKED", 2), "PROVE")
        assert ("WORK", 2) in conveyor.step(("ISO", 2), "BEGIN")
        assert ("STOP", 2) in conveyor.step(("ISO", 2), "UNISO")
        assert VIOL in conveyor.step(("RUN", 0), "BEGIN")  # the careless BEGIN of P12a F3
        assert VIOL in conveyor.step(("WORK", 0), "START")  # and the careless START

    def test_a_wrong_mode_completion_is_an_outcome_here_too(self, conveyor: ConveyorModel) -> None:
        for h in range(4):
            out = conveyor.step(("WORK", h), "COMPLETE")
            assert {("ISO", 0), ("ISO", h), ("WORK", h)} == out, h

    def test_it_never_depends_on_the_other_assets_availability(
        self, conveyor: ConveyorModel
    ) -> None:
        # no pair rule: a locked conveyor is just a locked conveyor
        assert conveyor.step(("LOCKED", 0), "NOOP") == {("LOCKED", 0)}
        assert conveyor.step(("FAIL", 3), "NOOP") == {("FAIL", 3)}


class TestSolvedSizes:
    """The numbers of the P5d-a note, pinned: regions, observables and the mean safe set."""

    def test_the_crusher_region_is_the_one_of_the_a9_to_a11_game(
        self, crusher_win: Winning
    ) -> None:
        assert len(crusher_win.states) == 1693
        assert crusher_win.iterations == 2
        obs = {s[:1] + s[2:] for s in crusher_win.states}  # type: ignore[index]
        assert len(obs) == 432
        mean = sum(len(v) for v in crusher_win.safe.values()) / len(crusher_win.safe)
        assert round(mean, 4) == 6.2103

    def test_the_pump_region_loses_exactly_the_both_unavailable_observables(
        self, pumps: PumpPairModel, pumps_win: Winning
    ) -> None:
        assert len(pumps_win.states) == 320 and pumps_win.iterations == 2
        assert {pumps.observable(s) for s in pumps_win.states} == {
            (a, b)
            for a in PUMP_OPS
            for b in PUMP_OPS
            if not (
                a in ("LOCKED", "ISO", "WORK", "FAIL") and b in ("LOCKED", "ISO", "WORK", "FAIL")
            )
        }
        mean = sum(len(v) for v in pumps_win.safe.values()) / len(pumps_win.safe)
        assert round(mean, 4) == 39.7969

    def test_every_conveyor_state_wins(
        self, conveyor: ConveyorModel, conveyor_win: Winning
    ) -> None:
        assert set(conveyor_win.states) == set(conveyor.states()) and conveyor_win.iterations == 1
        mean = sum(len(v) for v in conveyor_win.safe.values()) / len(conveyor_win.safe)
        assert round(mean, 4) == 6.8333


class TestParameters:
    def test_models_rebuild_from_their_parameters(self) -> None:
        for model in (
            CrusherJobModel(5),
            CrusherJobModel(2, spike_semantics=True),
            PumpPairModel(),
            PumpPairModel(spike_semantics=True),
            ConveyorModel(),
        ):
            rebuilt = model_from_parameters(model.parameters())
            assert rebuilt.name == model.name
            assert type(rebuilt) is type(model)
            assert getattr(rebuilt, "spike_semantics", None) == getattr(
                model, "spike_semantics", None
            )

    def test_unknown_kind_is_refused(self) -> None:
        with pytest.raises(ValueError, match="kind"):
            model_from_parameters({"kind": "turbine"})


def _load_spike() -> object:
    path = Path(__file__).resolve().parents[2] / "docs/adr/spikes/shield_feasibility.py"
    if not path.exists():
        pytest.skip("the feasibility spike is not in this checkout")
    spec = importlib.util.spec_from_file_location("feasibility_spike_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestSpikeParity:
    """With ``spike_semantics`` the production code is the spike, line for line in effect.

    This pins the claim that the only departures of the production models from the spike are the
    ones named in their docstrings: five for the crusher (the alarm onset, P2 for every permit
    other than VALID, renewal, ISSUE in WORK, and a spent permit at a handover), and for the pump
    pair renewal (ADR 0003 A11) and the locked pump with LOCK and PROVE in place of ISOLATE (A12.2).
    """

    def test_crusher_matches_the_spike_exactly_in_spike_semantics(self) -> None:
        spike = _load_spike()
        states = list(spike.c_states())  # type: ignore[attr-defined]
        spike_w, spike_safe = spike.solve(states, spike.ACTS, spike.c_step)  # type: ignore[attr-defined]
        legacy = CrusherJobModel(spike_semantics=True)
        for s in states:
            for c in spike.ACTS:  # type: ignore[attr-defined]
                assert legacy.step(s, c) == spike.c_step(s, c), (s, c)  # type: ignore[attr-defined]
        win = solve(legacy)
        assert set(win.states) == spike_w
        assert all(set(win.safe[s]) == set(spike_safe[s]) for s in spike_w)
        assert len(states) == 1728 and len(win.states) == 1692
        assert len({legacy.observable(s) for s in win.states}) == 432
        mean_safe = sum(len(v) for v in win.safe.values()) / len(win.safe)
        assert round(mean_safe, 2) == 6.08

    def test_pump_pair_matches_the_spike_exactly_in_spike_semantics(self) -> None:
        spike = _load_spike()
        legacy = PumpPairModel(spike_semantics=True)
        for s in legacy.states():
            for a1, a2 in product(spike.PA, spike.PA):  # type: ignore[attr-defined]
                assert legacy.step(s, f"{a1}|{a2}") == spike.pair_step(s, (a1, a2)), (s, a1, a2)  # type: ignore[attr-defined]
        win = solve(legacy)
        assert len(win.states) == 256

    def test_the_pump_pair_differs_from_the_spike_only_by_the_unrenewed_completion(
        self, pumps: PumpPairModel
    ) -> None:
        """On the spike's states and the commands both games have (ISOLATE is retired; LOCK and
        PROVE do not exist in the spike), the production pair differs by renewal alone."""
        spike = _load_spike()
        shared = [a for a in spike.PA if a != "ISOLATE"]  # type: ignore[attr-defined]
        for s in product(
            ("RUN", "STOP", "ISO", "WORK", "FAIL"),
            range(4),
            ("RUN", "STOP", "ISO", "WORK", "FAIL"),
            range(4),
        ):
            for a1, a2 in product(shared, shared):
                new = pumps.step(s, f"{a1}|{a2}")
                old = spike.pair_step(s, (a1, a2))  # type: ignore[attr-defined]
                if new == old:
                    continue
                # a difference needs a pump in WORK that was told COMPLETE ...
                assert any(
                    op == "WORK" and a == "COMPLETE" for op, a in ((s[0], a1), (s[2], a2))
                ), (s, a1, a2)
                # ... and every extra outcome keeps the bin of that pump as it was
                assert old <= new, (s, a1, a2)
                for t in new - old:
                    assert t != VIOL
                    unrenewed = [
                        t[i] == s[i] and t[i - 1] == "ISO" and s[i - 1] == "WORK"  # type: ignore[index]
                        for i in (1, 3)
                    ]
                    assert any(unrenewed), (s, a1, a2, t)

    def test_the_spike_pair_is_still_five_ops_and_isolate(self) -> None:
        legacy = PumpPairModel(spike_semantics=True)
        assert len(list(legacy.states())) == 400 and len(legacy.commands()) == 49
        assert "ISOLATE|ISOLATE" in legacy.commands() and "LOCK|NOOP" not in legacy.commands()

    def test_on_a_completion_with_nobody_inside_the_crusher_differs_only_by_renewal(
        self, crusher: CrusherJobModel
    ) -> None:
        """The alarm and P2 departures need someone inside, which a completion forbids. The two
        of ADR 0003 A12.1 are about ISSUE and an announced step, so states with ``soon`` set are
        left to the next test."""
        legacy = CrusherJobModel(spike_semantics=True)
        for s in crusher.states():
            if s[3] == 1 or s[6] == 1:  # type: ignore[index]
                continue
            new, old = crusher.step(s, "COMPLETE"), legacy.step(s, "COMPLETE")
            assert old <= new, s
            extra = new - old
            if s[0] == "WORK" and s[1] != 0:  # type: ignore[index]
                assert extra, s
                assert all(t[0] == "ISO" and t[1] == s[1] for t in extra), s  # type: ignore[index]
            else:
                assert not extra, s

    def test_with_an_announced_step_the_crusher_adds_the_spent_permit_and_a_second_announcement(
        self, crusher: CrusherJobModel
    ) -> None:
        legacy = CrusherJobModel(spike_semantics=True)
        for s in crusher.states():
            if s[3] == 1 or s[6] == 0 or s[2] == "NONE":  # type: ignore[index]
                continue
            for c in ("NOOP", "TEST", "CLOSE", "EXIT"):
                new, old = crusher.step(s, c), legacy.step(s, c)
                assert old <= new, (s, c)
                for extra in new - old:
                    if extra == VIOL:
                        continue
                    # an extra outcome is an old one with the permit spent (NONE where the old had
                    # REVAL), or with ``soon`` 1 where the old had 0, or both
                    base = (*extra[:6], 0)  # type: ignore[index]
                    spent = (*extra[:2], "REVAL", *extra[3:6], 0)  # type: ignore[index]
                    assert base in old or (extra[2] == "NONE" and spent in old), (s, c, extra)  # type: ignore[index]
                    assert extra[2] == "NONE" or extra[6] == 1, (s, c, extra)  # type: ignore[index]
