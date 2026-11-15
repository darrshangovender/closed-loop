"""The reference relations, tested against the English specification and nothing else.

Every expected set below is derived by hand from ADR 0001 section 4.1 and 5, ADR 0003 A9, A11 and
A12 and the prose specification in the P5a note. No production model is imported here: these tests
say what the reference means, so a regression in the reference is caught without any comparison.
"""

from __future__ import annotations

from itertools import product

import pytest

from closed_loop.shield.reference import (
    VIOL,
    ConveyorReference,
    CrusherReference,
    PumpPairReference,
)
from tests.shield.p5b_support import (
    CRUSHER_FIELDS,
    conveyor_reference,
    crusher_reference,
    crusher_state,
    field_of,
    pump_reference,
)

C = crusher_state
BOTH = (0, 1)


def out(s: tuple, c: str) -> frozenset:
    return crusher_reference().outcomes(s, c)


def pout(s: tuple, c: str) -> frozenset:
    return pump_reference().outcomes(s, c)


def cout(s: tuple, c: str) -> frozenset:
    return conveyor_reference().outcomes(s, c)


# ------------------------------------------------------------------------------------------
# shape
# ------------------------------------------------------------------------------------------


class TestShape:
    def test_crusher_sizes_order_and_initial_state(self) -> None:
        ref = crusher_reference()
        states = ref.states()
        assert len(states) == len(set(states)) == 6 * 4 * 3 * 2 * 3 * 2 * 2 == 1728
        assert states[0] == ("RUN", 0, "NONE", 0, 1, 0, 0)
        assert states[-1] == ("FAIL", 3, "REVAL", 1, 3, 1, 1)
        assert ref.commands() == (
            "NOOP", "START", "STOP", "LOCK", "PROVE", "ISSUE", "REVAL",
            "BEGIN", "TEST", "ENTER", "EXIT", "COMPLETE", "CLOSE", "UNISO",
        )  # fmt: skip
        assert ref.initial() == ("RUN", 0, "NONE", 0, 3, 0, 0)

    def test_the_retest_period_sets_the_gas_range(self) -> None:
        for r in (1, 2, 3, 8):
            ref = CrusherReference(r)
            assert len(ref.states()) == 6 * 4 * 3 * 2 * (r + 1) * 2 * 2
            assert ref.initial() == ("RUN", 0, "NONE", 0, r + 1, 0, 0)
        with pytest.raises(ValueError):
            CrusherReference(0)

    def test_observable_drops_only_the_health_bin(self) -> None:
        ref = crusher_reference()
        s = C(op="WORK", h=2, permit="VALID", occ=1, gas=2, alarm=1, soon=1)
        assert ref.observable(s) == ("WORK", "VALID", 1, 2, 1, 1)
        assert ref.health(s) == 2

    def test_pump_sizes_order_and_initial_state(self) -> None:
        # 0003 A12.2: six ops, 24 states a pump, 576 raw pair states, eight commands a pump (64)
        ref = pump_reference()
        states = ref.states()
        assert len(states) == len(set(states)) == (6 * 4) ** 2 == 576
        assert states[0] == ("RUN", 0, "RUN", 0) and states[-1] == ("FAIL", 3, "FAIL", 3)
        cmds = ref.commands()
        assert len(cmds) == len(set(cmds)) == 8 * 8 == 64
        assert cmds[:2] == ("NOOP|NOOP", "NOOP|START") and cmds[-1] == "UNISO|UNISO"
        assert cmds[8] == "START|NOOP"
        assert "NOOP|LOCK" in cmds and "PROVE|PROVE" in cmds
        assert not any("ISOLATE" in c for c in cmds)
        assert ref.initial() == ("RUN", 0, "RUN", 0)
        s = ("ISO", 1, "WORK", 3)
        assert ref.observable(s) == ("ISO", "WORK") and ref.health(s) == (1, 3)
        assert ref.observable(("LOCKED", 0, "RUN", 2)) == ("LOCKED", "RUN")

    def test_conveyor_sizes_order_and_initial_state(self) -> None:
        # 0003 A12.3: (op, h) with the pump's six ops, 24 raw states, the pump's eight commands
        ref = conveyor_reference()
        states = ref.states()
        assert len(states) == len(set(states)) == 6 * 4 == 24
        assert states[0] == ("RUN", 0) and states[-1] == ("FAIL", 3)
        assert ref.commands() == (
            "NOOP", "START", "STOP", "LOCK", "PROVE", "BEGIN", "COMPLETE", "UNISO",
        )  # fmt: skip
        assert ref.initial() == ("RUN", 0)
        assert ref.observable(("LOCKED", 2)) == ("LOCKED",) and ref.health(("LOCKED", 2)) == 2

    def test_unknown_commands_and_malformed_states_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            crusher_reference().outcomes(C(), "DERATE")
        with pytest.raises(ValueError):
            crusher_reference().outcomes(("RUN", 0, "NONE", 0, 4, 0, 0), "NOOP")  # gas 4 at R=2
        with pytest.raises(ValueError):
            crusher_reference().outcomes(("RUN", 0), "NOOP")
        with pytest.raises(ValueError):
            pump_reference().outcomes(("RUN", 0, "RUN", 0), "NOOP|ISOLATE")  # retired by A12.2
        with pytest.raises(ValueError):
            pump_reference().outcomes(("SPARE", 0, "RUN", 0), "NOOP|NOOP")
        with pytest.raises(ValueError):
            PumpPairReference().outcomes(("RUN", 0, "RUN", 0), "NOOP|NOOP|NOOP")
        with pytest.raises(ValueError):
            conveyor_reference().outcomes(("RUN", 0), "ISOLATE")
        with pytest.raises(ValueError):
            conveyor_reference().outcomes(("SPARE", 0), "NOOP")
        with pytest.raises(ValueError):
            conveyor_reference().outcomes(("RUN", 0, 0), "NOOP")
        with pytest.raises(ValueError):
            ConveyorReference().outcomes(("RUN", 4), "NOOP")


# ------------------------------------------------------------------------------------------
# crusher: single commands, derived by hand
# ------------------------------------------------------------------------------------------


class TestCrusherCommands:
    def test_start_runs_a_stopped_crusher_or_is_refused(self) -> None:
        got = out(C(op="STOP"), "START")
        running = {("RUN", h, "NONE", 0, 3, 0, soon) for h in (0, 1) for soon in BOTH}
        refused = {("STOP", 0, "NONE", 0, 3, 0, soon) for soon in BOTH}
        assert got == running | refused

    def test_start_under_lock_is_a_violation_but_a_refused_start_is_not(self) -> None:
        for op in ("LOCKED", "ISO", "WORK"):
            got = out(C(op=op), "START")
            assert VIOL in got
            assert got - {VIOL} == {(op, 0, "NONE", 0, 3, 0, soon) for soon in BOTH}

    def test_stop_is_never_refused_and_stops_the_failure_clock(self) -> None:
        got = out(C(op="RUN", h=3), "STOP")
        assert got == {("STOP", 3, "NONE", 0, 3, 0, soon) for soon in BOTH}

    def test_a_running_crusher_at_the_critical_bin_may_fail_which_is_a_violation(self) -> None:
        got = out(C(op="RUN", h=3), "NOOP")
        assert got == {VIOL} | {("RUN", 3, "NONE", 0, 3, 0, soon) for soon in BOTH}

    def test_health_rises_by_one_while_running_and_not_otherwise(self) -> None:
        running = out(C(op="RUN", h=1), "NOOP")
        assert {field_of(s, "h") for s in running} == {1, 2}
        for op in ("STOP", "LOCKED", "ISO", "WORK", "FAIL"):
            assert {field_of(s, "h") for s in out(C(op=op, h=2), "NOOP")} == {2}

    def test_a_running_crusher_below_the_critical_bin_cannot_fail(self) -> None:
        for h in (0, 1, 2):
            got = out(C(op="RUN", h=h), "NOOP")
            assert VIOL not in got
            assert {field_of(s, "h") for s in got} == {h, h + 1}

    def test_lock_and_prove(self) -> None:
        for op in ("STOP", "FAIL"):
            assert {field_of(s, "op") for s in out(C(op=op), "LOCK")} == {"LOCKED", op}
        assert {field_of(s, "op") for s in out(C(op="RUN"), "LOCK")} == {"RUN"}
        assert {field_of(s, "op") for s in out(C(op="LOCKED"), "PROVE")} == {"ISO", "LOCKED"}
        assert {field_of(s, "op") for s in out(C(op="STOP"), "PROVE")} == {"STOP"}

    def test_begin_needs_proof_and_a_valid_permit(self) -> None:
        got = out(C(op="ISO", permit="VALID"), "BEGIN")
        assert got == {("WORK", 0, "VALID", 0, 3, 0, s) for s in BOTH} | {
            ("ISO", 0, "VALID", 0, 3, 0, s) for s in BOTH
        }
        for op, permit in [("LOCKED", "VALID"), ("RUN", "VALID"), ("STOP", "NONE"),
                           ("FAIL", "NONE"), ("ISO", "NONE"), ("ISO", "REVAL")]:  # fmt: skip
            assert VIOL in out(C(op=op, permit=permit), "BEGIN"), (op, permit)
        assert VIOL not in out(C(op="WORK", permit="VALID"), "BEGIN")

    def test_issue_in_iso_with_no_permit(self) -> None:
        assert {field_of(s, "permit") for s in out(C(op="ISO"), "ISSUE")} == {"VALID", "NONE"}
        assert {field_of(s, "permit") for s in out(C(op="LOCKED"), "ISSUE")} == {"NONE"}
        assert {field_of(s, "permit") for s in out(C(op="ISO", permit="REVAL"), "ISSUE")} == {
            "REVAL"
        }

    def test_revalidation_depends_on_the_holder_and_adds_no_outcome_beyond_a_refusal(self) -> None:
        got = out(C(op="ISO", permit="REVAL"), "REVAL")
        assert got == {("ISO", 0, p, 0, 3, 0, s) for p in ("VALID", "REVAL") for s in BOTH}
        assert out(C(op="ISO", permit="VALID"), "REVAL") == out(C(op="ISO", permit="VALID"), "NOOP")

    def test_a_test_in_work_resets_the_gas_clock_and_may_raise_the_alarm(self) -> None:
        got = out(C(op="WORK", permit="VALID"), "TEST")
        tested = {("WORK", 0, "VALID", 0, 1, a, s) for a in BOTH for s in BOTH}
        refused = {("WORK", 0, "VALID", 0, 3, 0, s) for s in BOTH}
        assert got == tested | refused

    def test_a_test_outside_work_does_nothing(self) -> None:
        for op in ("RUN", "STOP", "LOCKED", "ISO", "FAIL"):
            assert out(C(op=op, permit="VALID"), "TEST") == out(C(op=op, permit="VALID"), "NOOP")

    def test_complete_renews_the_component_or_not_when_nobody_is_inside(self) -> None:
        """0003 A11: renewal is the environment's choice, so both bins are outcomes."""
        got = out(C(op="WORK", h=2, permit="VALID"), "COMPLETE")
        assert got == (
            {("ISO", 0, "VALID", 0, 3, 0, s) for s in BOTH}  # renewed
            | {("ISO", 2, "VALID", 0, 3, 0, s) for s in BOTH}  # wrong mode: the worn part stays
            | {("WORK", 2, "VALID", 0, 3, 0, s) for s in BOTH}  # refused
        )
        assert VIOL in out(C(op="WORK", permit="VALID", occ=1, gas=1), "COMPLETE")

    def test_a_completion_at_the_lowest_bin_has_one_isolated_outcome(self) -> None:
        got = out(C(op="WORK", h=0, permit="VALID"), "COMPLETE")
        assert {field_of(t, "h") for t in got} == {0}
        assert {field_of(t, "op") for t in got} == {"ISO", "WORK"}

    def test_close_and_uniso(self) -> None:
        assert {field_of(s, "permit") for s in out(C(op="ISO", permit="VALID"), "CLOSE")} == {
            "NONE", "VALID",
        }  # fmt: skip
        assert VIOL in out(C(op="WORK", permit="VALID"), "CLOSE")
        assert {field_of(s, "op") for s in out(C(op="ISO"), "UNISO")} == {"STOP", "ISO"}
        for permit in ("VALID", "REVAL"):
            assert VIOL in out(C(op="ISO", permit=permit), "UNISO")
        assert VIOL not in out(C(op="LOCKED"), "UNISO")

    def test_exit_is_never_refused_and_spoils_the_gas_test_even_if_nobody_was_in(self) -> None:
        inside = C(op="WORK", permit="VALID", occ=1, gas=1)
        assert out(inside, "EXIT") == {("WORK", 0, "VALID", 0, 3, 0, s) for s in BOTH}
        assert out(C(op="WORK", permit="VALID", gas=1), "EXIT") == {
            ("WORK", 0, "VALID", 0, 3, 0, s) for s in BOTH
        }


# ------------------------------------------------------------------------------------------
# crusher: the entry and the occupied-state rules (0003 A9)
# ------------------------------------------------------------------------------------------


class TestEntry:
    ready = C(op="WORK", permit="VALID", gas=1)

    def test_a_clean_entry_is_safe_even_though_the_alarm_may_rise_straight_after(self) -> None:
        got = out(self.ready, "ENTER")
        inside = {("WORK", 0, "VALID", 1, 2, a, s) for a in BOTH for s in BOTH}
        refused = {("WORK", 0, "VALID", 0, 2, 0, s) for s in BOTH}
        assert got == inside | refused
        assert VIOL not in got  # the point of A9: the spike made this branch a violation

    def test_entering_with_the_alarm_on_is_a_violation(self) -> None:
        got = out(C(op="WORK", permit="VALID", gas=1, alarm=1), "ENTER")
        assert got == {VIOL} | {("WORK", 0, "VALID", 0, 2, 1, s) for s in BOTH}

    def test_entering_before_a_handover_is_a_violation_through_the_permit(self) -> None:
        # 0003 A12.1: at the handover the permit goes to REVAL or ends, and the next quarter is
        # announced or not; either permit spoils an occupant
        got = out(C(op="WORK", permit="VALID", gas=1, soon=1), "ENTER")
        refused = {("WORK", 0, p, 0, 2, 0, s) for p in ("REVAL", "NONE") for s in BOTH}
        assert got == {VIOL} | refused

    def test_entering_needs_the_test_of_the_previous_tick(self) -> None:
        assert VIOL in out(C(op="WORK", permit="VALID", gas=2), "ENTER")
        assert VIOL in out(C(op="WORK", permit="VALID", gas=3), "ENTER")

    @pytest.mark.parametrize("op", ["RUN", "STOP", "LOCKED", "ISO", "FAIL"])
    def test_entering_outside_work_is_a_violation(self, op: str) -> None:
        assert VIOL in out(C(op=op, permit="VALID", gas=1), "ENTER")

    @pytest.mark.parametrize("permit", ["NONE", "REVAL"])
    def test_entering_without_a_valid_permit_is_a_violation(self, permit: str) -> None:
        assert VIOL in out(C(op="WORK", permit=permit, gas=1), "ENTER")

    def test_entering_when_already_inside_changes_nothing(self) -> None:
        inside = C(op="WORK", permit="VALID", occ=1, gas=1)
        assert out(inside, "ENTER") == out(inside, "NOOP")


class TestOccupiedStates:
    def test_an_alarm_that_was_already_on_must_be_answered_by_exit(self) -> None:
        s = C(op="WORK", permit="VALID", occ=1, gas=1, alarm=1)
        assert out(s, "NOOP") == {VIOL}
        assert out(s, "EXIT") == {("WORK", 0, "VALID", 0, 3, 1, x) for x in BOTH}

    def test_an_alarm_that_comes_on_in_this_tick_is_not_yet_a_violation(self) -> None:
        s = C(op="WORK", permit="VALID", occ=1, gas=1)
        assert ("WORK", 0, "VALID", 1, 2, 1, 0) in out(s, "NOOP")
        assert VIOL not in out(s, "NOOP")

    def test_a_stale_gas_test_with_someone_inside_is_a_violation(self) -> None:
        s = C(op="WORK", permit="VALID", occ=1, gas=2)
        assert out(s, "NOOP") == {VIOL}
        got = out(s, "TEST")
        assert got == {VIOL} | {("WORK", 0, "VALID", 1, 1, a, x) for a in BOTH for x in BOTH}

    def test_a_handover_with_someone_inside_is_a_violation_unless_they_leave(self) -> None:
        s = C(op="WORK", permit="VALID", occ=1, gas=1, soon=1)
        assert out(s, "NOOP") == {VIOL}
        assert out(s, "EXIT") == {  # A12.1: the permit goes either way, the announcement either way
            ("WORK", 0, p, 0, 3, 0, x) for p in ("REVAL", "NONE") for x in BOTH
        }

    def test_a_permit_awaiting_revalidation_with_someone_inside_is_a_violation(self) -> None:
        s = C(op="WORK", permit="REVAL", occ=1, gas=1)
        assert out(s, "NOOP") == {VIOL}

    def test_without_a_permit_with_someone_inside_is_a_violation(self) -> None:
        assert out(C(op="WORK", permit="NONE", occ=1, gas=1), "NOOP") == {VIOL}


class TestHandover:
    def test_a_live_permit_goes_to_revalidation_or_ends_and_none_stays_none(self) -> None:
        # 0003 A12.1: both outcomes, from VALID and from REVAL alike (before A12 only REVAL), and
        # the next quarter may be announced again (before A12 the announcement was cleared)
        both = {("ISO", 0, p, 0, 3, 0, s) for p in ("REVAL", "NONE") for s in BOTH}
        assert out(C(op="ISO", permit="VALID", soon=1), "NOOP") == both
        assert out(C(op="ISO", permit="REVAL", soon=1), "NOOP") == both
        assert out(C(op="ISO", permit="NONE", soon=1), "NOOP") == {
            ("ISO", 0, "NONE", 0, 3, 0, s) for s in BOTH
        }

    def test_a_permit_issued_in_the_last_tick_is_revoked_in_the_same_tick(self) -> None:
        got = out(C(op="ISO", soon=1), "ISSUE")
        assert got == {("ISO", 0, p, 0, 3, 0, s) for p in ("REVAL", "NONE") for s in BOTH}

    def test_a_handover_may_be_announced_at_any_step(self) -> None:
        assert {field_of(s, "soon") for s in out(C(op="ISO"), "NOOP")} == {0, 1}


class TestPermitLimit:
    """0003 A12.1: a permit lives at most 12 hours from issue. The game has no clock: the permit may
    end at any announced step, and ``ISSUE`` works again once it has."""

    def test_issue_works_in_work_with_nobody_inside(self) -> None:
        got = out(C(op="WORK"), "ISSUE")
        assert got == {("WORK", 0, p, 0, 3, 0, s) for p in ("VALID", "NONE") for s in BOTH}

    def test_issue_in_work_with_someone_inside_gives_nothing_but_the_violation(self) -> None:
        # the ADR's text: in WORK only with nobody inside (the first wording left that out);
        # NONE with an occupant is the sink as a successor in any case
        assert out(C(op="WORK", occ=1, gas=1), "ISSUE") == {VIOL}

    @pytest.mark.parametrize("op", ["RUN", "STOP", "LOCKED", "FAIL"])
    def test_issue_does_nothing_outside_iso_and_work(self, op: str) -> None:
        assert out(C(op=op), "ISSUE") == out(C(op=op), "NOOP")

    @pytest.mark.parametrize("op", ["ISO", "WORK"])
    @pytest.mark.parametrize("permit", ["VALID", "REVAL"])
    def test_issue_does_nothing_while_a_permit_is_on_record(self, op: str, permit: str) -> None:
        assert out(C(op=op, permit=permit), "ISSUE") == out(C(op=op, permit=permit), "NOOP")

    def test_a_permit_can_end_with_the_work_open_and_then_be_issued_again(self) -> None:
        ended = out(C(op="WORK", permit="VALID", soon=1), "NOOP")
        assert ended == {("WORK", 0, p, 0, 3, 0, s) for p in ("REVAL", "NONE") for s in BOTH}
        assert ("WORK", 0, "NONE", 0, 3, 0, 0) in ended
        assert ("WORK", 0, "VALID", 0, 3, 0, 0) in out(("WORK", 0, "NONE", 0, 3, 0, 0), "ISSUE")

    def test_a_permit_does_not_end_unless_a_step_was_announced(self) -> None:
        for permit in ("VALID", "REVAL"):
            got = out(C(op="ISO", permit=permit), "NOOP")
            assert got == {("ISO", 0, permit, 0, 3, 0, s) for s in BOTH}

    def test_the_announcement_may_repeat_at_the_step_that_consumes_it(self) -> None:
        # a handover and a permit limit can fall on consecutive quarters (0003 A12.1)
        for permit in ("NONE", "VALID", "REVAL"):
            assert {
                field_of(t, "soon") for t in out(C(op="ISO", permit=permit, soon=1), "NOOP")
            } == {0, 1}

    def test_an_occupant_cannot_outlive_either_outcome(self) -> None:
        s = C(op="WORK", permit="VALID", occ=1, gas=1, soon=1)
        for c in ("NOOP", "TEST", "ENTER", "ISSUE", "REVAL", "STOP"):
            assert out(s, c) == {VIOL}, c


# ------------------------------------------------------------------------------------------
# crusher: laws that hold over every (state, command)
# ------------------------------------------------------------------------------------------

ALL = [(s, c) for s in crusher_reference().states() for c in crusher_reference().commands()]


def non_viol(s: tuple, c: str) -> list[tuple]:
    return [t for t in out(s, c) if t != VIOL]


class TestCrusherLaws:
    def test_every_non_violating_outcome_is_a_state_of_the_game(self) -> None:
        space = set(crusher_reference().states())
        assert all(set(non_viol(s, c)) <= space for s, c in ALL)

    def test_every_command_has_an_outcome(self) -> None:
        assert all(out(s, c) for s, c in ALL)

    def test_a_refusable_command_always_has_the_noop_outcomes_too(self) -> None:
        for s, c in ALL:
            if c not in ("NOOP", "STOP", "EXIT"):
                assert out(s, "NOOP") <= out(s, c), (s, c)

    def test_the_crusher_never_enters_the_failed_state(self) -> None:
        # a failure of the critical asset is the violation, so FAIL can only be a raw start state
        for s, c in ALL:
            if any(field_of(t, "op") == "FAIL" for t in non_viol(s, c)):
                assert field_of(s, "op") == "FAIL", (s, c)

    def test_health_only_rises_by_one_while_running_and_only_a_renewal_lowers_it(self) -> None:
        for s, c in ALL:
            h, op = field_of(s, "h"), field_of(s, "op")
            for t in non_viol(s, c):
                nh = field_of(t, "h")
                if nh == h:
                    continue
                if nh == h + 1:
                    assert op == "RUN" or (op == "STOP" and c == "START"), (s, c, t)
                else:
                    assert nh == 0 and op == "WORK" and c == "COMPLETE", (s, c, t)

    def test_a_permit_ends_only_by_close_or_at_an_announced_step_and_begins_only_by_issue(
        self,
    ) -> None:
        for s, c in ALL:
            before, soon = field_of(s, "permit"), field_of(s, "soon")
            for t in non_viol(s, c):
                after = field_of(t, "permit")
                if before != "NONE" and after == "NONE":
                    assert soon == 1 or (c == "CLOSE" and field_of(s, "op") == "ISO"), (s, c, t)
                if before == "NONE" and after != "NONE":
                    assert c == "ISSUE" and field_of(s, "op") in ("ISO", "WORK"), (s, c, t)
                    if field_of(s, "op") == "WORK":  # in WORK only with nobody inside
                        assert field_of(s, "occ") == 0, (s, c, t)
                if soon == 1:  # the announcement may repeat, the permit may not stay as it was
                    assert after in ("REVAL", "NONE"), (s, c, t)

    def test_the_alarm_is_cleared_only_by_a_test_in_work(self) -> None:
        for s, c in ALL:
            if field_of(s, "alarm") == 1:
                for t in non_viol(s, c):
                    if field_of(t, "alarm") == 0:
                        assert c == "TEST" and field_of(s, "op") == "WORK", (s, c, t)

    def test_the_gas_clock_ticks_resets_on_a_test_and_is_spoilt_by_exit(self) -> None:
        for s, c in ALL:
            gas = field_of(s, "gas")
            for t in non_viol(s, c):
                ng = field_of(t, "gas")
                if ng == 1:
                    assert c == "TEST" and field_of(s, "op") == "WORK", (s, c, t)
                else:
                    assert ng in (min(gas + 1, 3), 3), (s, c, t)

    def test_nobody_is_inside_in_a_surviving_state_unless_every_occupancy_rule_holds(self) -> None:
        for s, c in ALL:
            for t in non_viol(s, c):
                if field_of(t, "occ") == 1:
                    assert field_of(t, "permit") == "VALID", (s, c, t)
                    assert field_of(t, "gas") != 3, (s, c, t)
                    assert field_of(s, "alarm") == 0, (s, c, t)

    def test_stop_stops_and_exit_empties_the_space_whatever_the_plant_does(self) -> None:
        for s, _ in ALL:
            if field_of(s, "op") == "RUN":
                assert all(field_of(t, "op") == "STOP" for t in non_viol(s, "STOP"))
            assert all(field_of(t, "occ") == 0 for t in non_viol(s, "EXIT"))

    def test_the_observable_and_the_health_bin_rebuild_the_state(self) -> None:
        ref = crusher_reference()
        for s in ref.states():
            o, h = ref.observable(s), ref.health(s)
            assert (o[0], h, *o[1:]) == s


class TestRetestPeriod:
    """The same rules at other re-test periods: the gas range stretches, the rules do not."""

    def test_a_longer_period_keeps_a_test_valid_for_more_ticks_for_an_occupant(self) -> None:
        ref = CrusherReference(8)
        s = C(op="WORK", permit="VALID", occ=1, gas=7)
        assert VIOL not in ref.outcomes(s, "NOOP")  # 7 -> 8 is still inside the period
        assert ref.outcomes(C(op="WORK", permit="VALID", occ=1, gas=8), "NOOP") == {VIOL}

    def test_entry_still_needs_a_test_from_the_previous_tick_whatever_the_period(self) -> None:
        ref = CrusherReference(8)
        assert VIOL in ref.outcomes(C(op="WORK", permit="VALID", gas=2), "ENTER")
        assert VIOL not in ref.outcomes(C(op="WORK", permit="VALID", gas=1), "ENTER")

    def test_a_period_of_one_quarter_can_never_be_entered(self) -> None:
        # gas runs 1..2: a test is valid for the tick it came out of, and the entry tick ages it
        # to the stale value, so every executed entry is a violation
        got = CrusherReference(1).outcomes(C(op="WORK", permit="VALID", gas=1), "ENTER")
        assert VIOL in got
        assert all(field_of(t, "occ") == 0 for t in got if t != VIOL)


# ------------------------------------------------------------------------------------------
# pump pair
# ------------------------------------------------------------------------------------------


class TestPumpPair:
    def test_two_running_pumps_each_stay_or_rise(self) -> None:
        got = pout(("RUN", 0, "RUN", 0), "NOOP|NOOP")
        assert got == {("RUN", a, "RUN", b) for a in (0, 1) for b in (0, 1)}

    def test_a_pump_that_fails_is_unavailable_but_the_pair_survives_if_the_other_is_available(
        self,
    ) -> None:
        assert pout(("RUN", 3, "STOP", 0), "NOOP|NOOP") == {
            ("RUN", 3, "STOP", 0),
            ("FAIL", 3, "STOP", 0),
        }

    def test_the_second_pump_failing_with_the_first_isolated_is_a_violation(self) -> None:
        assert pout(("RUN", 3, "ISO", 0), "NOOP|NOOP") == {VIOL, ("RUN", 3, "ISO", 0)}
        # stopping it in time is safe, and STOP is never refused
        assert pout(("RUN", 3, "ISO", 0), "STOP|NOOP") == {("STOP", 3, "ISO", 0)}

    def test_locking_the_last_available_pump_is_a_violation_unless_refused(self) -> None:
        assert pout(("STOP", 0, "ISO", 0), "LOCK|NOOP") == {VIOL, ("STOP", 0, "ISO", 0)}

    def test_both_pumps_unavailable_in_any_combination_is_a_violation(self) -> None:
        for a in ("LOCKED", "ISO", "WORK", "FAIL"):
            for b in ("LOCKED", "ISO", "WORK", "FAIL"):
                assert pout((a, 0, b, 0), "NOOP|NOOP") == {VIOL}, (a, b)

    def test_a_stopped_pump_counts_as_available(self) -> None:
        for other in ("LOCKED", "ISO", "WORK", "FAIL"):
            assert VIOL not in pout(("STOP", 0, other, 0), "NOOP|NOOP")

    def test_a_locked_pump_is_not_available(self) -> None:
        # 0003 A12.2: the plant and the abstraction used to count it, even a failed one
        assert pout(("RUN", 3, "LOCKED", 0), "NOOP|NOOP") == {VIOL, ("RUN", 3, "LOCKED", 0)}
        assert pout(("RUN", 3, "LOCKED", 0), "STOP|NOOP") == {("STOP", 3, "LOCKED", 0)}
        assert pout(("LOCKED", 3, "FAIL", 3), "NOOP|NOOP") == {VIOL}
        assert pout(("LOCKED", 0, "LOCKED", 0), "NOOP|NOOP") == {VIOL}

    def test_start_on_an_unenergisable_pump_and_work_on_one_not_isolated_are_violations(
        self,
    ) -> None:
        for op in ("LOCKED", "ISO", "WORK"):
            assert VIOL in pout((op, 0, "RUN", 0), "START|NOOP")
            assert VIOL in pout(("RUN", 0, op, 0), "NOOP|START")  # the second pump is no different
        for op in ("RUN", "STOP", "LOCKED", "FAIL"):
            assert VIOL in pout((op, 0, "STOP", 0), "BEGIN|NOOP")
            assert VIOL in pout(("STOP", 0, op, 0), "NOOP|BEGIN")
        assert VIOL not in pout(("ISO", 0, "STOP", 0), "BEGIN|NOOP")
        assert VIOL not in pout(("WORK", 0, "STOP", 0), "BEGIN|NOOP")  # already open: a no-op
        assert VIOL not in pout(("FAIL", 3, "STOP", 0), "START|NOOP")  # a failed pump just stays

    def test_each_pump_is_refused_on_its_own(self) -> None:
        got = pout(("STOP", 0, "STOP", 0), "START|START")
        assert VIOL not in got
        assert {(a, b) for a, _, b, _ in got} == {
            (x, y) for x in ("RUN", "STOP") for y in ("RUN", "STOP")
        }
        assert len(got) == 3 * 3  # (RUN, 0), (RUN, 1) or (STOP, 0) for each pump

    def test_lock_takes_a_stopped_or_failed_pump_to_locked(self) -> None:
        assert pout(("FAIL", 3, "RUN", 0), "LOCK|STOP") == {
            ("LOCKED", 3, "STOP", 0),
            ("FAIL", 3, "STOP", 0),
        }
        assert VIOL not in pout(("RUN", 0, "STOP", 0), "LOCK|NOOP")
        assert {a for a, *_ in pout(("RUN", 0, "STOP", 0), "LOCK|NOOP")} == {"RUN"}  # not a RUN op
        assert {a for a, *_ in pout(("STOP", 2, "RUN", 0), "LOCK|NOOP")} == {"LOCKED", "STOP"}

    def test_prove_takes_a_locked_pump_to_isolated_and_moves_nothing_else(self) -> None:
        got = pout(("LOCKED", 2, "RUN", 0), "PROVE|NOOP")
        assert got == {(a, 2, "RUN", b) for a in ("ISO", "LOCKED") for b in (0, 1)}
        for op in ("RUN", "STOP", "ISO", "WORK", "FAIL"):
            assert pout((op, 0, "STOP", 0), "PROVE|NOOP") == pout((op, 0, "STOP", 0), "NOOP|NOOP")

    def test_isolation_is_lock_then_prove_and_isolate_is_gone(self) -> None:
        first = pout(("STOP", 1, "RUN", 0), "LOCK|NOOP")
        assert ("LOCKED", 1, "RUN", 0) in first
        second = pout(("LOCKED", 1, "RUN", 0), "PROVE|NOOP")
        assert ("ISO", 1, "RUN", 0) in second
        with pytest.raises(ValueError):
            pout(("STOP", 1, "RUN", 0), "ISOLATE|NOOP")

    def test_complete_renews_the_pump_or_not_and_uniso_stops_it(self) -> None:
        got = pout(("WORK", 3, "RUN", 0), "COMPLETE|NOOP")
        assert got == (
            {("ISO", 0, "RUN", b) for b in (0, 1)}  # renewed
            | {("ISO", 3, "RUN", b) for b in (0, 1)}  # wrong mode: the worn part stays (A11)
            | {("WORK", 3, "RUN", b) for b in (0, 1)}  # refused
        )
        assert {a for a, *_ in pout(("ISO", 1, "STOP", 0), "UNISO|NOOP")} == {"STOP", "ISO"}

    def test_a_failed_pump_stays_failed_until_it_is_locked(self) -> None:
        assert pout(("FAIL", 3, "STOP", 0), "START|NOOP") == {("FAIL", 3, "STOP", 0)}
        assert pout(("FAIL", 3, "STOP", 0), "LOCK|NOOP") == {
            ("FAIL", 3, "STOP", 0),
            ("LOCKED", 3, "STOP", 0),
        }

    def test_stop_and_noop_are_never_refused(self) -> None:
        assert pout(("RUN", 3, "STOP", 0), "STOP|NOOP") == {("STOP", 3, "STOP", 0)}


class TestConveyor:
    """0003 A12.3: the pump's rules for one asset, no pair rule, two violation rules."""

    def test_begin_on_an_asset_not_proven_isolated_is_a_violation_unless_refused(self) -> None:
        for op in ("RUN", "STOP", "LOCKED", "FAIL"):
            assert cout((op, 0), "BEGIN") == {VIOL} | cout((op, 0), "NOOP"), op

    def test_begin_on_an_isolated_asset_opens_the_work_or_is_refused(self) -> None:
        assert cout(("ISO", 1), "BEGIN") == {("WORK", 1), ("ISO", 1)}
        assert cout(("WORK", 1), "BEGIN") == {("WORK", 1)}  # already open: nothing happens

    def test_start_under_lock_isolation_or_work_is_a_violation_a_refused_start_is_not(self) -> None:
        for op in ("LOCKED", "ISO", "WORK"):
            assert cout((op, 1), "START") == {VIOL, (op, 1)}, op
        assert cout(("FAIL", 3), "START") == {("FAIL", 3)}  # a failed conveyor just stays failed

    def test_a_stopped_conveyor_starts_or_is_refused(self) -> None:
        assert cout(("STOP", 0), "START") == {("RUN", 0), ("RUN", 1), ("STOP", 0)}
        assert cout(("STOP", 3), "START") == {("RUN", 3), ("FAIL", 3), ("STOP", 3)}

    def test_a_failure_is_not_a_violation_and_there_is_no_pair_rule(self) -> None:
        assert cout(("RUN", 3), "NOOP") == {("RUN", 3), ("FAIL", 3)}
        assert cout(("FAIL", 3), "NOOP") == {("FAIL", 3)}
        assert cout(("RUN", 3), "STOP") == {("STOP", 3)}  # never refused

    def test_the_isolation_path_is_lock_prove_begin_complete_uniso(self) -> None:
        assert cout(("STOP", 1), "LOCK") == {("LOCKED", 1), ("STOP", 1)}
        assert cout(("FAIL", 3), "LOCK") == {("LOCKED", 3), ("FAIL", 3)}
        assert cout(("RUN", 1), "LOCK") == cout(("RUN", 1), "NOOP")
        assert cout(("LOCKED", 1), "PROVE") == {("ISO", 1), ("LOCKED", 1)}
        assert cout(("WORK", 3), "COMPLETE") == {("ISO", 0), ("ISO", 3), ("WORK", 3)}
        assert cout(("ISO", 2), "UNISO") == {("STOP", 2), ("ISO", 2)}

    def test_stop_and_noop_are_never_refused(self) -> None:
        assert cout(("RUN", 1), "STOP") == {("STOP", 1)}
        assert cout(("LOCKED", 1), "NOOP") == {("LOCKED", 1)}

    def test_a_violation_arises_from_exactly_the_two_controller_rules(self) -> None:
        for (op, h), c in product(conveyor_reference().states(), conveyor_reference().commands()):
            expected = (c == "START" and op in ("LOCKED", "ISO", "WORK")) or (
                c == "BEGIN" and op in ("RUN", "STOP", "LOCKED", "FAIL")
            )
            assert (VIOL in cout((op, h), c)) == expected, (op, h, c)

    def test_every_non_violating_outcome_is_a_state_and_every_command_has_one(self) -> None:
        space = set(conveyor_reference().states())
        for s, c in product(space, conveyor_reference().commands()):
            got = cout(s, c)
            assert got and got - {VIOL} <= space

    def test_the_conveyor_is_a_pump_whose_partner_is_running_and_cannot_fail(self) -> None:
        # a pump pair whose second pump is running at the lowest bin and is told to do nothing
        # has no pair rule that can bite, so pump 1's outcomes are the conveyor's
        for s, c in product(conveyor_reference().states(), conveyor_reference().commands()):
            pair = pout((*s, "RUN", 0), f"{c}|NOOP")
            assert {VIOL if t == VIOL else t[:2] for t in pair} == cout(s, c), (s, c)


PAIRS = [(s, c) for s in pump_reference().states() for c in pump_reference().commands()]


def pair_non_viol(s: tuple, c: str) -> list[tuple]:
    return [t for t in pout(s, c) if t != VIOL]


class TestPumpLaws:
    def test_every_command_has_an_outcome_and_every_outcome_is_a_state(self) -> None:
        space = set(pump_reference().states())
        for s, c in PAIRS:
            assert pout(s, c)
            assert set(pair_non_viol(s, c)) <= space

    def test_no_surviving_outcome_leaves_both_pumps_unavailable(self) -> None:
        for s, c in PAIRS:
            for t in pair_non_viol(s, c):
                assert t[0] in ("RUN", "STOP") or t[2] in ("RUN", "STOP"), (s, c, t)

    def test_each_pump_degrades_only_while_running_and_only_renewal_lowers_it(self) -> None:
        for s, c in PAIRS:
            for t in pair_non_viol(s, c):
                for i, cmd in ((0, c.split("|")[0]), (2, c.split("|")[1])):
                    op, h, nop, nh = s[i], s[i + 1], t[i], t[i + 1]
                    if nh == h:
                        continue
                    if nh == h + 1:
                        assert op == "RUN" or (op == "STOP" and cmd == "START"), (s, c, t)
                    else:
                        assert nh == 0 and op == "WORK" and cmd == "COMPLETE" and nop == "ISO"

    def test_the_two_pumps_are_interchangeable(self) -> None:
        def swap_state(s: tuple) -> tuple:
            return (s[2], s[3], s[0], s[1])

        def swap_command(c: str) -> str:
            a, b = c.split("|")
            return f"{b}|{a}"

        for s, c in PAIRS:
            mirrored = {t if t == VIOL else swap_state(t) for t in pout(s, c)}
            assert mirrored == pout(swap_state(s), swap_command(c)), (s, c)

    def test_each_pump_op_changes_only_by_the_rule_for_its_own_command(self) -> None:
        moves = {
            ("START", "STOP"): "RUN",
            ("STOP", "RUN"): "STOP",
            ("LOCK", "STOP"): "LOCKED",
            ("LOCK", "FAIL"): "LOCKED",
            ("PROVE", "LOCKED"): "ISO",
            ("BEGIN", "ISO"): "WORK",
            ("COMPLETE", "WORK"): "ISO",
            ("UNISO", "ISO"): "STOP",
        }
        for s, c in PAIRS:
            for t in pair_non_viol(s, c):
                for i, cmd in ((0, c.split("|")[0]), (2, c.split("|")[1])):
                    op, nop = s[i], t[i]
                    if nop == op:
                        continue
                    if nop == "FAIL" and s[i + 1] == 3:
                        assert op == "RUN" or (op == "STOP" and cmd == "START"), (s, c, t)
                    else:
                        assert moves.get((cmd, op)) == nop, (s, c, t)

    def test_a_pump_fails_only_from_the_top_bin_while_running(self) -> None:
        for s, c in PAIRS:
            for t in pair_non_viol(s, c):
                for i in (0, 2):
                    if t[i] == "FAIL" and s[i] != "FAIL":
                        assert s[i] in ("RUN", "STOP") and s[i + 1] == 3 and t[i + 1] == 3

    def test_refusing_either_or_both_commands_is_an_outcome(self) -> None:
        refusable = ("START", "LOCK", "PROVE", "BEGIN", "COMPLETE", "UNISO")
        for s in pump_reference().states():
            for a in refusable:
                assert pout(s, "NOOP|NOOP") <= pout(s, f"{a}|{a}")
                for b in ("NOOP", "STOP"):
                    assert pout(s, f"NOOP|{b}") <= pout(s, f"{a}|{b}")
                    assert pout(s, f"{b}|NOOP") <= pout(s, f"{b}|{a}")


def test_the_crusher_field_names_used_by_the_tests_match_the_state_layout() -> None:
    assert CRUSHER_FIELDS == ("op", "h", "permit", "occ", "gas", "alarm", "soon")
