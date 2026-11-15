"""The sabotage matrix of packet P5b (ADR 0002 P5b, as amended by ADR 0003 A9 and A12).

Two kinds of sabotage, because there are two ways a shield goes wrong:

* **Table sabotage.** The synthesis was right and someone (or something) edited the table. The
  checker walks the edited table against the independent reference and must report a VIOLATION or a
  DEADLOCK.
* **Model sabotage.** The model the shield was built from is wrong. The wrong model is solved and a
  table is built from it as if it were the truth; the checker walks that table against the *true*
  reference and the differential compares the wrong relation with the true one. A wrong model that
  only drops an environment branch can leave the table safe (the dropped branch leads to a state the
  shield handles anyway); the differential is what catches that.

Nothing here uses the production models: the sabotaged relation is compared with the true
reference, the tables come from ``p5b_support.true_table`` and ``table_from_model`` (the reference
alone), so the matrix means the same thing whatever state the production code is in. Whether the
production model equals the reference is ``test_p5b_differential``'s question.

Every sabotage has a control: the same step on the unsabotaged object is green.
"""

from __future__ import annotations

from collections.abc import Hashable
from functools import cache
from typing import Any

import pytest

from closed_loop.shield.checker import check, differential
from closed_loop.shield.reference import (
    _C,
    TOP_BIN,
    VIOL,
    ConveyorReference,
    CrusherReference,
    PumpPairReference,
    _Env,
    _Mid,
)
from tests.shield.p5b_support import (
    KINDS,
    WIDTHS,
    before_a12,
    conveyor_reference,
    crusher_reference,
    crusher_state,
    edit_entries,
    field_of,
    pump_reference,
    reference_for,
    table_from_model,
    true_table,
)

# ------------------------------------------------------------------------------------------
# sabotaged models (each breaks exactly one rule of the reference)
# ------------------------------------------------------------------------------------------


class HandoverKeepsThePermit(CrusherReference):
    """(2) handover does not void the permit (0003 A9 replaces 'remove the soon guard')."""

    def _permit_after_handover(self, permit: str, ends: bool) -> str:
        return permit


class StartUnderLockIsFine(CrusherReference):
    """(3) START on a locked, isolated or worked asset is just a no-op."""

    def _controller_violation(self, st: _C, e: str) -> bool:
        return False if e == "START" else super()._controller_violation(st, e)


class NoRefusalBranch(CrusherReference):
    """(4) the plant never turns a command into NOOP."""

    def _executions(self, c: str) -> tuple[str, ...]:
        return (c,)


class EnterIgnoresTheAlarm(CrusherReference):
    """The alarm condition removed from the entry guard only."""

    def _may_enter(self, st: _C) -> bool:
        return st.op == "WORK" and st.permit == "VALID" and st.gas == 1


class NoAlarmCheckWhenOccupied(CrusherReference):
    """The one-tick alarm check (P6) removed from occupied states only."""

    def _p6_violated(self, before: _C) -> bool:
        return False


class EnterIgnoresTheAlarmAndTheOccupiedCheckIsGone(EnterIgnoresTheAlarm, NoAlarmCheckWhenOccupied):
    """(5) both: entering an alarmed space is nowhere a violation (0003 A9)."""


class FailureIsHarmless(CrusherReference):
    """A running crusher at the critical bin never fails."""

    def _environment_stage(self, before: _C, mid: _Mid, env: _Env) -> _C | None:
        if mid.op == "RUN" and mid.h == TOP_BIN:
            env = env._replace(health_up=False)
        return super()._environment_stage(before, mid, env)


class NoStaleTestRule(CrusherReference):
    """P3 removed: an occupant may stay with a lapsed gas test."""

    def _p3_violated(self, after: _C) -> bool:
        return False


class NoPermitRule(CrusherReference):
    """P2 removed: an occupant may stay with no valid permit."""

    def _p2_violated(self, after: _C) -> bool:
        return False


class AlarmNeverRisesOnItsOwn(CrusherReference):
    """An environment branch dropped: the alarm only ever comes from a test."""

    def _env_choices(self) -> tuple[_Env, ...]:
        return tuple(e for e in super()._env_choices() if not e.alarm_onset)


class HolderIsAlwaysPresent(CrusherReference):
    """The 'holder absent' branch of REVAL dropped (it duplicates a refused REVAL)."""

    def _command_stage(self, st: _C, e: str, holder_present: bool) -> _Mid | None:
        return super()._command_stage(st, e, True)


class PumpNoRefusalBranch(PumpPairReference):
    """(4, pumps) the plant never refuses a pump command."""

    def _pump_executions(self, cmd: str) -> tuple[str, ...]:
        return (cmd,)


class IsolatedPumpCountsAsAvailable(PumpPairReference):
    """P4 weakened: an isolated pump is still 'available' (a locked one is not)."""

    def _available(self, op: str) -> bool:
        return op in ("RUN", "STOP", "ISO")


class SpentPermitStaysValid(CrusherReference):
    """(A12.1) a permit that has reached its limit cannot end at an announced step: it keeps the
    status it had (a valid permit stays VALID) instead of becoming NONE."""

    def _permit_after_handover(self, permit: str, ends: bool) -> str:
        if permit == "NONE":
            return "NONE"
        return permit if ends else "REVAL"


class SpentPermitCannotEnd(CrusherReference):
    """(A12.1, the other way to drop the outcome) at an announced step a live permit only ever goes
    to REVAL, as before A12."""

    def _permit_after_handover(self, permit: str, ends: bool) -> str:
        return "NONE" if permit == "NONE" else "REVAL"


class AnnouncementCannotRepeat(CrusherReference):
    """(A12.1, second sentence) a handover step clears the announcement: the plant cannot announce
    twice in a row, though a handover and a permit limit can fall on consecutive quarters."""

    def _announcement(self, before: _C, env: _Env) -> int:
        return 0 if before.soon == 1 else int(env.soon_next)


class IssueOnlyInIso(CrusherReference):
    """(A12.1) ISSUE does not work in WORK: a spent permit cannot be replaced with the work open."""

    def _issue_applies(self, st: _C) -> bool:
        return st.op == "ISO" and st.permit == "NONE"


class LockedPumpCountsAsAvailable(PumpPairReference):
    """(A12.2) a locked pump is available: the plant and the abstraction used to count it."""

    def _available(self, op: str) -> bool:
        return op in ("RUN", "STOP", "LOCKED")


class PumpStartOnALockedPumpIsFine(PumpPairReference):
    """(A12.2) START on a locked pump is just a no-op."""

    def _pump_violation(self, op: str, e: str) -> bool:
        return False if (e == "START" and op == "LOCKED") else super()._pump_violation(op, e)


class ConveyorBeginOnAnUnprovenAsset(ConveyorReference):
    """(A12.3) BEGIN on a conveyor whose zero energy is not proven is admitted (P1's fourth
    conjunct removed): it is a no-op in the model."""

    def _pump_violation(self, op: str, e: str) -> bool:
        return False if e == "BEGIN" else super()._pump_violation(op, e)


class ConveyorStartUnderLockIsFine(ConveyorReference):
    """(A12.3) START on a locked, isolated or worked conveyor is just a no-op."""

    def _pump_violation(self, op: str, e: str) -> bool:
        return False if e == "START" else super()._pump_violation(op, e)


class ConveyorNoRefusalBranch(ConveyorReference):
    """(A12.3, 4) the plant never refuses a conveyor command."""

    def _pump_executions(self, cmd: str) -> tuple[str, ...]:
        return (cmd,)


@cache
def wrong(cls: type[Any]) -> Any:
    """One instance per sabotaged model, so its relation is computed once."""
    return cls()


@cache
def shield_built_from(cls: type[Any], width: int = 2) -> Any:
    return table_from_model(wrong(cls), width)


def truth(cls: type[Any]) -> Any:
    if issubclass(cls, CrusherReference):
        return crusher_reference()
    return conveyor_reference() if issubclass(cls, ConveyorReference) else pump_reference()


def findings_of(cls: type[Any], width: int = 2) -> tuple[tuple[str, Hashable, object], ...]:
    ref = truth(cls)
    return check(ref, shield_built_from(cls, width), ref.initial(), width).findings


def differential_is_red(cls: type[Any]) -> bool:
    return bool(differential(truth(cls), wrong(cls), limit=1))


def visited_of(kind: str, width: int) -> set[tuple[Hashable, tuple[int, ...]]]:
    ref = reference_for(kind)
    result = check(ref, true_table(kind, width), ref.initial(), width)
    assert result.findings == ()  # the control: the unedited table is green
    return set(result.visited)


# ------------------------------------------------------------------------------------------
# the clean shield passes
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", WIDTHS)
def test_a_clean_shield_passes(kind: str, width: int) -> None:
    ref = reference_for(kind)
    result = check(ref, true_table(kind, width), ref.initial(), width)
    assert result.clean and result.reachable > 0


# ------------------------------------------------------------------------------------------
# (1) add BEGIN on a reachable LOCKED entry
# ------------------------------------------------------------------------------------------


class TestSabotage1BeginOnALockedEntry:
    width = 2

    def locked_entries(self) -> set[tuple[Hashable, tuple[int, ...]]]:
        ref = crusher_reference()
        return {
            (ref.observable(s), box)
            for s, box in visited_of("crusher", self.width)
            if field_of(s, "op") == "LOCKED"
        }

    def test_the_checker_turns_red(self) -> None:
        locked = self.locked_entries()
        assert locked, "no LOCKED entry is reachable: the sabotage would test nothing"
        table = true_table("crusher", self.width)
        bad = edit_entries(
            table, lambda o, box, cmds: cmds | {"BEGIN"} if (o, box) in locked else cmds
        )
        ref = crusher_reference()
        result = check(ref, bad, ref.initial(), self.width)
        assert result.findings
        assert {(kind, detail) for kind, _state, detail in result.findings} == {
            ("VIOLATION", "BEGIN")
        }
        assert all(field_of(state[0], "op") == "LOCKED" for _k, state, _d in result.findings)

    def test_one_edited_entry_is_enough(self) -> None:
        one = sorted(self.locked_entries(), key=repr)[0]
        table = true_table("crusher", self.width)
        bad = edit_entries(
            table, lambda o, box, cmds: cmds | {"BEGIN"} if (o, box) == one else cmds
        )
        ref = crusher_reference()
        result = check(ref, bad, ref.initial(), self.width)
        assert result.findings and all(f[1][1] == one[1] for f in result.findings)

    def test_an_edited_entry_nobody_can_reach_is_not_reported(self) -> None:
        ref = crusher_reference()
        table = true_table("crusher", self.width)
        reach_obs = {ref.observable(s) for s, _ in visited_of("crusher", self.width)}
        # a locked crusher with someone inside exists in the table (a raw state) but is unreachable
        unreachable = {o for (o, _box) in table.entries if o[0] == "LOCKED" and o[2] == 1}
        assert unreachable and not unreachable & reach_obs
        bad = edit_entries(
            table, lambda o, box, cmds: cmds | {"BEGIN"} if o in unreachable else cmds
        )
        assert bad.entries != table.entries
        assert check(ref, bad, ref.initial(), self.width).clean


# ------------------------------------------------------------------------------------------
# (2) handover does not void the permit (0003 A9); the old 'soon guard' wording as a table edit
# ------------------------------------------------------------------------------------------


class TestSabotage2HandoverDoesNotVoidThePermit:
    def test_the_differential_turns_red_on_states_with_a_handover_pending(self) -> None:
        found = differential(crusher_reference(), wrong(HandoverKeepsThePermit), limit=40)
        assert found
        assert all(field_of(s, "soon") == 1 and field_of(s, "permit") != "NONE" for s, *_ in found)

    def test_a_shield_built_from_that_model_admits_an_entry_before_a_handover(self) -> None:
        findings = findings_of(HandoverKeepsThePermit)
        assert findings
        assert {f[0] for f in findings} == {"VIOLATION"}
        enters = [f for f in findings if f[2] == "ENTER"]
        assert enters and all(field_of(f[1][0], "soon") == 1 for f in enters)

    def test_removing_the_soon_guard_from_enter_in_the_table_turns_the_checker_red(self) -> None:
        width = 2
        ref = crusher_reference()
        guarded = {  # the entries whose only missing condition is that no handover is pending
            (ref.observable(s), box)
            for s, box in visited_of("crusher", width)
            if (field_of(s, "op"), field_of(s, "permit"), field_of(s, "occ"))
            == ("WORK", "VALID", 0)
            and (field_of(s, "gas"), field_of(s, "alarm"), field_of(s, "soon")) == (1, 0, 1)
        }
        assert guarded, "no entry with a handover pending is reachable"
        bad = edit_entries(
            true_table("crusher", width),
            lambda o, box, cmds: cmds | {"ENTER"} if (o, box) in guarded else cmds,
        )
        result = check(ref, bad, ref.initial(), width)
        assert result.findings
        assert {(k, d) for k, _s, d in result.findings} == {("VIOLATION", "ENTER")}


# ------------------------------------------------------------------------------------------
# (3) allow START under lock
# ------------------------------------------------------------------------------------------


class TestSabotage3StartUnderLock:
    width = 2
    under_lock = ("LOCKED", "ISO", "WORK")

    def test_adding_start_to_the_table_of_a_locked_asset_turns_the_checker_red(self) -> None:
        ref = crusher_reference()
        reach = visited_of("crusher", self.width)
        entries = {
            (ref.observable(s), box) for s, box in reach if field_of(s, "op") in self.under_lock
        }
        assert {field_of(s, "op") for s, _ in reach if field_of(s, "op") in self.under_lock} == set(
            self.under_lock
        )
        bad = edit_entries(
            true_table("crusher", self.width),
            lambda o, box, cmds: cmds | {"START"} if (o, box) in entries else cmds,
        )
        result = check(ref, bad, ref.initial(), self.width)
        assert result.findings
        assert {(k, d) for k, _s, d in result.findings} == {("VIOLATION", "START")}
        assert {field_of(s[0], "op") for _k, s, _d in result.findings} == set(self.under_lock)

    def test_a_model_that_allows_it_is_caught_by_the_differential_and_by_the_checker(self) -> None:
        assert differential_is_red(StartUnderLockIsFine)
        findings = findings_of(StartUnderLockIsFine)
        assert findings and {(k, d) for k, _s, d in findings} == {("VIOLATION", "START")}

    def test_the_control_start_is_never_offered_under_lock_by_the_true_table(self) -> None:
        for (o, _box), cmds in true_table("crusher", self.width).entries.items():
            if o[0] in self.under_lock:
                assert "START" not in cmds


# ------------------------------------------------------------------------------------------
# (4) drop the refusal branch from the model
# ------------------------------------------------------------------------------------------


class TestSabotage4RefusalBranchDropped:
    def test_crusher_differential_turns_red_and_every_difference_is_a_missing_outcome(self) -> None:
        found = differential(crusher_reference(), wrong(NoRefusalBranch), limit=200)
        assert len(found) == 200
        for _s, c, theirs, mine in found:
            assert mine < theirs  # the sabotaged relation is a strict subset: outcomes are missing
            assert c not in ("NOOP", "STOP", "EXIT")

    def test_pump_differential_turns_red(self) -> None:
        found = differential(pump_reference(), wrong(PumpNoRefusalBranch), limit=200)
        assert len(found) == 200 and all(mine < theirs for _s, _c, theirs, mine in found)

    def test_a_shield_built_without_refusals_is_also_unsafe_for_the_checker(self) -> None:
        assert {k for k, *_ in findings_of(NoRefusalBranch)} == {"VIOLATION"}
        assert {k for k, *_ in findings_of(PumpNoRefusalBranch)} == {"VIOLATION"}

    def test_the_forced_commands_are_not_a_difference(self) -> None:
        ref = crusher_reference()
        sab = wrong(NoRefusalBranch)
        for s in list(ref.states())[::97]:
            for c in ("NOOP", "STOP", "EXIT"):
                assert sab.outcomes(s, c) == ref.outcomes(s, c)


# ------------------------------------------------------------------------------------------
# (5) ENTER ignores the alarm, and the alarm check on occupied states is gone (0003 A9)
# ------------------------------------------------------------------------------------------


class TestSabotage5EnterIgnoresTheAlarm:
    def test_the_differential_turns_red(self) -> None:
        found = differential(
            crusher_reference(), wrong(EnterIgnoresTheAlarmAndTheOccupiedCheckIsGone), limit=60
        )
        assert found
        # the differences are exactly the entries into an alarmed space and what follows
        assert all(field_of(s, "alarm") == 1 for s, *_ in found)
        assert any(c == "ENTER" for _s, c, *_ in found)

    def test_a_shield_built_from_that_model_admits_entry_into_an_alarmed_space(self) -> None:
        findings = findings_of(EnterIgnoresTheAlarmAndTheOccupiedCheckIsGone)
        enters = [f for f in findings if f[2] == "ENTER"]
        assert enters and all(field_of(f[1][0], "alarm") == 1 for f in enters)

    def test_the_alarm_guard_on_enter_alone_is_an_equivalent_mutant(self) -> None:
        # the occupied-state check already makes an entry into an alarmed space a violation, so
        # sabotaging only the guard changes nothing; this is why 0003 A9 asks for both
        assert differential(crusher_reference(), wrong(EnterIgnoresTheAlarm)) == []

    def test_the_occupied_state_alarm_check_alone_is_caught_by_the_differential(self) -> None:
        found = differential(crusher_reference(), wrong(NoAlarmCheckWhenOccupied), limit=20)
        assert found and all(
            field_of(s, "occ") == 1 and field_of(s, "alarm") == 1 for s, *_ in found
        )

    def test_under_the_shield_an_occupant_can_stay_one_quarter_only(self) -> None:
        # Why the occupied-state alarm check alone cannot turn the *checker* red. With a 2-quarter
        # period and a refusable TEST, an occupant's test lapses after the entry tick, so the only
        # occupied state the shield ever reaches has gas 2 and its only safe command is EXIT.
        for width in WIDTHS:
            reach = {s for s, _ in visited_of("crusher", width)}
            occupied = {s for s in reach if field_of(s, "occ") == 1}
            assert occupied
            assert {field_of(s, "gas") for s in occupied} == {2}
            for (o, _box), cmds in true_table("crusher", width).entries.items():
                if o[2] == 1 and o[3] == 2 and o[0] == "WORK":
                    assert cmds <= {"EXIT"}, (o, cmds)


# ------------------------------------------------------------------------------------------
# (6) remove STOP from the pump table when the other pump is in WORK
# ------------------------------------------------------------------------------------------


class TestSabotage6StopRemovedFromThePumpTable:
    width = 2

    @staticmethod
    def without_stop(o: Hashable, _box: tuple[int, ...], cmds: frozenset[str]) -> frozenset[str]:
        if o == ("RUN", "WORK"):  # pump 1 running, pump 2 in WORK: pump 1's STOP is the way out
            return frozenset(c for c in cmds if c.split("|")[0] != "STOP")
        if o == ("WORK", "RUN"):
            return frozenset(c for c in cmds if c.split("|")[1] != "STOP")
        return cmds

    def test_the_situation_is_reachable_and_the_true_table_offers_stop_there(self) -> None:
        ref = pump_reference()
        reach_obs = {ref.observable(s) for s, _ in visited_of("pumps", self.width)}
        assert {("RUN", "WORK"), ("WORK", "RUN")} <= reach_obs
        table = true_table("pumps", self.width)
        critical = [
            cmds
            for (o, box), cmds in table.entries.items()
            if o == ("RUN", "WORK") and box[0] <= 3 <= box[1]
        ]
        assert critical and all(c.split("|")[0] == "STOP" for cmds in critical for c in cmds)

    def test_the_checker_turns_red(self) -> None:
        ref = pump_reference()
        bad = edit_entries(true_table("pumps", self.width), self.without_stop)
        result = check(ref, bad, ref.initial(), self.width)
        assert result.findings
        assert {f[0] for f in result.findings} <= {"DEADLOCK", "VIOLATION"}
        assert any(
            f[0] == "DEADLOCK" and f[2] in (("RUN", "WORK"), ("WORK", "RUN"))
            for f in result.findings
        )

    def test_removing_only_the_pair_that_stops_both_is_not_enough_to_notice(self) -> None:
        # a control that the sabotage above is not trivially red: other stopping pairs remain
        ref = pump_reference()
        bad = edit_entries(
            true_table("pumps", self.width),
            lambda o, box, cmds: cmds - {"STOP|STOP"} if o == ("RUN", "WORK") else cmds,
        )
        assert check(ref, bad, ref.initial(), self.width).clean


# ------------------------------------------------------------------------------------------
# (7) 0003 A12.1: a spent permit cannot end at an announced step (the permit stays VALID)
# ------------------------------------------------------------------------------------------


class TestSabotage7ASpentPermitCannotEnd:
    """The plant ends a permit at its 12-hour limit and the abstraction lets the environment end it
    at any announced step. A model without that branch is wrong, but not unsafe: an occupant is
    spoilt by a permit that must be revalidated just as by one that has ended, so the shield built
    from the wrong model is the true shield. Only the differential can see it (and, at the
    plant, the conformance check). The checker's half of this item is the table edit below, on
    states that exist only because a permit can end with the work open."""

    def test_the_differential_turns_red_for_both_ways_of_dropping_the_outcome(self) -> None:
        def at_an_announced_step(s: Hashable) -> bool:
            return field_of(s, "soon") == 1 and field_of(s, "permit") != "NONE"

        cannot_end = differential(crusher_reference(), wrong(SpentPermitCannotEnd), limit=400)
        assert len(cannot_end) == 400 and all(at_an_announced_step(s) for s, *_ in cannot_end)
        for _s, _c, theirs, mine in cannot_end:  # only outcomes missing, and all of them NONE
            assert mine < theirs
            assert all(t != VIOL and field_of(t, "permit") == "NONE" for t in theirs - mine)

        stays = differential(crusher_reference(), wrong(SpentPermitStaysValid), limit=400)
        assert len(stays) == 400 and all(at_an_announced_step(s) for s, *_ in stays)
        assert any(  # the permit that should have ended is still VALID in the wrong model
            t != VIOL and field_of(t, "permit") == "VALID"
            for *_, theirs, mine in stays
            for t in mine - theirs
        )

    def test_hand_worked_witness_a_valid_permit_at_an_announced_step(self) -> None:
        s = crusher_state(op="ISO", permit="VALID", soon=1)
        both = (0, 1)  # the next quarter is announced or not (A12.1)
        assert crusher_reference().outcomes(s, "NOOP") == {
            ("ISO", 0, p, 0, 3, 0, x) for p in ("REVAL", "NONE") for x in both
        }
        assert wrong(SpentPermitStaysValid).outcomes(s, "NOOP") == {
            ("ISO", 0, p, 0, 3, 0, x)
            for p in ("REVAL", "VALID")
            for x in both  # stays VALID
        }
        assert wrong(SpentPermitCannotEnd).outcomes(s, "NOOP") == {
            ("ISO", 0, "REVAL", 0, 3, 0, x) for x in both
        }

    def test_no_difference_without_an_announced_step_or_without_a_live_permit(self) -> None:
        for cls in (SpentPermitStaysValid, SpentPermitCannotEnd):
            sab, ref = wrong(cls), crusher_reference()
            for permit, soon in (("NONE", 1), ("VALID", 0), ("REVAL", 0), ("NONE", 0)):
                s = crusher_state(op="WORK", permit=permit, soon=soon)
                assert sab.outcomes(s, "NOOP") == ref.outcomes(s, "NOOP"), (cls, permit, soon)

    def test_the_shield_built_from_either_wrong_model_is_the_true_shield(self) -> None:
        for cls in (SpentPermitStaysValid, SpentPermitCannotEnd):
            assert findings_of(cls) == ()
            assert dict(shield_built_from(cls).entries) == dict(true_table("crusher", 2).entries)

    def test_admitting_enter_with_a_spent_permit_turns_the_checker_red(self) -> None:
        width, ref = 2, crusher_reference()
        spent = {
            (ref.observable(s), box)
            for s, box in visited_of("crusher", width)
            if (field_of(s, "op"), field_of(s, "permit"), field_of(s, "occ")) == ("WORK", "NONE", 0)
            and (field_of(s, "gas"), field_of(s, "alarm")) == (1, 0)
        }
        assert spent, "a permit spent with the work open and a fresh gas test is not reachable"
        bad = edit_entries(
            true_table("crusher", width),
            lambda o, box, cmds: cmds | {"ENTER"} if (o, box) in spent else cmds,
        )
        result = check(ref, bad, ref.initial(), width)
        assert result.findings
        assert {(k, d) for k, _s, d in result.findings} == {("VIOLATION", "ENTER")}
        assert {field_of(s[0], "permit") for _k, s, _d in result.findings} == {"NONE"}

    def test_those_states_cannot_be_reached_in_the_game_before_a12_so_the_edit_was_invisible(
        self,
    ) -> None:
        old = before_a12()
        reach = {s for s, _ in check(old, table_from_model(old, 2), old.initial(), 2).visited}
        assert not any(
            field_of(s, "op") == "WORK" and field_of(s, "permit") == "NONE" for s in reach
        )

    def test_issue_in_work_is_what_lets_the_shield_replace_a_spent_permit(self) -> None:
        # the model that forgot it differs at exactly the 4 bins x 3 gas ages x 2 alarm x 2 soon
        # raw states with the work open, no permit and nobody inside, for the one command ISSUE
        found = differential(crusher_reference(), wrong(IssueOnlyInIso))
        assert len(found) == 4 * 3 * 2 * 2
        assert {c for _s, c, *_ in found} == {"ISSUE"}
        assert {
            (field_of(s, "op"), field_of(s, "permit"), field_of(s, "occ")) for s, *_ in found
        } == {("WORK", "NONE", 0)}
        # safe without it, as the model: a spent permit is simply never replaced
        assert findings_of(IssueOnlyInIso) == ()


class TestSabotage7BTheAnnouncementCannotRepeat:
    """0003 A12.1: a handover and a permit limit can fall on consecutive quarters, so the step that
    consumes an announcement may announce the next quarter again. A model that clears it is
    missing transitions the plant makes. Like the dropped end of a permit, it changes no safe set
    (the successors it omits are winning states), so the differential is the only thing that sees
    it, and the plant conformance check."""

    def test_the_differential_turns_red_at_announced_steps_only(self) -> None:
        found = differential(crusher_reference(), wrong(AnnouncementCannotRepeat))
        assert found and all(field_of(s, "soon") == 1 for s, *_ in found)
        for _s, _c, theirs, mine in found:  # only outcomes missing, and all of them announced
            assert mine < theirs
            assert all(t != VIOL and field_of(t, "soon") == 1 for t in theirs - mine)

    def test_hand_worked_witness(self) -> None:
        s = crusher_state(op="ISO", permit="VALID", soon=1)
        assert {field_of(t, "soon") for t in crusher_reference().outcomes(s, "NOOP")} == {0, 1}
        assert {
            field_of(t, "soon") for t in wrong(AnnouncementCannotRepeat).outcomes(s, "NOOP")
        } == {0}

    def test_the_shield_built_from_it_is_the_true_shield(self) -> None:
        assert findings_of(AnnouncementCannotRepeat) == ()
        table = shield_built_from(AnnouncementCannotRepeat)
        assert dict(table.entries) == dict(true_table("crusher", 2).entries)


# ------------------------------------------------------------------------------------------
# (8) 0003 A12.2: a locked pump counts as available
# ------------------------------------------------------------------------------------------


class TestSabotage8ALockedPumpCountsAsAvailable:
    width = 2

    def test_the_differential_turns_red_and_only_the_wrong_model_is_more_permissive(self) -> None:
        found = differential(pump_reference(), wrong(LockedPumpCountsAsAvailable), limit=500)
        assert len(found) == 500
        for _s, _c, theirs, mine in found:
            assert VIOL in theirs  # the true relation is the stricter one
            assert theirs - {VIOL} <= mine and mine - theirs  # the wrong one has outcomes to spare

    def test_hand_worked_witness_a_running_pump_at_the_top_bin_beside_a_locked_one(self) -> None:
        s = ("RUN", 3, "LOCKED", 0)
        assert pump_reference().outcomes(s, "NOOP|NOOP") == {VIOL, ("RUN", 3, "LOCKED", 0)}
        assert wrong(LockedPumpCountsAsAvailable).outcomes(s, "NOOP|NOOP") == {
            ("RUN", 3, "LOCKED", 0),
            ("FAIL", 3, "LOCKED", 0),  # both pumps gone, and the wrong model calls that fine
        }

    def test_a_shield_built_from_that_model_locks_the_last_available_pump(self) -> None:
        findings = findings_of(LockedPumpCountsAsAvailable, self.width)
        assert findings and {f[0] for f in findings} == {"VIOLATION"}
        # the witness: pump 1 stopped, pump 2 isolated, and the wrong shield offers to lock pump 1
        assert any(
            f[1][0][0] == "STOP" and f[1][0][2] == "ISO" and f[2].split("|")[0] == "LOCK"
            for f in findings
        )

    def test_the_true_shield_never_offers_it(self) -> None:
        table = true_table("pumps", self.width)
        for (o, _box), cmds in table.entries.items():
            if o == ("STOP", "ISO"):
                assert not any(c.split("|")[0] == "LOCK" for c in cmds), cmds

    def test_adding_lock_for_the_last_available_pump_to_the_table_turns_the_checker_red(
        self,
    ) -> None:
        ref = pump_reference()
        reach = visited_of("pumps", self.width)
        entries = {(ref.observable(s), box) for s, box in reach if s[0] == "STOP" and s[2] == "ISO"}
        assert entries
        bad = edit_entries(
            true_table("pumps", self.width),
            lambda o, box, cmds: cmds | {"LOCK|NOOP"} if (o, box) in entries else cmds,
        )
        result = check(ref, bad, ref.initial(), self.width)
        assert {(k, d) for k, _s, d in result.findings} == {("VIOLATION", "LOCK|NOOP")}

    def test_a_pump_started_while_locked_is_a_violation_in_the_model_and_a_wrong_model_misses_it(
        self,
    ) -> None:
        assert differential_is_red(PumpStartOnALockedPumpIsFine)
        findings = findings_of(PumpStartOnALockedPumpIsFine)
        assert findings and {f[0] for f in findings} == {"VIOLATION"}
        assert all("START" in f[2] for f in findings)


# ------------------------------------------------------------------------------------------
# (9) 0003 A12.3: the conveyor's BEGIN on an unproven asset is admitted
# ------------------------------------------------------------------------------------------


class TestSabotage9ConveyorBeginOnAnUnprovenAsset:
    width = 2
    unproven = ("RUN", "STOP", "LOCKED", "FAIL")

    def test_the_differential_turns_red_at_exactly_sixteen_states_and_one_command(self) -> None:
        found = differential(conveyor_reference(), wrong(ConveyorBeginOnAnUnprovenAsset))
        # four unproven ops at four bins, one command (hand count)
        assert len(found) == 4 * 4
        assert {c for _s, c, *_ in found} == {"BEGIN"}
        assert {s[0] for s, *_ in found} == set(self.unproven)
        assert all(VIOL in theirs and VIOL not in mine for _s, _c, theirs, mine in found)

    def test_a_shield_built_from_that_model_admits_begin_on_an_unproven_conveyor(self) -> None:
        findings = findings_of(ConveyorBeginOnAnUnprovenAsset, self.width)
        assert findings and {(f[0], f[2]) for f in findings} == {("VIOLATION", "BEGIN")}
        assert {f[1][0][0] for f in findings} <= set(self.unproven)

    def test_adding_begin_to_the_table_of_an_unproven_conveyor_turns_the_checker_red(self) -> None:
        ref = conveyor_reference()
        reach = visited_of("conveyor", self.width)
        entries = {(ref.observable(s), box) for s, box in reach if s[0] in self.unproven}
        assert {o[0] for o, _ in entries} == set(self.unproven)
        bad = edit_entries(
            true_table("conveyor", self.width),
            lambda o, box, cmds: cmds | {"BEGIN"} if (o, box) in entries else cmds,
        )
        result = check(ref, bad, ref.initial(), self.width)
        assert result.findings and {(k, d) for k, _s, d in result.findings} == {
            ("VIOLATION", "BEGIN")
        }
        assert {s[0][0] for _k, s, _d in result.findings} == set(self.unproven)

    def test_an_edited_entry_nobody_can_reach_is_not_reported(self) -> None:
        # a failed conveyor is only ever at bin 3, so an entry for a set that excludes bin 3 is
        # never walked: the same blindness as for the crusher, and it is by design
        ref = conveyor_reference()
        table = true_table("conveyor", self.width)
        low = {(o, box) for (o, box) in table.entries if o == ("FAIL",) and box[1] < 3}
        assert low
        bad = edit_entries(
            table, lambda o, box, cmds: cmds | {"BEGIN"} if (o, box) in low else cmds
        )
        assert check(ref, bad, ref.initial(), self.width).clean
        high = {
            (o, box) for (o, box) in table.entries if o == ("FAIL",) and box[0] >= 1 and box[1] == 3
        }
        bad = edit_entries(
            table, lambda o, box, cmds: cmds | {"BEGIN"} if (o, box) in high else cmds
        )
        assert not check(ref, bad, ref.initial(), self.width).clean

    def test_the_control_begin_is_offered_on_an_isolated_conveyor_and_nowhere_else(self) -> None:
        for (o, _box), cmds in true_table("conveyor", self.width).entries.items():
            offered_at = (("ISO",), ("WORK",))  # at WORK it is a no-op
            assert ("BEGIN" in cmds) == (o in offered_at), (o, cmds)

    def test_start_under_lock_is_caught_too(self) -> None:
        assert differential_is_red(ConveyorStartUnderLockIsFine)
        findings = findings_of(ConveyorStartUnderLockIsFine)
        assert {(f[0], f[2]) for f in findings} == {("VIOLATION", "START")}
        assert {f[1][0][0] for f in findings} == {"LOCKED", "ISO", "WORK"}

    def test_a_conveyor_that_is_never_refused_differs_in_fifty_six_pairs_and_is_still_safe(
        self,
    ) -> None:
        # the six refusable commands change something at these (state, command) pairs, by hand:
        # BEGIN 4 unproven ops + ISO, 5 x 4 = 20; START 3 locked ops + STOP, 4 x 4 = 16; LOCK 2 x 4;
        # PROVE, COMPLETE, UNISO 1 x 4 each
        found = differential(conveyor_reference(), wrong(ConveyorNoRefusalBranch))
        assert len(found) == 20 + 16 + 8 + 4 + 4 + 4 == 56
        assert all(mine < theirs for _s, _c, theirs, mine in found)
        assert findings_of(ConveyorNoRefusalBranch) == ()  # NOOP is always safe for a conveyor


# ------------------------------------------------------------------------------------------
# further model sabotage: each must be seen by the differential; most also by the checker
# ------------------------------------------------------------------------------------------

MODEL_SABOTAGE = [
    # (sabotaged model, does a shield built from it also fail the checker?)
    (HandoverKeepsThePermit, True),
    (StartUnderLockIsFine, True),
    (NoRefusalBranch, True),
    (PumpNoRefusalBranch, True),
    (EnterIgnoresTheAlarmAndTheOccupiedCheckIsGone, True),
    (FailureIsHarmless, True),
    (NoStaleTestRule, True),
    (NoPermitRule, True),
    (IsolatedPumpCountsAsAvailable, True),
    (LockedPumpCountsAsAvailable, True),
    (PumpStartOnALockedPumpIsFine, True),
    (ConveyorBeginOnAnUnprovenAsset, True),
    (ConveyorStartUnderLockIsFine, True),
    # caught by the differential only: the shield stays safe because the branch they drop (or the
    # check they remove) is never what stands between a reachable state and the sink
    (NoAlarmCheckWhenOccupied, False),
    (AlarmNeverRisesOnItsOwn, False),
    # A12: a permit that can end, ISSUE in WORK and a conveyor that is told to do what it is told
    (SpentPermitStaysValid, False),
    (SpentPermitCannotEnd, False),
    (AnnouncementCannotRepeat, False),
    (IssueOnlyInIso, False),
    (ConveyorNoRefusalBranch, False),
]


@pytest.mark.parametrize(
    "cls, checker_also_red", MODEL_SABOTAGE, ids=lambda x: getattr(x, "__name__", str(x))
)
def test_every_model_sabotage_turns_the_differential_red(
    cls: type[Any], checker_also_red: bool
) -> None:
    assert differential_is_red(cls)
    if checker_also_red:
        assert {k for k, *_ in findings_of(cls)} == {"VIOLATION"}


def test_dropping_the_holder_absent_branch_is_an_equivalent_mutant() -> None:
    # an absent holder leaves the permit as it was, which is also what a refused REVAL does
    assert differential(crusher_reference(), wrong(HolderIsAlwaysPresent)) == []
    assert findings_of(HolderIsAlwaysPresent) == ()
