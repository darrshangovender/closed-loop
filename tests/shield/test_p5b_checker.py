"""The checker: reachability of the shielded product against the independent reference.

Counts are this checker's own. ADR 0003 A9 voids the spike's 200 / 496 / 792 / 992 (they came from a
game in which nobody could enter); A11 adds the unrenewed completion, which takes the crusher from
216 / 536 / 856 / 1,072 to 240 / 600 / 960 / 1,200; A12.1 (a permit can be spent with the work open)
takes it to 304 / 760 / 1,216 / 1,520. The pump pair, with the locked pump of A12.2, reaches
272 / 1,680 / 4,288 / 6,720 (it was 208 / 1,280 / 3,264 / 5,120). The conveyor of A12.3 reaches
21 / 52 / 83 / 104, which is counted by hand below. They are pinned so a change to either side shows
up.

Classes with ``Shipped`` or ``Production`` in the name need the production models and tables; every
other test here uses tables built from the reference alone (``p5b_support.true_table``).
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from typing import Any

import pytest

from closed_loop.shield.checker import (
    CheckResult,
    check,
    empty_entries,
    health_vector,
)
from closed_loop.shield.reference import VIOL, ReferenceGame
from tests.shield.p5b_support import (
    KINDS,
    WIDTHS,
    before_a12,
    conveyor_reference,
    crusher_reference,
    edit_entries,
    field_of,
    literal_issue,
    pump_reference,
    reference_for,
    renews_always,
    shipped_table,
    solved,
    spike_reading,
    table_from_model,
    true_table,
)

CRUSHER_REACHABLE = {1: 304, 2: 760, 3: 1216, 4: 1520}
PUMP_REACHABLE = {1: 272, 2: 1680, 3: 4288, 4: 6720}
CONVEYOR_REACHABLE = {1: 21, 2: 52, 3: 83, 4: 104}
REACHABLE = {"crusher": CRUSHER_REACHABLE, "pumps": PUMP_REACHABLE, "conveyor": CONVEYOR_REACHABLE}
BEFORE_A12_REACHABLE = {1: 240, 2: 600, 3: 960, 4: 1200}  # the crusher at 0003 A11


# ------------------------------------------------------------------------------------------
# a game small enough to count by hand
# ------------------------------------------------------------------------------------------


class Toy:
    """a (bin 0) -go-> b (bin 1) -go-> c (bin 3); 'bad' is a violation in b and c, maybe in a."""

    name = "toy"
    _bin = {"a": 0, "b": 1, "c": 3}
    _next = {("a", "go"): {"b"}, ("b", "go"): {"c"}, ("c", "go"): {"c"}}

    def states(self) -> list[Hashable]:
        return ["a", "b", "c"]

    def commands(self) -> tuple[str, ...]:
        return ("go", "stay", "bad", "split")

    def initial(self) -> Hashable:
        return "a"

    def observable(self, s: Hashable) -> Hashable:
        return s

    def health(self, s: Hashable) -> int:
        return self._bin[str(s)]

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self.outcomes(s, c)

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c == "stay":
            return frozenset({s})
        if c == "split":  # the environment picks: stay put, or jump to the critical bin
            return frozenset({"a", "c"}) if s == "a" else frozenset({s})
        if c == "bad":
            return frozenset({VIOL, s}) if s == "a" else frozenset({VIOL})
        return frozenset(self._next[(str(s), "go")])


class StubTable:
    """Just enough of a ShieldTable for the walk: a fixed answer per observable."""

    def __init__(self, answers: dict[Hashable, Sequence[str]], default: Sequence[str]) -> None:
        self.answers = answers
        self.default = default
        self.asked: list[tuple[Hashable, tuple[int, ...]]] = []

    def allowed(self, o: Hashable, health_set: Sequence[int]) -> frozenset[str]:
        self.asked.append((o, tuple(health_set)))
        return frozenset(self.answers.get(o, self.default))


class TestTheWalkOnAToyGame:
    @pytest.mark.parametrize("width, expected", [(1, 3), (2, 7), (3, 11), (4, 14)])
    def test_every_set_that_holds_the_next_bin_and_fits_the_width_is_visited(
        self, width: int, expected: int
    ) -> None:
        # by hand: bin 0 holds w sets, bin 3 holds w sets, bin 1 holds 1, 3, 5, 6 sets at w = 1..4
        result = check(Toy(), StubTable({}, ("go", "stay")), "a", width)  # type: ignore[arg-type]
        assert result.clean
        assert result.reachable == len(result.visited) == expected

    def test_starting_sets_are_every_set_that_holds_the_starting_bin(self) -> None:
        result = check(Toy(), StubTable({}, ("stay",)), "a", 3)  # type: ignore[arg-type]
        assert {box for _s, box in result.visited} == {(0, 0), (0, 1), (0, 2)}
        assert {s for s, _box in result.visited} == {"a"}

    def test_an_admitted_command_with_the_sink_among_its_outcomes_is_a_violation(self) -> None:
        table = StubTable({"a": ("bad",)}, ("go", "stay"))
        result = check(Toy(), table, "a", 1)  # type: ignore[arg-type]
        assert result.findings == (("VIOLATION", ("a", (0, 0)), "bad"),)
        assert not result.clean

    def test_every_admitted_command_is_taken_not_only_the_first(self) -> None:
        for allowed in (("stay", "bad"), ("bad", "stay"), ("go", "bad")):
            table = StubTable({"a": allowed}, ("stay",))
            result = check(Toy(), table, "a", 1)  # type: ignore[arg-type]
            assert result.findings == (("VIOLATION", ("a", (0, 0)), "bad"),), allowed

    def test_every_outcome_of_a_nondeterministic_command_is_followed(self) -> None:
        # 'split' may leave the game at a or send it to c; c is where 'bad' is admitted
        table = StubTable({"a": ("split",), "c": ("bad",)}, ("stay",))
        result = check(Toy(), table, "a", 1)  # type: ignore[arg-type]
        assert result.findings == (("VIOLATION", ("c", (3, 3)), "bad"),)
        assert result.reachable == 2  # a and c; b is never reached

    def test_a_reachable_entry_that_admits_nothing_is_a_deadlock(self) -> None:
        table = StubTable({"b": ()}, ("go", "stay"))
        result = check(Toy(), table, "a", 1)  # type: ignore[arg-type]
        assert result.findings == (("DEADLOCK", ("b", (1, 1)), "b"),)

    def test_an_unreachable_empty_entry_is_not_a_finding(self) -> None:
        table = StubTable({"c": ()}, ("stay",))  # c is never reached when nothing moves
        assert check(Toy(), table, "a", 2).clean  # type: ignore[arg-type]

    def test_the_drop_hook_removes_one_command_at_one_observable(self) -> None:
        table = StubTable({}, ("go", "stay"))
        full = check(Toy(), table, "a", 1)  # type: ignore[arg-type]
        assert full.reachable == 3
        cut = check(Toy(), table, "a", 1, drop=("a", "go"))  # type: ignore[arg-type]
        assert cut.reachable == 1 and cut.clean  # only 'stay' is left at a
        dead = check(Toy(), StubTable({}, ("go",)), "a", 1, drop=("a", "go"))  # type: ignore[arg-type]
        assert dead.findings == (("DEADLOCK", ("a", (0, 0)), "a"),)

    def test_max_width_outside_one_to_four_is_refused(self) -> None:
        for bad in (0, 5, -1):
            with pytest.raises(ValueError):
                check(Toy(), StubTable({}, ("stay",)), "a", bad)  # type: ignore[arg-type]

    def test_health_vector_accepts_a_bin_or_a_tuple_of_bins(self) -> None:
        assert health_vector(2) == (2,) and health_vector((1, 3)) == (1, 3)


# ------------------------------------------------------------------------------------------
# the true tables
# ------------------------------------------------------------------------------------------


class TestACleanShieldPasses:
    """Tables made from the reference alone (``true_table``): this needs no production code."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_table_has_no_finding_and_the_pinned_reach(self, kind: str, width: int) -> None:
        ref = reference_for(kind)
        result = check(ref, true_table(kind, width), ref.initial(), width)
        assert result.findings == ()
        assert result.reachable == REACHABLE[kind][width]

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("width", WIDTHS)
    def test_no_entry_of_any_table_admits_nothing(self, kind: str, width: int) -> None:
        assert empty_entries(true_table(kind, width)) == []

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_wide_table_walked_at_a_narrower_limit_matches_the_narrow_table(
        self, kind: str
    ) -> None:
        ref = reference_for(kind)
        wide = true_table(kind, 4)
        for width in (1, 2, 3):
            assert check(ref, wide, ref.initial(), width).reachable == REACHABLE[kind][width]

    def test_reach_grows_with_the_width_limit(self) -> None:
        for reach in REACHABLE.values():
            values = [reach[w] for w in WIDTHS]
            assert values == sorted(set(values))

    def test_the_conveyor_reach_is_the_hand_counted_one(self) -> None:
        # 0003 A12.3. Every state wins and the safe set depends on the op alone, so the shield
        # admits every command that is no violation at that op. Reachable: five ops at all four
        # bins, and a failed conveyor only at bin 3 (it fails from there): 21 states of 24. A state
        # is visited with every set that holds its bin and is no wider than the limit: bins 0 and 3
        # lie in w such sets, bins 1 and 2 in 1 / 3 / 5 / 6, so 10 (B0 + B1) + B0.
        ref = conveyor_reference()
        for width, (b0, b1) in zip(WIDTHS, [(1, 1), (2, 3), (3, 5), (4, 6)], strict=True):
            result = check(ref, true_table("conveyor", width), ref.initial(), width)
            assert result.reachable == 10 * (b0 + b1) + b0 == CONVEYOR_REACHABLE[width]
        states = {s for s, _ in check(ref, true_table("conveyor", 1), ref.initial(), 1).visited}
        assert states == {
            (op, h) for op in ("RUN", "STOP", "LOCKED", "ISO", "WORK") for h in range(4)
        } | {("FAIL", 3)}

    def test_the_conveyor_table_has_the_hand_counted_number_of_entries(self) -> None:
        # six observable states, each with every set of at most w bins: 4 / 7 / 9 / 10 sets
        for width, sets in zip(WIDTHS, (4, 7, 9, 10), strict=True):
            assert len(true_table("conveyor", width).entries) == 6 * sets


class TestShippedTablesAreClean:
    """The table the production synthesis built from the production model, walked against the
    reference. Needs the production models of A12 (``ConveyorModel`` among them)."""

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_shipped_table_has_no_finding_and_the_reach_of_the_reference_table(
        self, kind: str, width: int
    ) -> None:
        ref = reference_for(kind)
        result = check(ref, shipped_table(kind, width), ref.initial(), width)
        assert result.findings == ()
        assert result.reachable == REACHABLE[kind][width]

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("width", WIDTHS)
    def test_no_entry_of_a_shipped_table_admits_nothing(self, kind: str, width: int) -> None:
        assert empty_entries(shipped_table(kind, width)) == []


class TestTheProofIsNotVacuous:
    """The spike proved its properties over states nobody could reach (0003 A9). This one is not."""

    def visited(self, width: int = 1) -> set[Hashable]:
        ref = crusher_reference()
        return {
            s for s, _ in check(ref, true_table("crusher", width), ref.initial(), width).visited
        }

    def test_people_enter_the_space_under_the_shield(self) -> None:
        inside = [s for s in self.visited() if field_of(s, "occ") == 1]
        assert inside, "the shield admits no entry: the occupancy properties would be vacuous"
        assert {field_of(s, "op") for s in inside} == {"WORK"}

    def test_every_occupancy_property_has_a_reachable_witness(self) -> None:
        reach = self.visited()
        assert any(field_of(s, "occ") == 1 and field_of(s, "alarm") == 1 for s in reach)  # P6
        assert any(field_of(s, "occ") == 1 and field_of(s, "gas") == 2 for s in reach)  # P3 edge
        assert any(field_of(s, "occ") == 1 and field_of(s, "soon") == 1 for s in reach)  # P2 edge
        assert any(field_of(s, "permit") == "REVAL" for s in reach)  # a handover was survived
        assert any(field_of(s, "h") == 3 and field_of(s, "op") == "RUN" for s in reach)  # P5 edge

    def test_a_permit_is_spent_with_the_work_open_and_issued_again(self) -> None:
        # 0003 A12.1: this state was unreachable while a handover could only send a permit to REVAL
        reach = self.visited()
        spent = [
            s for s in reach if field_of(s, "op") == "WORK" and field_of(s, "permit") == "NONE"
        ]
        assert spent and all(field_of(s, "occ") == 0 for s in spent)
        table = true_table("crusher", 1)
        assert any(
            "ISSUE" in table.allowed(crusher_reference().observable(s), (field_of(s, "h"),) * 2)
            for s in spent
        )

    def test_every_live_op_is_reachable_and_the_failed_op_is_not(self) -> None:
        ops = {field_of(s, "op") for s in self.visited()}
        assert ops == {"RUN", "STOP", "LOCKED", "ISO", "WORK"}  # a crusher failure is the sink

    def test_both_pumps_get_isolated_for_work_in_turn_but_never_together(self) -> None:
        ref = pump_reference()
        reach = {s for s, _ in check(ref, true_table("pumps", 1), ref.initial(), 1).visited}
        assert any(s[0] == "WORK" for s in reach) and any(s[2] == "WORK" for s in reach)
        assert any(s[0] == "LOCKED" for s in reach) and any(s[2] == "LOCKED" for s in reach)
        unavailable = ("LOCKED", "ISO", "WORK", "FAIL")
        assert not any(s[0] in unavailable and s[2] in unavailable for s in reach)

    def test_with_one_pump_locked_the_runner_at_the_top_bin_can_only_be_stopped(self) -> None:
        # 0003 A12.2: a locked pump is not available, so the other is the only one there is
        ref = pump_reference()
        table = true_table("pumps", 4)
        reach = check(ref, table, ref.initial(), 4).visited
        critical = [
            (s, box) for s, box in reach if s[0] == "LOCKED" and s[2] == "RUN" and s[3] == 3
        ]
        assert critical
        for s, box in critical:
            assert {c.split("|")[1] for c in table.allowed(ref.observable(s), box)} == {"STOP"}

    def test_the_conveyor_fails_only_from_the_top_bin_and_every_other_op_is_reached(self) -> None:
        ref = conveyor_reference()
        reach = {s for s, _ in check(ref, true_table("conveyor", 1), ref.initial(), 1).visited}
        assert {s for s in reach if s[0] == "FAIL"} == {("FAIL", 3)}
        assert {s[0] for s in reach} == {"RUN", "STOP", "LOCKED", "ISO", "WORK", "FAIL"}

    @pytest.mark.parametrize("kind", KINDS)
    def test_every_reachable_state_is_winning(self, kind: str) -> None:
        ref = reference_for(kind)
        reach = {s for s, _ in check(ref, true_table(kind, 4), ref.initial(), 4).visited}
        assert reach <= solved(ref)[0]


# ------------------------------------------------------------------------------------------
# teeth: tables that are not safe
# ------------------------------------------------------------------------------------------


class TestAUnsafeTableIsReported:
    def test_a_table_that_admits_everything_is_full_of_violations(self) -> None:
        ref = crusher_reference()
        everything = StubTable({}, ref.commands())
        result = check(ref, everything, ref.initial(), 1)  # type: ignore[arg-type]
        kinds = {f[0] for f in result.findings}
        assert kinds == {"VIOLATION"}
        assert len(result.findings) > 100

    def test_a_table_that_only_does_nothing_lets_the_critical_crusher_fail(self) -> None:
        ref = crusher_reference()
        result = check(ref, StubTable({}, ("NOOP",)), ref.initial(), 1)  # type: ignore[arg-type]
        assert result.findings
        assert {f[2] for f in result.findings} == {"NOOP"}
        assert {field_of(f[1][0], "h") for f in result.findings} == {3}

    def test_the_pump_pair_left_alone_loses_both_pumps(self) -> None:
        ref = pump_reference()
        result = check(ref, StubTable({}, ("NOOP|NOOP",)), ref.initial(), 1)  # type: ignore[arg-type]
        assert result.findings and {f[0] for f in result.findings} == {"VIOLATION"}

    def test_a_conveyor_table_that_admits_everything_is_full_of_violations(self) -> None:
        ref = conveyor_reference()
        result = check(ref, StubTable({}, ref.commands()), ref.initial(), 1)  # type: ignore[arg-type]
        assert {f[0] for f in result.findings} == {"VIOLATION"}
        assert {f[2] for f in result.findings} == {"START", "BEGIN"}

    def test_a_conveyor_left_alone_is_safe_because_its_failure_is_not_a_violation(self) -> None:
        # 0003 A12.3: no pair rule and no rule against failing, so doing nothing never breaks one
        ref = conveyor_reference()
        for width in WIDTHS:
            assert check(ref, StubTable({}, ("NOOP",)), ref.initial(), width).clean  # type: ignore[arg-type]

    @pytest.mark.slow
    def test_a_stale_spec_shield_is_unsafe_against_the_current_retest_rule(self) -> None:
        # built for the 2-hour interval of the superseded revision (8 quarters), judged against 2
        stale = table_from_model(crusher_reference(8), 2)
        ref = crusher_reference()
        result = check(ref, stale, ref.initial(), 2)
        assert result.findings
        bad = [f for f in result.findings if f[0] == "VIOLATION"]
        assert bad and all(field_of(f[1][0], "occ") == 1 for f in bad)  # an occupant's test lapsed

    @pytest.mark.slow
    def test_the_stale_spec_shield_is_clean_against_its_own_period(self) -> None:
        ref = crusher_reference(8)
        assert check(ref, table_from_model(ref, 2), ref.initial(), 2).clean


class TestTheOldCountsAreExplained:
    """0002 P5b expected 200 / 496 / 792 / 992. Those came from a game in which nobody can enter;
    this checker reproduces them exactly on that reading, so the counting is the same and the
    difference is the change of A9 (216 / 536 / 856 / 1,072), then A11 (240 / 600 / 960 / 1,200),
    then A12.1 (304 / 760 / 1,216 / 1,520)."""

    def test_the_checker_returns_the_spike_counts_on_the_spike_reading(self) -> None:
        ref = spike_reading()
        assert len(solved(ref)[0]) == 1692  # the spike's winning region
        counts = {}
        for width in WIDTHS:
            result = check(ref, table_from_model(ref, width), ref.initial(), width)
            assert result.clean
            counts[width] = result.reachable
            assert not any(field_of(s, "occ") == 1 for s, _ in result.visited)  # nobody enters
        assert counts == {1: 200, 2: 496, 3: 792, 4: 992}

    def test_every_extra_node_of_the_amended_game_has_someone_inside_or_a_pending_entry(
        self,
    ) -> None:
        ref = renews_always()  # A9 alone: the reading between the spike and A11
        old = spike_reading()
        old_table = table_from_model(old, 1)
        old_nodes = {s for s, _ in check(old, old_table, old.initial(), 1).visited}
        new_table = table_from_model(ref, 1)
        new_nodes = {s for s, _ in check(ref, new_table, ref.initial(), 1).visited}
        assert len(new_nodes) == 216 and len(old_nodes) == 200
        assert old_nodes <= new_nodes
        extra = new_nodes - old_nodes
        assert len(extra) == 16
        assert all(field_of(s, "op") == "WORK" for s in extra)  # only the work state gains nodes


class TestDeadlocksAndTheDropHook:
    def test_emptying_every_entry_of_a_reachable_observable_is_a_deadlock(self) -> None:
        ref = crusher_reference()
        table = true_table("crusher", 2)
        first = check(ref, table, ref.initial(), 2)
        target = ref.observable(ref.initial())

        def empty_start(o: Hashable, box: tuple[int, ...], cmds: frozenset[str]) -> frozenset[str]:
            return frozenset() if o == target else cmds

        result = check(ref, edit_entries(table, empty_start), ref.initial(), 2)
        assert {f[0] for f in result.findings} == {"DEADLOCK"}
        assert result.reachable < first.reachable
        assert empty_entries(edit_entries(table, empty_start))

    def test_dropping_stop_at_a_running_observable_leaves_a_critical_crusher_stuck(self) -> None:
        ref = crusher_reference()
        table = true_table("crusher", 4)
        running = ref.observable(ref.initial())
        result = check(ref, table, ref.initial(), 4, drop=(running, "STOP"))
        assert any(f[0] == "DEADLOCK" for f in result.findings)

    def test_dropping_a_command_nobody_needs_changes_nothing_but_the_reach(self) -> None:
        ref = crusher_reference()
        table = true_table("crusher", 1)
        base = check(ref, table, ref.initial(), 1)
        cut = check(ref, table, ref.initial(), 1, drop=(("ISO", "NONE", 0, 3, 0, 0), "ISSUE"))
        assert cut.clean and cut.reachable <= base.reachable

    def test_a_conveyor_cannot_be_stuck_by_dropping_one_command_it_still_has_noop(self) -> None:
        ref = conveyor_reference()
        table = true_table("conveyor", 4)
        result = check(ref, table, ref.initial(), 4, drop=(("RUN",), "STOP"))
        assert result.clean  # failure is not a violation here, so STOP is not what saves it

    def test_dropping_every_command_at_a_conveyor_observable_is_a_deadlock(self) -> None:
        ref = conveyor_reference()
        table = true_table("conveyor", 1)

        def empty_run(o: Hashable, box: tuple[int, ...], cmds: frozenset[str]) -> frozenset[str]:
            return frozenset() if o == ("RUN",) else cmds

        result = check(ref, edit_entries(table, empty_run), ref.initial(), 1)
        assert {f[0] for f in result.findings} == {"DEADLOCK"}
        assert result.reachable == 1  # the start state and nothing after it


class TestTheCheckerAgreesWithARandomWalk:
    """An independent way to see the same fact: simulate, do not enumerate."""

    @staticmethod
    def walk(ref: ReferenceGame, table: Any, steps: int, seed: int, shielded: bool) -> int:
        rng = random.Random(seed)
        violations = 0
        s = ref.initial()
        for _ in range(steps):
            box: list[int] = []
            for b in health_vector(ref.health(s)):  # a random set that holds the true bins
                lo = rng.randint(max(0, b - 3), b)
                box += [lo, rng.randint(b, min(3, lo + 3))]
            allowed = table.allowed(ref.observable(s), box)
            c = rng.choice(sorted(allowed) if shielded else list(ref.commands()))
            t = rng.choice(sorted(ref.outcomes(s, c), key=repr))
            if t == VIOL:
                violations += 1
                t = ref.initial()
            s = t
        return violations

    @pytest.mark.parametrize("kind", KINDS)
    def test_shielded_walks_never_violate_and_unshielded_walks_do(self, kind: str) -> None:
        ref = reference_for(kind)
        table = true_table(kind, 4)
        assert self.walk(ref, table, 20_000, seed=7, shielded=True) == 0
        assert self.walk(ref, table, 5_000, seed=7, shielded=False) > 50


def test_check_result_is_a_frozen_value() -> None:
    result = CheckResult(1, ())
    assert result.clean and result.visited == frozenset()
    with pytest.raises(AttributeError):
        result.reachable = 2  # type: ignore[misc]


class TestTheRenewalChoice:
    """0003 A11: completing the work may or may not renew the component (a wrong-mode job
    completes and does not renew, 0003 A3). The winning region and every safe set are unchanged;
    only the set of states the shield can reach grows. (Stated on the game as it stood at A11.)"""

    def test_the_region_and_every_safe_set_are_those_of_the_always_renew_game(self) -> None:
        region_new, safe_new = solved(before_a12())
        region_old, safe_old = solved(renews_always())
        assert region_new == region_old and len(region_new) == 1693
        assert safe_new == safe_old

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_true_table_is_clean_against_both_readings_and_reach_grows(
        self, width: int
    ) -> None:
        always, either = renews_always(), before_a12()
        table = table_from_model(either, width)
        a = check(always, table, always.initial(), width)
        b = check(either, table, either.initial(), width)
        assert a.clean and b.clean
        assert a.reachable == {1: 216, 2: 536, 3: 856, 4: 1072}[width]
        assert b.reachable == BEFORE_A12_REACHABLE[width] > a.reachable
        assert a.visited <= b.visited

    @pytest.mark.parametrize("width", WIDTHS)
    def test_every_extra_node_is_an_isolated_crusher_that_was_not_renewed(self, width: int) -> None:
        always, either = renews_always(), before_a12()
        table = table_from_model(either, width)
        extra = (
            check(either, table, either.initial(), width).visited
            - check(always, table, always.initial(), width).visited
        )
        assert extra
        assert all(field_of(s, "op") == "ISO" and field_of(s, "h") >= 1 for s, _ in extra)


class TestThePermitLimitChangesTheReachAndNothingElse:
    """0003 A12.1. Predicted before any run, from the English: the winning region and every safe
    set are those of the game before A12, because the new outcomes (a spent permit with nobody
    inside) are winning states and the occupied cases were violations already. What changes is
    what the shield can reach."""

    def test_the_region_and_every_safe_set_are_those_of_the_game_before_a12(self) -> None:
        region_new, safe_new = solved(crusher_reference())
        region_old, safe_old = solved(before_a12())
        assert region_new == region_old and len(region_new) == 1693
        assert safe_new == safe_old

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_table_is_the_same_table_and_the_old_one_is_clean_against_the_new_relation(
        self, width: int
    ) -> None:
        new_table, old_table = true_table("crusher", width), table_from_model(before_a12(), width)
        assert dict(new_table.entries) == dict(old_table.entries)
        ref = crusher_reference()
        assert check(ref, old_table, ref.initial(), width).reachable == CRUSHER_REACHABLE[width]

    @pytest.mark.parametrize("width", WIDTHS)
    def test_reach_grows_by_sixteen_observable_states_at_all_four_bins(self, width: int) -> None:
        # the sets that hold a bin number w at bins 0 and 3 and 1 / 3 / 5 / 6 at bins 1 and 2, so
        # 64 new states cost 16 x (sum over the four bins) nodes: 64 / 160 / 256 / 320
        grown = CRUSHER_REACHABLE[width] - BEFORE_A12_REACHABLE[width]
        per_bin = {1: 4, 2: 10, 3: 16, 4: 20}[width]
        assert grown == 16 * per_bin

    def test_the_new_states_are_a_spent_permit_with_nobody_inside(self) -> None:
        ref, old = crusher_reference(), before_a12()
        new = {s for s, _ in check(ref, true_table("crusher", 1), ref.initial(), 1).visited}
        was = {s for s, _ in check(old, table_from_model(old, 1), old.initial(), 1).visited}
        assert was < new and len(new) - len(was) == 64
        extra = new - was
        assert {field_of(s, "permit") for s in extra} == {"NONE"}
        assert {field_of(s, "occ") for s in extra} == {0}
        assert {field_of(s, "op") for s in extra} == {"ISO", "WORK"}
        assert len({ref.observable(s) for s in extra}) == 16
        assert {field_of(s, "h") for s in extra} == {0, 1, 2, 3}


class TestTheIssueGuardReading:
    """The one place the first wording of A12.1 left room, settled since: the ADR now says ``ISSUE``
    works in ``WORK`` only with nobody inside, which is the reference's reading. The word-for-word
    reading of the first wording differs, and makes no difference to the proof: this pins exactly
    how little."""

    def test_the_two_readings_differ_at_four_raw_states_and_only_in_an_outcome_set(self) -> None:
        strict, literal = crusher_reference(), literal_issue()
        differing = [
            (s, c)
            for s in strict.states()
            for c in strict.commands()
            if strict.outcomes(s, c) != literal.outcomes(s, c)
        ]
        assert sorted(differing) == [(("WORK", h, "NONE", 1, 1, 0, 0), "ISSUE") for h in range(4)]
        for s, c in differing:
            assert VIOL in strict.outcomes(s, c) and VIOL in literal.outcomes(s, c)

    def test_the_winning_region_and_every_safe_set_are_the_same_under_both(self) -> None:
        assert solved(literal_issue()) == solved(crusher_reference())
