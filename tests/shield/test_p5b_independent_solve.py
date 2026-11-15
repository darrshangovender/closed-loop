"""The reference solved from scratch: winning region, safe sets and tables, then against production.

``check`` shows a table is safe. It does not show that the table is the *maximally permissive* safe
table, or that the region is the right one. These tests solve the reference's own game (a ten-line
greatest fixed point in ``p5b_support``) and state the results, most of them derived by hand from
the English before anything was run (0003 A12; the derivations are in the P5d note). Classes with
``Production`` in the name then compare with what the production solver and table builder shipped
and need the production models.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from itertools import product

import pytest

from closed_loop.shield.reference import VIOL
from tests.shield.p5b_support import (
    KINDS,
    WIDTHS,
    conveyor_reference,
    crusher_reference,
    field_of,
    independent_entries,
    independent_solve,
    production_winning,
    pump_reference,
    reference_for,
    shipped_table,
    solved,
    true_table,
)


@pytest.fixture(scope="module")
def crusher_solved() -> tuple[frozenset[Hashable], Mapping[Hashable, frozenset[str]]]:
    return solved(crusher_reference())


@pytest.fixture(scope="module")
def pump_solved() -> tuple[frozenset[Hashable], Mapping[Hashable, frozenset[str]]]:
    return solved(pump_reference())


@pytest.fixture(scope="module")
def conveyor_solved() -> tuple[frozenset[Hashable], Mapping[Hashable, frozenset[str]]]:
    return solved(conveyor_reference())


class TestCrusherRegion:
    """Predicted before the first run of the A12 reference: nothing changes (see the P5d note)."""

    def test_the_winning_region_has_the_size_it_had_before_a12(self, crusher_solved) -> None:
        region, _ = crusher_solved
        assert len(region) == 1693

    def test_the_states_that_lose_are_a_running_crusher_at_the_critical_bin_with_someone_in(
        self, crusher_solved
    ) -> None:
        region, _ = crusher_solved
        lost = [s for s in crusher_reference().states() if s not in region]
        assert len(lost) == 35
        assert {(field_of(s, "op"), field_of(s, "h"), field_of(s, "occ")) for s in lost} == {
            ("RUN", 3, 1)
        }

    def test_the_one_occupied_critical_state_that_wins_is_the_one_where_stop_keeps_the_rules(
        self, crusher_solved
    ) -> None:
        # 36 occupied, running, bin-3 states; only STOP stops the failure and then P2, P3 and P6
        # must hold after the tick: a valid permit with no handover, a test from the last tick,
        # and no alarm already on
        region, safe = crusher_solved
        winners = [
            s
            for s in region
            if (field_of(s, "op"), field_of(s, "h"), field_of(s, "occ")) == ("RUN", 3, 1)
        ]
        assert winners == [("RUN", 3, "VALID", 1, 1, 0, 0)]
        assert safe[winners[0]] == {"STOP"}

    def test_the_initial_state_wins_and_the_region_is_a_fixed_point(self, crusher_solved) -> None:
        region, safe = crusher_solved
        ref = crusher_reference()
        assert ref.initial() in region
        again = {
            s
            for s in ref.states()
            if any(
                VIOL not in (out := ref.outcomes(s, c)) and out <= region for c in ref.commands()
            )
        }
        assert again == set(region)
        assert all(safe[s] for s in region)

    def test_counts_reported_by_the_amendment(self, crusher_solved) -> None:
        region, safe = crusher_solved
        ref = crusher_reference()
        assert len({ref.observable(s) for s in region}) == 432
        assert sum(len(safe[s]) for s in region) == 10514
        assert round(10514 / len(region), 2) == 6.21

    def test_entry_is_safe_in_26_states_four_of_them_with_nobody_inside(
        self, crusher_solved
    ) -> None:
        # 0003 A9: the spike's version of this number was zero
        _, safe = crusher_solved
        enter = [s for s, cmds in safe.items() if "ENTER" in cmds]
        assert len(enter) == 26
        empty = [s for s in enter if field_of(s, "occ") == 0]
        assert len(empty) == 4
        assert {
            (field_of(s, "op"), field_of(s, "permit"), field_of(s, "gas"))
            + (field_of(s, "alarm"), field_of(s, "soon"))
            for s in empty
        } == {("WORK", "VALID", 1, 0, 0)}
        assert {field_of(s, "h") for s in empty} == {0, 1, 2, 3}

    def test_stop_is_what_the_shield_must_offer_at_the_critical_bin_while_running(
        self, crusher_solved
    ) -> None:
        _, safe = crusher_solved
        for s, cmds in safe.items():
            if field_of(s, "op") == "RUN" and field_of(s, "h") == 3:
                assert cmds == {"STOP"}

    def test_issue_is_safe_exactly_where_noop_is(self, crusher_solved) -> None:
        # 0003 A12.1: ISSUE works in WORK with nobody inside, and a refused ISSUE is a NOOP, so
        # ISSUE is safe exactly where NOOP is
        _, safe = crusher_solved
        for s, cmds in safe.items():
            assert ("ISSUE" in cmds) == ("NOOP" in cmds), s


class TestCrusherTables:
    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_table_from_the_definition_has_the_entries_it_had_before_a12(
        self, crusher_solved, width: int
    ) -> None:
        _, safe = crusher_solved
        entries = independent_entries(crusher_reference(), safe, width)
        assert len(entries) == {1: 1693, 2: 2989, 3: 3853, 4: 4285}[width]
        assert all(entries.values())  # no entry admits nothing

    def test_entries_are_the_intersection_of_the_safe_sets_of_the_winning_bins_in_the_set(
        self, crusher_solved
    ) -> None:
        _, safe = crusher_solved
        ref = crusher_reference()
        entries = independent_entries(ref, safe, 4)
        o = ("RUN", "NONE", 0, 3, 0, 0)  # a running crusher: at bin 3 only STOP is safe
        singles = {h: safe[(o[0], h, *o[1:])] for h in range(4)}
        assert singles[3] == {"STOP"} and singles[3] < singles[0]  # the bins differ: it can tell
        assert entries[(o, (0, 0))] == singles[0]
        assert entries[(o, (0, 3))] == singles[0] & singles[1] & singles[2] & singles[3] == {"STOP"}
        assert entries[(o, (1, 2))] == singles[1] & singles[2]


class _Chain:
    """a -go-> b, and b has only violations: a loses because of where it leads, not what it does."""

    name = "chain"

    def states(self) -> list[Hashable]:
        return ["a", "b", "c"]

    def commands(self) -> list[str]:
        return ["go", "stay"]

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        table = {
            ("a", "go"): {"b"},
            ("a", "stay"): {"a", VIOL},
            ("b", "go"): {VIOL},
            ("b", "stay"): {VIOL},
            ("c", "go"): {"a"},
            ("c", "stay"): {"c"},
        }
        return frozenset(table[(str(s), c)])


class TestTheSolverItself:
    def test_a_state_whose_only_violation_free_command_leads_to_a_lost_state_loses(self) -> None:
        # one round of "has a command with no violation" would keep a; the fixed point drops it,
        # and c, whose safe command is 'stay', wins because it never has to go anywhere
        region, safe = independent_solve(_Chain())  # type: ignore[arg-type]
        assert region == {"c"}
        assert safe == {"c": frozenset({"stay"})}


class TestPumpRegionAndTables:
    """Predicted before the first run, by hand, from the English of 0003 A12.2."""

    def test_the_winning_region_is_every_state_with_a_pump_that_is_running_or_stopped(
        self, pump_solved
    ) -> None:
        region, _ = pump_solved
        ref = pump_reference()
        assert len(ref.states()) == 576 and len(region) == 320
        assert region == {
            s for s in ref.states() if s[0] in ("RUN", "STOP") or s[2] in ("RUN", "STOP")
        }

    def test_the_states_that_lose_are_exactly_those_with_both_pumps_unavailable(
        self, pump_solved
    ) -> None:
        region, _ = pump_solved
        lost = {s for s in pump_reference().states() if s not in region}
        unavailable = ("LOCKED", "ISO", "WORK", "FAIL")
        assert lost == {
            s for s in pump_reference().states() if s[0] in unavailable and s[2] in unavailable
        }
        assert len(lost) == 16 * 16 == 256

    def test_observable_states_and_mean_safe_commands(self, pump_solved) -> None:
        region, safe = pump_solved
        ref = pump_reference()
        assert len({ref.observable(s) for s in region}) == 6 * 6 - 4 * 4 == 20
        total = sum(len(safe[s]) for s in region)
        assert total == 164**2 - 119**2 == 12735  # the hand derivation, below
        assert round(total / len(region), 2) == 39.80

    def test_every_safe_set_is_what_a_hand_count_of_each_pump_gives(self, pump_solved) -> None:
        """For each pump separately: which of its eight commands break a controller rule, and of
        the others which can leave it unavailable. A pair command is safe iff neither pump breaks a
        rule and the two cannot both end unavailable. So |safe| = N1 N2 - M1 M2."""
        _, safe = pump_solved
        eight = ("NOOP", "START", "STOP", "LOCK", "PROVE", "BEGIN", "COMPLETE", "UNISO")

        def breaks_a_rule(op: str, cmd: str) -> bool:
            return (cmd == "START" and op in ("LOCKED", "ISO", "WORK")) or (
                cmd == "BEGIN" and op in ("RUN", "STOP", "LOCKED", "FAIL")
            )

        def can_end_unavailable(op: str, h: int, cmd: str) -> bool:
            if op in ("LOCKED", "ISO", "WORK", "FAIL"):
                return True  # only STOP and NOOP are never refused and neither moves such a pump
            if op == "RUN":
                return h == 3 and cmd != "STOP"  # it may fail; only STOP is sure to stop it first
            return cmd == "LOCK" or (
                cmd == "START" and h == 3
            )  # STOP: lockable, or starts and fails

        def ok(op: str) -> list[str]:
            return [c for c in eight if not breaks_a_rule(op, c)]

        for s in pump_reference().states():
            (op1, h1, op2, h2) = s
            n1, n2 = ok(op1), ok(op2)
            expected = {
                f"{a}|{b}"
                for a in n1
                for b in n2
                if not (can_end_unavailable(op1, h1, a) and can_end_unavailable(op2, h2, b))
            }
            assert (safe[s] if s in safe else frozenset()) == expected, s

    def test_the_hand_totals_are_the_sums_of_the_per_pump_counts(self) -> None:
        # N is 7 at every op except LOCKED (6): 4 bins each. M is 0 for a running pump below the top
        # bin, 6 at the top, 1 for a stopped pump (2 at the top), and N for the four unavailable ops
        assert 5 * 4 * 7 + 4 * 6 == 164
        assert (3 * 0 + 6) + (3 * 1 + 2) + 4 * 6 + 4 * 7 + 4 * 7 + 4 * 7 == 119

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_table_from_the_definition_has_the_pinned_number_of_entries(
        self, pump_solved, width: int
    ) -> None:
        _, safe = pump_solved
        entries = independent_entries(pump_reference(), safe, width)
        sets = {1: 4, 2: 7, 3: 9, 4: 10}[width]  # sets of at most `width` bins in four bins
        # 20 observable states, a set per pump, and every pair of sets holds a winning pair of bins
        # (winning depends on the ops alone), so 20 x sets^2: 320 / 980 / 1,620 / 2,000
        assert len(entries) == 20 * sets**2
        assert all(entries.values())

    def test_the_lone_available_pump_at_the_top_bin_may_only_be_stopped(self, pump_solved) -> None:
        _, safe = pump_solved
        for op in ("LOCKED", "ISO", "WORK", "FAIL"):
            s = (op, 0, "RUN", 3)
            assert {c.split("|")[1] for c in safe[s]} == {"STOP"}, op


class TestConveyorRegionAndTables:
    """Predicted before the first run, by hand, from the English of 0003 A12.3."""

    def test_every_state_wins_because_nothing_but_the_two_controller_rules_is_a_violation(
        self, conveyor_solved
    ) -> None:
        region, _ = conveyor_solved
        assert len(conveyor_reference().states()) == 24 and len(region) == 24

    def test_six_observable_states_and_the_hand_counted_mean(self, conveyor_solved) -> None:
        region, safe = conveyor_solved
        ref = conveyor_reference()
        assert len({ref.observable(s) for s in region}) == 6
        total = sum(len(safe[s]) for s in region)
        assert total == 4 * (5 * 7 + 6) == 164
        assert round(total / 24, 2) == 6.83

    def test_the_safe_set_is_every_command_but_the_two_that_break_a_rule_and_ignores_the_bin(
        self, conveyor_solved
    ) -> None:
        _, safe = conveyor_solved
        eight = {"NOOP", "START", "STOP", "LOCK", "PROVE", "BEGIN", "COMPLETE", "UNISO"}
        for (op, h), cmds in safe.items():
            expected = set(eight)
            if op in ("LOCKED", "ISO", "WORK"):
                expected.discard("START")
            if op in ("RUN", "STOP", "LOCKED", "FAIL"):
                expected.discard("BEGIN")
            assert cmds == expected, (op, h)

    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_table_has_six_observable_states_times_the_sets_of_that_width(
        self, conveyor_solved, width: int
    ) -> None:
        _, safe = conveyor_solved
        entries = independent_entries(conveyor_reference(), safe, width)
        assert len(entries) == 6 * {1: 4, 2: 7, 3: 9, 4: 10}[width]
        # the bin does not matter to a conveyor, so every entry is its op's safe set
        for (o, _box), cmds in entries.items():
            assert cmds == safe[(o[0], 0)]


@pytest.mark.slow
class TestStaleSpec:
    """The same comparison for the superseded 2-hour interval (8 quarters)."""

    def test_region_and_tables_at_a_period_of_eight(self) -> None:
        ref = crusher_reference(8)
        region, safe = solved(ref)
        assert len(ref.states()) == 5184
        assert len(region) == 5083
        assert len({ref.observable(s) for s in region}) == 1296
        assert round(sum(len(safe[s]) for s in region) / len(region), 2) == 6.38


class TestProductionAgreesOnTheRegionAndTheSafeSets:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_region_and_every_safe_set_are_the_ones_production_found(self, kind: str) -> None:
        region, safe = solved(reference_for(kind))
        win = production_winning(kind)
        assert region == win.states
        assert dict(safe) == dict(win.safe)

    @pytest.mark.slow
    def test_the_stale_spec_region_and_safe_sets_match_at_a_period_of_eight(self) -> None:
        from closed_loop.shield.game import solve
        from tests.shield.p5b_support import crusher_model

        win = solve(crusher_model(8))
        region, safe = solved(crusher_reference(8))
        assert region == win.states and dict(safe) == dict(win.safe)


class TestShippedTablesAreTheTablesFromTheDefinition:
    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("width", WIDTHS)
    def test_the_shipped_table_is_exactly_the_table_from_the_definition(
        self, kind: str, width: int
    ) -> None:
        ref = reference_for(kind)
        _, safe = solved(ref)
        assert independent_entries(ref, safe, width) == dict(shipped_table(kind, width).entries)
        assert independent_entries(ref, safe, width) == dict(true_table(kind, width).entries)

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("width", [1, 2, 3])
    def test_a_set_wider_than_the_table_was_built_for_is_answered_exactly(
        self, kind: str, width: int
    ) -> None:
        # `allowed` covers a wide set with narrow windows; the answer must still be the
        # intersection over every winning bin in the set, which the width-4 definition gives
        ref = reference_for(kind)
        _, safe = solved(ref)
        full = independent_entries(ref, safe, 4)
        narrow = shipped_table(kind, width)
        intervals = [(lo, hi) for lo in range(4) for hi in range(lo, 4)]
        dims = len(next(iter(full))[1]) // 2
        for o in {o for o, _ in full}:
            for boxes in product(intervals, repeat=dims):
                box = tuple(v for lo_hi in boxes for v in lo_hi)
                assert narrow.allowed(o, box) == full.get((o, box), frozenset())

    def test_the_priority_order_starts_with_stop_exit_noop(self) -> None:
        priority = shipped_table("crusher", 1).priority
        assert priority[:3] == ("STOP", "EXIT", "NOOP")
        assert sorted(priority) == sorted(crusher_reference().commands())
