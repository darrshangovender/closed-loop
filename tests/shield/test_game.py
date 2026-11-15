"""The solver: greatest fixed point, maximal permissiveness, and the numbers it produces."""

from __future__ import annotations

from collections.abc import Hashable

import pytest

from closed_loop.shield.abstraction import (
    VIOL,
    ConveyorModel,
    CrusherJobModel,
    PumpPairModel,
)
from closed_loop.shield.game import Winning, canonical, solve, winning_operator
from tests.shield.brute import TinyGame


class TestTinyGames:
    def test_one_losing_outcome_among_winning_ones_loses_the_command(self) -> None:
        # regression: a command is safe only if EVERY outcome is winning, not if any is
        game = TinyGame({("a", "split"): ["a", "d"], ("d", "go"): [VIOL]})
        win = solve(game)
        assert win.states == frozenset()  # d loses, so the only command of a has a losing outcome
        both = TinyGame({("a", "split"): ["a", "d"], ("a", "hold"): ["a"], ("d", "go"): [VIOL]})
        assert solve(both).safe["a"] == {"hold"}

    def test_the_digest_sees_an_outcome_change_that_keeps_names_states_and_commands(self) -> None:
        # regression: the digest is what lets a table say which abstraction it came from
        base = {("a", "go"): ["a"], ("a", "alt"): ["a", "b"], ("b", "go"): ["a"]}
        changed = {**base, ("a", "alt"): ["a"]}
        one, two = solve(TinyGame(base, name="t")), solve(TinyGame(changed, name="t"))
        assert one.relation_sha256 != two.relation_sha256
        assert solve(TinyGame(base, name="t")).relation_sha256 == one.relation_sha256

    def test_a_safe_cycle_wins(self) -> None:
        # a least fixed point would drop the cycle (it never reaches a goal); a safety game keeps it
        game = TinyGame({("a", "go"): ["b"], ("b", "go"): ["a"]})
        win = solve(game)
        assert win.states == {"a", "b"}
        assert win.safe["a"] == {"go"} and win.safe["b"] == {"go"}
        assert win.iterations == 1

    def test_a_state_whose_only_command_is_unsafe_loses_and_drags_predecessors(self) -> None:
        game = TinyGame(
            {
                ("a", "go"): ["b"],
                ("a", "stay"): ["a"],
                ("b", "go"): ["c"],
                ("c", "go"): [VIOL],
            }
        )
        win = solve(game)
        assert win.states == {"a"}  # c loses, so b loses, a survives by staying
        assert win.safe["a"] == {"stay"}
        assert win.iterations == 3  # remove c; remove b; confirm

    def test_an_environment_choice_counts_against_the_command(self) -> None:
        game = TinyGame({("a", "risky"): ["a", VIOL], ("a", "sure"): ["a"]})
        assert solve(game).safe["a"] == {"sure"}

    def test_no_winning_state_is_an_empty_region_not_an_error(self) -> None:
        game = TinyGame({("a", "go"): [VIOL]})
        win = solve(game)
        assert win.states == frozenset() and win.safe == {}

    def test_winning_region_is_immutable(self) -> None:
        win = solve(TinyGame({("a", "go"): ["a"]}))
        with pytest.raises(TypeError):
            win.safe["z"] = frozenset()  # type: ignore[index]


def naive_solve(model: CrusherJobModel | PumpPairModel) -> set[Hashable]:
    """The textbook loop, written independently of ``solve`` (no cache, no rounds)."""
    region = set(model.states())
    changed = True
    while changed:
        changed = False
        for s in list(region):
            if not any(
                all(t != VIOL and t in region for t in model.step(s, c)) for c in model.commands()
            ):
                region.discard(s)
                changed = True
    return region


class TestCrusherSolution:
    def test_measured_counts(self, crusher: CrusherJobModel, crusher_win: Winning) -> None:
        assert len(list(crusher.states())) == 1728
        assert len(crusher_win.states) == 1693
        assert len({crusher.observable(s) for s in crusher_win.states}) == 432
        mean_safe = sum(len(v) for v in crusher_win.safe.values()) / len(crusher_win.safe)
        assert round(mean_safe, 2) == 6.21
        assert crusher_win.iterations == 2
        assert crusher.initial() in crusher_win.states

    def test_the_states_that_lose_are_the_running_ones_with_someone_inside(
        self, crusher: CrusherJobModel, crusher_win: Winning
    ) -> None:
        losing = [s for s in crusher.states() if s not in crusher_win.states]
        assert len(losing) == 35
        assert {s[0] for s in losing} == {"RUN"}  # type: ignore[index]
        assert all(s[3] == 1 for s in losing)  # type: ignore[index]

    def test_the_region_is_a_fixed_point_of_the_operator(
        self, crusher: CrusherJobModel, crusher_win: Winning
    ) -> None:
        assert winning_operator(crusher, crusher_win.states) == crusher_win.states

    def test_the_region_is_the_greatest_one(
        self, crusher: CrusherJobModel, crusher_win: Winning
    ) -> None:
        assert naive_solve(crusher) == set(crusher_win.states)

    def test_every_safe_command_stays_inside_and_every_other_leaves(
        self, crusher: CrusherJobModel, crusher_win: Winning
    ) -> None:
        for s in crusher_win.states:
            for c in crusher.commands():
                inside = all(t != VIOL and t in crusher_win.states for t in crusher.step(s, c))
                assert (c in crusher_win.safe[s]) == inside, (s, c)

    def test_every_winning_state_has_a_safe_command(self, crusher_win: Winning) -> None:
        assert all(crusher_win.safe[s] for s in crusher_win.states)

    def test_stop_is_the_only_safe_command_for_a_running_crusher_at_bin_3(
        self, crusher_win: Winning
    ) -> None:
        running_critical = [s for s in crusher_win.states if s[0] == "RUN" and s[1] == 3]  # type: ignore[index]
        assert running_critical
        assert all(crusher_win.safe[s] == {"STOP"} for s in running_critical)

    def test_the_relation_digest_depends_on_the_relation(self, crusher_win: Winning) -> None:
        other = solve(CrusherJobModel(retest_q=3))
        assert len(crusher_win.relation_sha256) == 64
        assert other.relation_sha256 != crusher_win.relation_sha256
        assert solve(CrusherJobModel()).relation_sha256 == crusher_win.relation_sha256


class TestPumpSolution:
    def test_measured_counts(self, pumps: PumpPairModel, pumps_win: Winning) -> None:
        # ADR 0003 A12.2: a locked pump is a state (576 pairs) with eight commands (64 pairs), and
        # a pair with both pumps in LOCKED, ISO, WORK or FAIL loses (it was 400, 49 and 256)
        assert len(list(pumps.states())) == 576
        assert len(pumps.commands()) == 64
        assert len(pumps_win.states) == 320
        assert pumps.initial() in pumps_win.states

    def test_fixed_point_and_greatest(self, pumps: PumpPairModel, pumps_win: Winning) -> None:
        assert winning_operator(pumps, pumps_win.states) == pumps_win.states
        assert naive_solve(pumps) == set(pumps_win.states)

    def test_safe_sets_are_exact(self, pumps: PumpPairModel, pumps_win: Winning) -> None:
        for s in pumps_win.states:
            for c in pumps.commands():
                inside = all(t != VIOL and t in pumps_win.states for t in pumps.step(s, c))
                assert (c in pumps_win.safe[s]) == inside, (s, c)

    def test_the_second_pump_is_pinned_while_the_first_is_under_maintenance(
        self, pumps_win: Winning
    ) -> None:
        # pump 1 in WORK, pump 2 running at bin 3: the only safe joint commands stop pump 2
        s = ("WORK", 0, "RUN", 3)
        assert s in pumps_win.states
        assert all(c.endswith("|STOP") for c in pumps_win.safe[s])


class TestConveyorSolution:
    def test_measured_counts(self, conveyor: ConveyorModel, conveyor_win: Winning) -> None:
        assert len(list(conveyor.states())) == 24 and len(conveyor.commands()) == 8
        assert len(conveyor_win.states) == 24 and conveyor_win.iterations == 1
        assert conveyor.initial() in conveyor_win.states

    def test_fixed_point_and_greatest(self, conveyor: ConveyorModel, conveyor_win: Winning) -> None:
        assert winning_operator(conveyor, conveyor_win.states) == conveyor_win.states
        assert naive_solve(conveyor) == set(conveyor_win.states)

    def test_safe_sets_are_exact(self, conveyor: ConveyorModel, conveyor_win: Winning) -> None:
        for s in conveyor_win.states:
            for c in conveyor.commands():
                inside = all(t != VIOL and t in conveyor_win.states for t in conveyor.step(s, c))
                assert (c in conveyor_win.safe[s]) == inside, (s, c)

    def test_what_is_unsafe_is_start_under_lock_and_begin_without_proof(
        self, conveyor_win: Winning
    ) -> None:
        for (op, _h), safe in conveyor_win.safe.items():
            assert ("START" not in safe) == (op in ("LOCKED", "ISO", "WORK")), op
            assert ("BEGIN" not in safe) == (op in ("RUN", "STOP", "LOCKED", "FAIL")), op
            assert {"NOOP", "STOP"} <= safe, op  # never refused, never a violation


class TestCanonical:
    def test_sets_and_tuples_have_one_form(self) -> None:
        assert canonical(frozenset({("b", 1), ("a", 2)})) == [["a", 2], ["b", 1]]
        assert canonical(("x", (1, 2))) == ["x", [1, 2]]

    def test_unsupported_values_are_refused(self) -> None:
        with pytest.raises(TypeError):
            canonical(1.5)
