"""Differential test: a model's relation against the independent reference, state by state.

The production models and the reference were written by different engineers, each without reading
the other's code, from the same English (ADR 0003 A12 for the latest round). Where they disagree the
English decides which is wrong. ``TestProductionAgreesWithTheReference`` is the one class that needs
the production models; the rest show, with the reference on both sides, that the comparison reports
what differs and looks everywhere (see also ``test_p5b_sabotage``).
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import pytest

from closed_loop.shield.checker import differential, structural_differences
from closed_loop.shield.reference import VIOL, CrusherReference, PumpPairReference
from tests.shield.p5b_support import (
    KINDS,
    crusher_reference,
    model_for,
    pump_reference,
    reference_for,
)


class TestProductionAgreesWithTheReference:
    @pytest.mark.parametrize("kind", KINDS)
    def test_the_relation_is_identical_on_every_state_and_command(self, kind: str) -> None:
        assert differential(model_for(kind), reference_for(kind)) == []

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_shape_is_identical(self, kind: str) -> None:
        assert structural_differences(model_for(kind), reference_for(kind)) == []

    def test_the_pump_pair_compares_all_sixty_four_commands_and_the_conveyor_all_eight(
        self,
    ) -> None:
        pumps, conveyor = model_for("pumps"), model_for("conveyor")
        assert len(list(pumps.commands())) == 64 and len(list(pumps.states())) == 576
        assert len(list(conveyor.commands())) == 8 and len(list(conveyor.states())) == 24
        assert "LOCK|PROVE" in pumps.commands() and not any(
            "ISOLATE" in c for c in pumps.commands()
        )

    @pytest.mark.parametrize("retest_q", [1, pytest.param(8, marks=pytest.mark.slow)])
    def test_the_agreement_holds_at_other_retest_periods(self, retest_q: int) -> None:
        # 8 is the stale-spec period (the 2-hour interval of the superseded revision)
        from tests.shield.p5b_support import crusher_model

        model = crusher_model(retest_q)
        ref = crusher_reference(retest_q)
        assert structural_differences(model, ref) == []
        assert differential(model, ref) == []


class _Subset:
    """A game restricted to some states (for an exact comparison of a slice)."""

    def __init__(self, base: Any, keep: Iterable[Hashable]) -> None:
        self.base = base
        self.name = base.name
        self._keep = set(keep)

    def states(self) -> Iterable[Hashable]:
        return (s for s in self.base.states() if s in self._keep)

    def commands(self) -> Sequence[str]:
        return self.base.commands()

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self.base.step(s, c)

    def initial(self) -> Hashable:
        return self.base.initial()

    def observable(self, s: Hashable) -> Hashable:
        return self.base.observable(s)

    def health(self, s: Hashable) -> int | tuple[int, ...]:
        return self.base.health(s)


class _Outcomes:
    """Only the relation of a game: it lists no states and no commands of its own."""

    def __init__(self, base: Any) -> None:
        self.base = base

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self.base.outcomes(s, c)


class _Perturbed:
    """A reference that reports one extra outcome at exactly one (state, command)."""

    def __init__(self, base: Any, at: tuple[Hashable, str]) -> None:
        self.base = base
        self.at = at

    def outcomes(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        out = self.base.outcomes(s, c)
        return out | {"EXTRA"} if (s, c) == self.at else out


class _Forgetful(_Subset):
    """A model that does not list one command and refuses it, as a model that forgot it would."""

    def __init__(self, base: Any, forgotten: str) -> None:
        super().__init__(base, base.states())
        self.forgotten = forgotten

    def commands(self) -> Sequence[str]:
        return [c for c in self.base.commands() if c != self.forgotten]

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c == self.forgotten:
            raise ValueError(f"unknown command {c!r}")
        return self.base.step(s, c)


class _Stubborn(_Subset):
    """A model that still lists a command the reference has retired, and answers it."""

    def __init__(self, base: Any, extra: str) -> None:
        super().__init__(base, base.states())
        self.extra = extra

    def commands(self) -> Sequence[str]:
        return [*self.base.commands(), self.extra]

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return frozenset({s}) if c == self.extra else self.base.step(s, c)


class TestTheComparisonReportsWhatDiffers:
    def test_a_reference_with_one_rule_removed_is_reported_with_both_outcome_sets(self) -> None:
        class NoStaleRule(CrusherReference):
            def _p3_violated(self, after: tuple) -> bool:  # type: ignore[override]
                return False

        found = differential(crusher_reference(), NoStaleRule())
        assert found
        _s, c, theirs, mine = found[0]
        assert isinstance(c, str) and theirs != mine
        # what the reference now lets through is an occupied state with a stale test
        extra = mine - theirs
        assert extra and all(t[3] == 1 and t[4] == 3 for t in extra)
        # the rule bites only when someone is already inside
        assert all(state[3] == 1 for state, *_ in found)

    def test_limit_stops_the_comparison_early(self) -> None:
        class NoStaleRule(CrusherReference):
            def _p3_violated(self, after: tuple) -> bool:  # type: ignore[override]
                return False

        assert len(differential(crusher_reference(), NoStaleRule(), limit=3)) == 3
        assert len(differential(crusher_reference(), NoStaleRule(), limit=1)) == 1

    def test_a_state_the_reference_does_not_accept_is_a_difference_not_a_crash(self) -> None:
        # a period of 8 has gas values 4..9 that a period-2 reference rejects
        found = differential(crusher_reference(8), crusher_reference(2), limit=1000)
        assert found
        rejected = [m for *_, m in found if any("REFERENCE REJECTS" in str(x) for x in m)]
        assert rejected
        assert structural_differences(crusher_reference(8), crusher_reference(2))

    def test_structure_notices_a_different_command_order_and_initial_state(self) -> None:
        class Reordered(CrusherReference):
            def commands(self) -> tuple[str, ...]:
                return tuple(reversed(super().commands()))

            def initial(self) -> Hashable:
                return ("STOP", 0, "NONE", 0, 3, 0, 0)

        problems = structural_differences(crusher_reference(), Reordered())
        assert any(p.startswith("commands differ") for p in problems)
        assert any(p.startswith("initial differs") for p in problems)

    def test_structure_notices_a_swapped_state_of_the_same_count_and_a_different_order(
        self,
    ) -> None:
        class SwappedState(CrusherReference):
            def states(self) -> list[Hashable]:
                return [*super().states()[:-1], ("RUN", 0, "NONE", 0, 3, 0, 2)]

        class Shuffled(CrusherReference):
            def states(self) -> list[Hashable]:
                return list(reversed(super().states()))

        swapped = structural_differences(crusher_reference(), SwappedState())
        assert len(SwappedState().states()) == len(crusher_reference().states())
        assert any(p.startswith("states differ") for p in swapped)
        shuffled = structural_differences(crusher_reference(), Shuffled())
        assert any(p.startswith("states differ") for p in shuffled)

    def test_structure_says_when_the_states_are_the_same_but_in_a_different_order(self) -> None:
        class Shuffled(PumpPairReference):
            def states(self) -> list[Hashable]:
                return list(reversed(super().states()))

        class Fewer(PumpPairReference):
            def states(self) -> list[Hashable]:
                return super().states()[:-1]

        (order,) = structural_differences(pump_reference(), Shuffled())
        assert "different order" in order and "only in" not in order
        (missing,) = structural_differences(pump_reference(), Fewer())
        assert "only in model" in missing and "different order" not in missing

    def test_structure_notices_a_changed_observable_and_a_changed_health(self) -> None:
        class KeepsSoon(CrusherReference):
            def observable(self, s: Hashable) -> Hashable:
                return super().observable(s)[:-1]  # type: ignore[index]

            def health(self, s: Hashable) -> int:
                return 0

        problems = structural_differences(crusher_reference(), KeepsSoon())
        assert any("observable" in p for p in problems)
        assert any("health" in p for p in problems)

    def test_pump_pair_reference_with_a_changed_rule_is_reported(self) -> None:
        class IsoAndLockedAreAvailable(PumpPairReference):
            def _p4_violated(self, op1: str, op2: str) -> bool:
                return {op1, op2} <= {"WORK", "FAIL"}

        found = differential(pump_reference(), IsoAndLockedAreAvailable(), limit=5)
        assert found and all(VIOL in theirs for _s, _c, theirs, _m in found)

    def test_subset_comparison_is_exact_on_the_slice(self) -> None:
        ref = crusher_reference()
        slice_ = [s for s in ref.states() if s[0] in ("WORK", "ISO") and s[3] == 1]
        assert slice_
        inside, outside = slice_[len(slice_) // 2], ("RUN", 0, "NONE", 0, 3, 0, 0)
        model = _Subset(ref, slice_)
        assert differential(model, _Outcomes(ref)) == []
        # a difference inside the slice is found, one outside it is not looked at
        assert len(differential(model, _Perturbed(ref, (inside, "NOOP")))) == 1
        assert differential(model, _Perturbed(ref, (outside, "NOOP"))) == []


class TestTheDifferentialCoversEverythingEitherSideLists:
    """0003 A12.2 changes the pump's command list. A model that forgot a command must be reported
    and not skipped, and so must one that kept a command the reference has retired."""

    def test_a_command_the_model_forgot_is_compared_and_reported_for_every_state(self) -> None:
        ref = pump_reference()
        found = differential(_Forgetful(ref, "NOOP|LOCK"), _Outcomes(ref))
        assert found == []  # an outcomes-only reference lists nothing of its own: nothing to add
        found = differential(_Forgetful(ref, "NOOP|LOCK"), ref)
        assert len(found) == 576
        assert {c for _s, c, *_ in found} == {"NOOP|LOCK"}
        assert all(any("MODEL REJECTS" in str(x) for x in theirs) for *_, theirs, _m in found)

    def test_a_command_the_reference_retired_is_reported_for_every_state(self) -> None:
        ref = pump_reference()
        found = differential(_Stubborn(ref, "ISOLATE|NOOP"), ref)
        assert len(found) == 576
        assert {c for _s, c, *_ in found} == {"ISOLATE|NOOP"}
        assert all(any("REFERENCE REJECTS" in str(x) for x in mine) for *_, mine in found)

    def test_a_state_the_model_forgot_is_compared_and_reported_for_every_command(self) -> None:
        ref = pump_reference()
        lost = ("LOCKED", 0, "RUN", 0)

        class Refusing(_Subset):
            def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
                if s == lost:
                    raise ValueError("no such state")
                return self.base.step(s, c)

        found = differential(Refusing(ref, [s for s in ref.states() if s != lost]), ref)
        assert len(found) == 64 and {s for s, *_ in found} == {lost}

    def test_a_model_that_raises_is_reported_and_does_not_end_the_comparison(self) -> None:
        ref = pump_reference()

        class Broken(_Subset):
            def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
                if c == "STOP|STOP":
                    raise KeyError(c)
                return self.base.step(s, c)

        found = differential(Broken(ref, ref.states()), ref)
        assert len(found) == 576 and {c for _s, c, *_ in found} == {"STOP|STOP"}
        assert all("KeyError" in str(theirs) for *_, theirs, _m in found)


class TestTheDifferentialLooksEverywhere:
    """One planted difference at the first / last state and the first / last command must be the
    only thing reported: nothing is skipped and nothing else is flagged."""

    @pytest.mark.parametrize("which_state", ["first", "middle", "last"])
    @pytest.mark.parametrize("which_command", ["first", "middle", "last"])
    @pytest.mark.parametrize("kind", KINDS)
    def test_a_single_planted_difference_is_found_exactly(
        self, kind: str, which_state: str, which_command: str
    ) -> None:
        ref = reference_for(kind)
        states, commands = list(ref.states()), list(ref.commands())
        s = {"first": states[0], "middle": states[len(states) // 2], "last": states[-1]}[
            which_state
        ]
        c = {"first": commands[0], "middle": commands[len(commands) // 2], "last": commands[-1]}[
            which_command
        ]
        found = differential(ref, _Perturbed(ref, (s, c)))
        assert len(found) == 1
        got_s, got_c, theirs, mine = found[0]
        assert (got_s, got_c) == (s, c)
        assert mine == theirs | {"EXTRA"}
