"""Mutation honesty for the shield (0002 section 1, rule 9): sabotage the model or the table and
confirm the solver or the table tests notice.

Each sabotaged model is solved and a table is built from it as if it were the truth. The table is
then judged against the true model with the brute-force check, and compared with the true table.
These are tests of P5a's own solver and table; the independent checker of P5b repeats the idea
with a model written from the English spec.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Hashable, Iterable, Sequence
from itertools import product

import pytest

from closed_loop.shield.abstraction import (
    VIOL,
    CrusherJobModel,
    PumpPairModel,
    _pump_outcomes,
)
from closed_loop.shield.game import Winning, solve
from closed_loop.shield.table import ShieldTable, build_shield, matches_model
from tests.shield.brute import critical_run_violations, unsafe_allowed

Step = Callable[[Hashable, str], frozenset[Hashable | str]]


class Sabotaged:
    """A model that delegates everything to ``base`` except ``step``."""

    def __init__(self, base: CrusherJobModel | PumpPairModel, label: str, step: Step) -> None:
        self.base = base
        self.name = f"{base.name}+{label}"
        self._step = step

    def states(self) -> Iterable[Hashable]:
        return self.base.states()

    def commands(self) -> Sequence[str]:
        return self.base.commands()

    def initial(self) -> Hashable:
        return self.base.initial()

    def observable(self, s: Hashable) -> Hashable:
        return self.base.observable(s)

    def health(self, s: Hashable) -> int | tuple[int, ...]:
        return self.base.health(s)

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return self._step(s, c)

    def parameters(self) -> dict[str, str]:
        return dict(self.base.parameters())


def _with(s: Hashable, index: int, value: object) -> Hashable:
    items = list(s)  # type: ignore[call-overload]
    items[index] = value
    return tuple(items)


def crusher_sabotages(base: CrusherJobModel) -> dict[str, Step]:
    def drop_refusal(s: Hashable, c: str) -> frozenset[Hashable | str]:
        # every command is always executed: the refusal branch of A_exec is gone
        return frozenset(base._execute(s, c))

    def enter_ignores_alarm(s: Hashable, c: str) -> frozenset[Hashable | str]:
        return base.step(_with(s, 5, 0), c) if c == "ENTER" else base.step(s, c)

    def begin_without_proof(s: Hashable, c: str) -> frozenset[Hashable | str]:
        if c == "BEGIN" and s[0] == "LOCKED":  # type: ignore[index]
            return base.step(_with(s, 0, "ISO"), c)  # zero energy taken as proven
        return base.step(s, c)

    def running_critical_never_fails(s: Hashable, c: str) -> frozenset[Hashable | str]:
        out = base.step(s, c)
        if s[0] == "RUN" and s[1] == 3 and c in ("NOOP", "START", "TEST", "LOCK"):  # type: ignore[index]
            return frozenset(out - {VIOL})  # the uncontrolled-failure transition is removed
        return out

    def handover_keeps_the_permit(s: Hashable, c: str) -> frozenset[Hashable | str]:
        return base.step(_with(s, 6, 0), c)  # the announcement never matters

    def retest_never_goes_stale(s: Hashable, c: str) -> frozenset[Hashable | str]:
        return base.step(_with(s, 4, 1), c) if s[3] == 1 else base.step(s, c)  # type: ignore[index]

    return {
        "drop_refusal": drop_refusal,
        "enter_ignores_alarm": enter_ignores_alarm,
        "begin_without_proof": begin_without_proof,
        "running_critical_never_fails": running_critical_never_fails,
        "handover_keeps_the_permit": handover_keeps_the_permit,
        "retest_never_goes_stale": retest_never_goes_stale,
    }


def pump_sabotages(base: PumpPairModel) -> dict[str, Step]:
    def no_availability_check(s: Hashable, c: str) -> frozenset[Hashable | str]:
        op1, h1, op2, h2 = s  # type: ignore[misc]
        a1, a2 = c.split("|")
        out: set[Hashable | str] = set()
        for t1, t2 in product(_pump_outcomes(op1, h1, a1), _pump_outcomes(op2, h2, a2)):
            out.add(VIOL if isinstance(t1, str) or isinstance(t2, str) else t1 + t2)
        return frozenset(out)

    def drop_refusal(s: Hashable, c: str) -> frozenset[Hashable | str]:
        op1, h1, op2, h2 = s  # type: ignore[misc]
        a1, a2 = c.split("|")
        # the refused branch of each pump is removed by asking for the command twice as forced:
        # only the all-executed combination is kept
        out = base.step(s, c)
        refused_1 = _pump_outcomes(op1, h1, "NOOP")
        refused_2 = _pump_outcomes(op2, h2, "NOOP")
        executed_1 = _pump_outcomes(op1, h1, a1) - (
            refused_1 if a1 not in ("NOOP", "STOP") else set()
        )
        executed_2 = _pump_outcomes(op2, h2, a2) - (
            refused_2 if a2 not in ("NOOP", "STOP") else set()
        )
        kept: set[Hashable | str] = set()
        for t1, t2 in product(executed_1 or refused_1, executed_2 or refused_2):
            if isinstance(t1, str) or isinstance(t2, str):
                kept.add(VIOL)
            elif t1[0] in ("ISO", "WORK", "FAIL") and t2[0] in ("ISO", "WORK", "FAIL"):
                kept.add(VIOL)
            else:
                kept.add(t1 + t2)
        return frozenset(kept) if kept else out

    return {"no_availability_check": no_availability_check, "drop_refusal": drop_refusal}


@pytest.fixture(scope="module")
def sabotaged_crushers(
    crusher: CrusherJobModel,
) -> dict[str, tuple[Sabotaged, Winning, ShieldTable]]:
    out = {}
    for label, step in crusher_sabotages(crusher).items():
        model = Sabotaged(crusher, label, step)
        win = solve(model)
        out[label] = (model, win, build_shield(model, win, 4))
    return out


@pytest.fixture(scope="module")
def sabotaged_pumps(pumps: PumpPairModel) -> dict[str, tuple[Sabotaged, Winning, ShieldTable]]:
    out = {}
    for label, step in pump_sabotages(pumps).items():
        model = Sabotaged(pumps, label, step)
        win = solve(model)
        out[label] = (model, win, build_shield(model, win, 4))
    return out


CRUSHER_LABELS = list(crusher_sabotages(CrusherJobModel()))
PUMP_LABELS = list(pump_sabotages(PumpPairModel()))


class TestSabotagedCrusher:
    @pytest.mark.parametrize("label", CRUSHER_LABELS)
    def test_every_sabotage_changes_the_solution_the_hash_and_the_rebuild_check(
        self,
        label: str,
        sabotaged_crushers: dict[str, tuple[Sabotaged, Winning, ShieldTable]],
        crusher: CrusherJobModel,
        crusher_win: Winning,
        crusher_tables: dict[int, ShieldTable],
    ) -> None:
        _model, win, table = sabotaged_crushers[label]
        assert win.relation_sha256 != crusher_win.relation_sha256  # the relation digest sees it
        assert dict(win.safe) != dict(crusher_win.safe)  # the solver's answer moves
        assert table.sha256() != crusher_tables[4].sha256()
        assert not matches_model(table, crusher)

    @pytest.mark.parametrize("label", CRUSHER_LABELS)
    def test_the_brute_force_table_check_turns_red(
        self,
        label: str,
        sabotaged_crushers: dict[str, tuple[Sabotaged, Winning, ShieldTable]],
        crusher: CrusherJobModel,
        crusher_win: Winning,
    ) -> None:
        _model, _win, table = sabotaged_crushers[label]
        assert unsafe_allowed(table, crusher, crusher_win), label

    def test_the_true_table_passes_the_same_check(
        self,
        crusher: CrusherJobModel,
        crusher_win: Winning,
        crusher_tables: dict[int, ShieldTable],
    ) -> None:
        assert unsafe_allowed(crusher_tables[4], crusher, crusher_win) == []

    def test_enter_ignoring_the_alarm_admits_an_entry_into_an_alarmed_space(
        self, sabotaged_crushers: dict[str, tuple[Sabotaged, Winning, ShieldTable]]
    ) -> None:
        _m, _w, table = sabotaged_crushers["enter_ignores_alarm"]
        alarmed_entry = ("WORK", "VALID", 0, 1, 1, 0)
        assert "ENTER" in table.allowed(alarmed_entry, (0, 0))

    def test_begin_without_proof_admits_work_on_a_locked_unproven_asset(
        self, sabotaged_crushers: dict[str, tuple[Sabotaged, Winning, ShieldTable]]
    ) -> None:
        _m, _w, table = sabotaged_crushers["begin_without_proof"]
        assert "BEGIN" in table.allowed(("LOCKED", "VALID", 0, 3, 0, 0), (0, 0))

    def test_removing_the_failure_transition_lets_a_critical_crusher_run_on(
        self, sabotaged_crushers: dict[str, tuple[Sabotaged, Winning, ShieldTable]]
    ) -> None:
        _m, _w, table = sabotaged_crushers["running_critical_never_fails"]
        assert "NOOP" in table.allowed(("RUN", "NONE", 0, 3, 0, 0), (3, 3))

    def test_a_handover_that_does_not_void_the_permit_admits_a_late_entry(
        self, sabotaged_crushers: dict[str, tuple[Sabotaged, Winning, ShieldTable]]
    ) -> None:
        _m, _w, table = sabotaged_crushers["handover_keeps_the_permit"]
        assert "ENTER" in table.allowed(("WORK", "VALID", 0, 1, 0, 1), (0, 0))

    def test_a_retest_that_never_goes_stale_admits_unattended_occupancy(
        self, sabotaged_crushers: dict[str, tuple[Sabotaged, Winning, ShieldTable]]
    ) -> None:
        _m, _w, table = sabotaged_crushers["retest_never_goes_stale"]
        assert "NOOP" in table.allowed(("WORK", "VALID", 1, 2, 0, 0), (0, 0))


class TestSabotagedPumps:
    @pytest.mark.parametrize("label", PUMP_LABELS)
    def test_every_sabotage_is_seen(
        self,
        label: str,
        sabotaged_pumps: dict[str, tuple[Sabotaged, Winning, ShieldTable]],
        pumps: PumpPairModel,
        pumps_win: Winning,
        pump_tables: dict[int, ShieldTable],
    ) -> None:
        _model, win, table = sabotaged_pumps[label]
        assert win.relation_sha256 != pumps_win.relation_sha256
        assert table.sha256() != pump_tables[4].sha256()
        assert not matches_model(table, pumps)
        assert unsafe_allowed(table, pumps, pumps_win), label

    def test_without_the_availability_check_the_second_pump_may_be_left_to_fail(
        self, sabotaged_pumps: dict[str, tuple[Sabotaged, Winning, ShieldTable]]
    ) -> None:
        _m, _w, table = sabotaged_pumps["no_availability_check"]
        assert "NOOP|NOOP" in table.allowed(("WORK", "RUN"), (0, 0, 3, 3))


class TestTamperedTables:
    """Hand edits are what the hash and the rebuild check exist to catch."""

    def _tampered(self, table: ShieldTable, edit: Callable[[list], None]) -> ShieldTable:
        raw = json.loads(table.to_json())
        edit(raw["entries"])
        return ShieldTable.from_json(json.dumps(raw))

    def test_stop_removed_from_a_critical_running_entry(
        self,
        crusher: CrusherJobModel,
        crusher_win: Winning,
        crusher_tables: dict[int, ShieldTable],
    ) -> None:
        def edit(entries: list) -> None:
            row = next(r for r in entries if r[0][0] == "RUN" and r[1] == [3, 3] and "STOP" in r[2])
            row[2] = []  # an empty entry: a deadlock

        forged = self._tampered(crusher_tables[4], edit)
        assert critical_run_violations(forged, crusher, crusher_win)
        assert critical_run_violations(crusher_tables[4], crusher, crusher_win) == []
        assert not matches_model(forged, crusher)

    def test_an_unsafe_command_added_to_a_critical_running_entry(
        self,
        crusher: CrusherJobModel,
        crusher_win: Winning,
        crusher_tables: dict[int, ShieldTable],
    ) -> None:
        def edit(entries: list) -> None:
            row = next(r for r in entries if r[0][0] == "RUN" and r[1] == [3, 3] and r[0][2] == 0)
            row[2] = sorted({*row[2], "NOOP"})

        forged = self._tampered(crusher_tables[4], edit)
        assert critical_run_violations(forged, crusher, crusher_win)
        assert unsafe_allowed(forged, crusher, crusher_win)
        assert forged.sha256() != crusher_tables[4].sha256()
