"""Brute-force checks of a shield table against a model, used by several test files.

These re-derive, from the winning region alone, what a table entry must hold. They are tests of
P5a's own solver and table (the packet text asks for them); they are not the independent
reachability checker of P5b.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from itertools import product

from closed_loop.shield.abstraction import VIOL, AbstractModel
from closed_loop.shield.game import Winning
from closed_loop.shield.table import ShieldTable


class TinyGame:
    """A hand-built game: ``edges[(state, command)]`` lists the outcomes; absent means VIOL."""

    def __init__(
        self, edges: dict[tuple[str, str], Sequence[str]], initial: str = "a", name: str = "tiny"
    ) -> None:
        self.name = name
        self._edges = edges
        self._initial = initial
        self._states = sorted(
            {s for s, _ in edges} | {t for v in edges.values() for t in v} - {VIOL}
        )
        self._commands = sorted({c for _, c in edges})

    def states(self) -> Iterable[Hashable]:
        return list(self._states)

    def commands(self) -> Sequence[str]:
        return self._commands

    def step(self, s: Hashable, c: str) -> frozenset[Hashable | str]:
        return frozenset(self._edges.get((s, c), (VIOL,)))  # type: ignore[arg-type]

    def initial(self) -> Hashable:
        return self._initial

    def observable(self, s: Hashable) -> Hashable:
        return s

    def health(self, s: Hashable) -> int:
        return 0


def hvec(h: int | tuple[int, ...]) -> tuple[int, ...]:
    return (h,) if isinstance(h, int) else tuple(h)


_INDEX: dict[int, dict[Hashable, list[Hashable]]] = {}


def _by_observable(model: AbstractModel, win: Winning) -> dict[Hashable, list[Hashable]]:
    """Winning states grouped by observable; kept per solved region for the session."""
    index = _INDEX.get(id(win))
    if index is None:
        index = {}
        for s in win.states:
            index.setdefault(model.observable(s), []).append(s)
        _INDEX[id(win)] = index
    return index


def box_members(
    model: AbstractModel, win: Winning, o: Hashable, health_set: tuple[int, ...]
) -> list[Hashable]:
    """Winning states at observable ``o`` whose hidden bins lie in the set."""
    ranges = [range(lo, hi + 1) for lo, hi in zip(health_set[0::2], health_set[1::2], strict=True)]
    wanted = set(product(*ranges))
    return [s for s in _by_observable(model, win).get(o, ()) if hvec(model.health(s)) in wanted]


def expected_entry(
    model: AbstractModel, win: Winning, o: Hashable, health_set: tuple[int, ...]
) -> frozenset[str] | None:
    """The formula of the table docstring, applied directly. None when the box has no member."""
    members = box_members(model, win, o, health_set)
    if not members:
        return None
    return frozenset.intersection(*(win.safe[s] for s in members))


def critical_run_violations(
    table: ShieldTable, model: AbstractModel, win: Winning
) -> list[tuple[Hashable, tuple[int, ...], frozenset[str]]]:
    """Entries for a RUN crusher whose set holds a winning bin-3 state but do not allow exactly
    STOP, plus any empty entry (a deadlock). Empty list on a sound crusher table."""
    critical = {model.observable(s) for s in win.states if model.health(s) == 3}
    bad = []
    for (o, hs), cmds in table.entries.items():
        if not cmds:
            bad.append((o, hs, cmds))
        elif o[0] == "RUN" and o in critical and hs[-1] == 3 and cmds != {"STOP"}:  # type: ignore[index]
            bad.append((o, hs, cmds))
    return bad


def unsafe_allowed(
    table: ShieldTable, model: AbstractModel, win: Winning
) -> list[tuple[Hashable, tuple[int, ...], str, Hashable]]:
    """Every (observable, set, command, state) where the table allows a command that, for some
    winning state in the set, can reach VIOL or leave the winning region under ``model``."""
    by_obs: dict[Hashable, list[Hashable]] = {}
    for s in win.states:
        by_obs.setdefault(model.observable(s), []).append(s)
    bad = []
    for (o, hs), cmds in table.entries.items():
        for s in by_obs.get(o, ()):
            h = hvec(model.health(s))
            if not all(lo <= x <= hi for x, lo, hi in zip(h, hs[0::2], hs[1::2], strict=True)):
                continue
            for c in sorted(cmds):
                if any(t == VIOL or t not in win.states for t in model.step(s, c)):
                    bad.append((o, hs, c, s))
    return bad
