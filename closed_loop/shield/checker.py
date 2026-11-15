"""Independent checker (packet P5b): reachability of the shielded product, and a differential test.

``check`` walks the product of a game state, the hidden health bins and the perception set the
shield is shown. From every set that contains the starting bins, it takes **every** command the
table admits, **every** outcome of the reference relation, and **every** next set that contains the
next bins and respects the width limit. It reports a ``VIOLATION`` (an admitted command that has
the sink among its outcomes) or a ``DEADLOCK`` (a reachable entry that admits nothing).

Nothing here reads ``abstraction.py``, ``game.py`` or ``table.py``, and nothing here imports them,
not even for types: the relation is the reference of ``shield.reference``, and the table is used
only through ``allowed`` (and ``entries``, for ``empty_entries``). Any game with an ``int`` or a
tuple health vector can be walked, which is how the pump pair (two bins) and the conveyor (one bin)
go through the same code as the crusher.

``differential`` compares a model's ``step`` with the reference on every (state, command) pair that
either side defines.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from itertools import product
from typing import Protocol

from closed_loop.shield.reference import BINS, VIOL, ReferenceGame, ReferenceModel

Product = tuple[Hashable, tuple[int, ...]]  # (game state, flattened perception set)


class ShieldLookup(Protocol):
    """The one thing the walk asks of a shield table: what it admits here."""

    def allowed(self, o: Hashable, health_set: tuple[int, ...], /) -> Iterable[str]: ...


class ShieldEntries(Protocol):
    """A table that can be listed, for ``empty_entries``."""

    @property
    def entries(self) -> Mapping[Product, Collection[str]]: ...


@dataclass(frozen=True)
class CheckResult:
    reachable: int
    findings: tuple[tuple[str, Hashable, object], ...]  # ("VIOLATION"|"DEADLOCK", state, detail)
    visited: frozenset[Product] = field(default_factory=frozenset, compare=False, repr=False)

    @property
    def clean(self) -> bool:
        return not self.findings


def health_vector(h: int | tuple[int, ...]) -> tuple[int, ...]:
    return (h,) if isinstance(h, int) else tuple(h)


def _boxes_containing(hv: tuple[int, ...], max_width: int) -> Iterator[tuple[int, ...]]:
    """Every flattened perception set ``(lo1, hi1, ...)`` that holds ``hv`` and is narrow enough."""
    per_component = [
        [(lo, hi) for lo in BINS for hi in BINS if lo <= h <= hi and hi - lo + 1 <= max_width]
        for h in hv
    ]
    for combo in product(*per_component):
        yield tuple(v for lo_hi in combo for v in lo_hi)


def check(
    ref: ReferenceGame,
    table: ShieldLookup,
    initial: Hashable,
    max_width: int,
    drop: tuple[Hashable, str] | None = None,
) -> CheckResult:
    """Explicit reachability of the shielded product against the reference relation.

    ``drop = (observable, command)`` removes that command from every table entry at that
    observable before the walk: the hook for "what if the shield had not offered this".
    """
    if not 1 <= max_width <= len(BINS):
        raise ValueError(f"max_width must be in 1..{len(BINS)}, got {max_width}")

    allowed_cache: dict[tuple[Hashable, tuple[int, ...]], tuple[str, ...]] = {}
    order = {c: i for i, c in enumerate(ref.commands())}

    def admitted(o: Hashable, box: tuple[int, ...]) -> tuple[str, ...]:
        key = (o, box)
        hit = allowed_cache.get(key)
        if hit is None:
            cmds = set(table.allowed(o, box))
            if drop is not None and o == drop[0]:
                cmds.discard(drop[1])
            hit = tuple(sorted(cmds, key=lambda c: (order.get(c, len(order)), c)))
            allowed_cache[key] = hit
        return hit

    seen: set[Product] = set()
    queue: deque[Product] = deque()
    findings: list[tuple[str, Hashable, object]] = []
    entered: set[Hashable] = set()  # states whose every perception set is already queued

    def push(s: Hashable) -> None:
        if s in entered:
            return
        entered.add(s)
        for box in _boxes_containing(health_vector(ref.health(s)), max_width):
            seen.add((s, box))
            queue.append((s, box))

    push(initial)
    while queue:
        s, box = queue.popleft()
        commands = admitted(ref.observable(s), box)
        if not commands:
            findings.append(("DEADLOCK", (s, box), ref.observable(s)))
            continue
        for c in commands:
            for t in ref.outcomes(s, c):
                if t == VIOL:
                    findings.append(("VIOLATION", (s, box), c))
                else:
                    push(t)
    return CheckResult(len(seen), tuple(findings), frozenset(seen))


def empty_entries(table: ShieldEntries) -> list[Product]:
    """Entries of the table that admit no command at all (reachable or not)."""
    return [key for key, cmds in table.entries.items() if not cmds]


Mismatch = tuple[Hashable, str, frozenset[Hashable], frozenset[Hashable]]


def _listed(thing: object, name: str) -> list[Hashable]:
    """What ``thing.<name>()`` lists, or nothing for an object that only has ``outcomes``."""
    getter = getattr(thing, name, None)
    return list(getter()) if callable(getter) else []


def differential(
    model: ReferenceGame, ref: ReferenceModel, limit: int | None = None
) -> list[Mismatch]:
    """``[(state, command, model outcomes, reference outcomes), ...]`` where they differ.

    Empty when the two relations are identical over every state and every command that either of
    them lists (the model's first, then whatever only the reference lists, so a command the model
    forgot is compared and not skipped). A pair that one side does not accept is reported with the
    reason in place of that side's outcomes. ``limit`` stops after that many mismatches (the
    sabotage tests only need one).
    """
    found: list[Mismatch] = []
    states = list(dict.fromkeys([*model.states(), *_listed(ref, "states")]))
    commands = list(dict.fromkeys([*model.commands(), *_listed(ref, "commands")]))
    for s in states:
        for c in commands:
            try:
                theirs = frozenset(model.step(s, str(c)))
            except Exception as exc:  # the model is the thing under test: report, do not crash
                theirs = frozenset({f"MODEL REJECTS: {type(exc).__name__}: {exc}"})
            try:
                mine = frozenset(ref.outcomes(s, str(c)))
            except ValueError as exc:
                mine = frozenset({f"REFERENCE REJECTS: {exc}"})
            if theirs != mine:
                found.append((s, str(c), theirs, mine))
                if limit is not None and len(found) >= limit:
                    return found
    return found


def structural_differences(model: ReferenceGame, ref: ReferenceGame) -> list[str]:
    """Differences in the shape of the two games (empty when identical).

    The state space, the command list in order (the substitution priority depends on it), the
    initial state, and the observable and health of every state.
    """
    problems: list[str] = []
    theirs = list(model.states())
    mine = list(ref.states())
    if theirs != mine:
        only_theirs = set(theirs) - set(mine)
        only_mine = set(mine) - set(theirs)
        if only_theirs or only_mine or len(theirs) != len(mine):
            problems.append(
                f"states differ: model {len(theirs)}, reference {len(mine)}; only in model "
                f"{sorted(only_theirs, key=repr)[:3]!r}, only in reference "
                f"{sorted(only_mine, key=repr)[:3]!r}"
            )
        else:
            problems.append(
                f"states differ: the same {len(mine)} states in a different order (model "
                f"starts {theirs[:2]!r}, reference starts {mine[:2]!r}); harmless to the proof"
            )
    theirs_cmds, mine_cmds = tuple(model.commands()), tuple(ref.commands())
    if theirs_cmds != mine_cmds:
        problems.append(f"commands differ: model {theirs_cmds!r}, reference {mine_cmds!r}")
    if model.initial() != ref.initial():
        problems.append(f"initial differs: model {model.initial()!r}, reference {ref.initial()!r}")
    for s in set(theirs) & set(mine):
        if model.observable(s) != ref.observable(s):
            problems.append(f"observable of {s!r} differs")
            break
    for s in set(theirs) & set(mine):
        if health_vector(model.health(s)) != health_vector(ref.health(s)):
            problems.append(f"health of {s!r} differs")
            break
    return problems
