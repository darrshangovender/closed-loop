"""Safety-game solver: the greatest fixed point of "some command keeps every outcome winning".

``solve`` is generic over ``AbstractModel``. A state is winning when some command exists whose
every environment outcome is not ``VIOL`` and is itself winning. The set of winning states ``W`` is
the largest set closed under that condition, reached by starting from all states and removing the
states that fail it until none fails (Knaster-Tarski; the operator is monotone and the state space
is finite, so the iteration ends).

What the result proves, and what it does not: ``W`` is exact for the finite abstraction only, under
the assumptions the model states (A_exec, A_phys, A_iso, A_dec; see ``abstraction``). It says
nothing about the plant until the abstraction map and the perception contract C1 are also granted,
and C1 is read per life (ADR 0003, A6).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from closed_loop.shield.abstraction import VIOL, AbstractModel


def canonical(x: Any) -> Any:
    """JSON-ready, order-independent form of a state, command or outcome set.

    Tuples become lists, sets become lists sorted by their own JSON text, so the result does not
    depend on the hash seed of the process.
    """
    if isinstance(x, (tuple, list)):
        return [canonical(v) for v in x]
    if isinstance(x, (frozenset, set)):
        return sorted((canonical(v) for v in x), key=_dump)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    raise TypeError(f"cannot canonicalise {type(x).__name__}")


def _dump(x: Any) -> str:
    return json.dumps(x, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


@dataclass(frozen=True)
class Winning:
    """Result of ``solve``.

    ``safe[s]`` is every command whose outcomes all stay in ``states`` without reaching VIOL (the
    maximally permissive set, not one chosen command). ``iterations`` counts applications of the
    operator, including the last one that changed nothing, so a model in which every state wins has
    ``iterations == 1``. ``relation_sha256`` digests the whole transition relation that was solved,
    so a table can say which abstraction it came from.
    """

    states: frozenset[Hashable]
    safe: Mapping[Hashable, frozenset[str]]
    iterations: int
    relation_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "safe", MappingProxyType(dict(self.safe)))


def solve(model: AbstractModel) -> Winning:
    commands = tuple(model.commands())
    # (state) -> [(command, non-VIOL successors, reaches VIOL)], computed once.
    relation: dict[Hashable, list[tuple[str, frozenset[Hashable], bool]]] = {}
    digest = hashlib.sha256()
    digest.update(_dump({"model": model.name, "commands": list(commands)}).encode("ascii"))
    for s in model.states():
        rows = []
        for c in commands:
            outcomes = model.step(s, c)
            rows.append((c, frozenset(o for o in outcomes if o != VIOL), VIOL in outcomes))
            # repr of a tuple of str and int is the same in every process; a set's order is not.
            digest.update(repr((s, c, sorted(outcomes, key=repr))).encode("utf-8"))
        relation[s] = rows

    region: set[Hashable] = set(relation)
    iterations = 0
    while True:
        iterations += 1
        kept = {
            s
            for s in region
            if any(not reaches_viol and nxt <= region for _c, nxt, reaches_viol in relation[s])
        }
        if len(kept) == len(region):
            break
        region = kept

    safe = {
        s: frozenset(
            c for c, nxt, reaches_viol in relation[s] if not reaches_viol and nxt <= region
        )
        for s in region
    }
    return Winning(frozenset(region), safe, iterations, digest.hexdigest())


def winning_operator(model: AbstractModel, region: frozenset[Hashable]) -> frozenset[Hashable]:
    """One application of the operator, straight from ``model.step`` (no cache).

    Exposed so that a test can confirm ``solve`` returned a fixed point without trusting
    ``solve``'s own bookkeeping.
    """
    commands = tuple(model.commands())
    return frozenset(
        s for s in model.states() if any(_all_inside(model.step(s, c), region) for c in commands)
    )


def _all_inside(outcomes: frozenset[Hashable | str], region: frozenset[Hashable]) -> bool:
    return all(o != VIOL and o in region for o in outcomes)
