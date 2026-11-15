"""Shared, cached objects and helpers for the P5b / P5d-b tests (independent checker).

Two kinds of table are used and the names say which:

* ``true_table`` is built from the reference alone: the reference is solved by ``independent_solve``
  and the table is made from the definition (``independent_entries``). Nothing in it comes from the
  production solver, so the checker and the sabotage matrix can be run without the production
  model, and keep meaning the same thing while it is being changed.
* ``shipped_table`` is what the production synthesis built from the production model. Only the
  tests whose class name contains ``Production`` or ``Shipped`` use it (or the production models).

The production modules are imported inside the functions that need them, so a production module
that is mid-change breaks only the tests that use it.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from functools import cache
from itertools import product
from typing import Any

from closed_loop.shield.reference import (
    VIOL,
    ConveyorReference,
    CrusherReference,
    PumpPairReference,
    ReferenceGame,
)

CRUSHER_FIELDS = ("op", "h", "permit", "occ", "gas", "alarm", "soon")
WIDTHS = (1, 2, 3, 4)
KINDS = ("crusher", "pumps", "conveyor")


def crusher_state(**fields: Any) -> tuple[Any, ...]:
    """A crusher state by name; the defaults are an idle, unpermitted, untested crusher."""
    base: dict[str, Any] = dict(op="RUN", h=0, permit="NONE", occ=0, gas=3, alarm=0, soon=0)
    unknown = set(fields) - set(base)
    if unknown:
        raise KeyError(f"unknown crusher fields {sorted(unknown)}")
    base.update(fields)
    return tuple(base[k] for k in CRUSHER_FIELDS)


def field_of(s: Hashable, name: str) -> Any:
    return s[CRUSHER_FIELDS.index(name)]  # type: ignore[index]


@cache
def crusher_reference(retest_q: int = 2) -> CrusherReference:
    return CrusherReference(retest_q)


@cache
def pump_reference() -> PumpPairReference:
    return PumpPairReference()


@cache
def conveyor_reference() -> ConveyorReference:
    return ConveyorReference()


def reference_for(kind: str) -> ReferenceGame:
    return {
        "crusher": crusher_reference,
        "pumps": pump_reference,
        "conveyor": conveyor_reference,
    }[kind]()


# --------------------------------------------------------------------------------------------
# the production side (imported late, used only by the Production and Shipped tests)
# --------------------------------------------------------------------------------------------


@cache
def crusher_model(retest_q: int = 2) -> Any:
    from closed_loop.shield.abstraction import CrusherJobModel

    return CrusherJobModel(retest_q)


@cache
def pump_model() -> Any:
    from closed_loop.shield.abstraction import PumpPairModel

    return PumpPairModel()


@cache
def conveyor_model() -> Any:
    from closed_loop.shield.abstraction import ConveyorModel

    return ConveyorModel()


def model_for(kind: str) -> Any:
    return {"crusher": crusher_model, "pumps": pump_model, "conveyor": conveyor_model}[kind]()


@cache
def production_winning(kind: str) -> Any:
    from closed_loop.shield.game import solve

    return solve(model_for(kind))


@cache
def shipped_table(kind: str, width: int) -> Any:
    """The table the production synthesis built from the production model."""
    from closed_loop.shield.table import build_shield

    return build_shield(model_for(kind), production_winning(kind), width)


# --------------------------------------------------------------------------------------------
# the reference side: solve, tables from the definition
# --------------------------------------------------------------------------------------------


def independent_solve(
    ref: ReferenceGame,
) -> tuple[frozenset[Hashable], Mapping[Hashable, frozenset[str]]]:
    """Winning region and safe sets from the reference alone (a greatest fixed point)."""
    states = list(ref.states())
    commands = list(ref.commands())
    region = set(states)
    while True:
        keep = {
            s
            for s in region
            if any(VIOL not in (out := ref.outcomes(s, c)) and out <= region for c in commands)
        }
        if len(keep) == len(region):
            break
        region = keep
    safe = {
        s: frozenset(
            c for c in commands if VIOL not in (out := ref.outcomes(s, c)) and out <= region
        )
        for s in region
    }
    return frozenset(region), safe


@cache
def solved(ref: ReferenceGame) -> tuple[frozenset[Hashable], Mapping[Hashable, frozenset[str]]]:
    """``independent_solve`` once per game object (the sabotaged games are solved once each)."""
    return independent_solve(ref)


def health_vector(h: int | tuple[int, ...]) -> tuple[int, ...]:
    return (h,) if isinstance(h, int) else tuple(h)


def independent_entries(
    ref: ReferenceGame, safe: Mapping[Hashable, frozenset[str]], width: int
) -> dict[tuple[Hashable, tuple[int, ...]], frozenset[str]]:
    """Table entries from the definition: intersect safe sets over the winning bins in the set."""
    by_obs: dict[Hashable, dict[tuple[int, ...], frozenset[str]]] = {}
    for s, cmds in safe.items():
        by_obs.setdefault(ref.observable(s), {})[health_vector(ref.health(s))] = cmds
    dims = len(health_vector(ref.health(ref.initial())))
    intervals = [(lo, hi) for lo in range(4) for hi in range(lo, min(lo + width, 4))]
    entries: dict[tuple[Hashable, tuple[int, ...]], frozenset[str]] = {}
    for o, slot in by_obs.items():
        for box in product(intervals, repeat=dims):
            members = [
                slot[h] for h in product(*(range(lo, hi + 1) for lo, hi in box)) if h in slot
            ]
            if members:
                entries[(o, tuple(v for lo_hi in box for v in lo_hi))] = frozenset.intersection(
                    *members
                )
    return entries


@dataclass(frozen=True)
class ReferenceTable:
    """A shield table made from a reference alone. ``allowed`` is an exact lookup: a set of any
    width up to the one the table was built for is answered, a set with no winning bin in it
    admits nothing."""

    model_name: str
    entries: Mapping[tuple[Hashable, tuple[int, ...]], frozenset[str]]

    def allowed(self, o: Hashable, health_set: tuple[int, ...]) -> frozenset[str]:
        return self.entries.get((o, tuple(health_set)), frozenset())


def table_from_model(model: ReferenceGame, width: int) -> ReferenceTable:
    """The table a correct synthesis would ship if it were given ``model`` as the truth."""
    _region, safe = solved(model)
    return ReferenceTable(model.name, independent_entries(model, safe, width))


@cache
def true_table(kind: str, width: int) -> ReferenceTable:
    return table_from_model(reference_for(kind), width)


def edit_entries(
    table: Any,
    edit: Callable[[Hashable, tuple[int, ...], frozenset[str]], frozenset[str]],
) -> Any:
    """A copy of ``table`` with every entry's command set rewritten by ``edit``."""
    entries = {key: edit(key[0], key[1], cmds) for key, cmds in table.entries.items()}
    if isinstance(table, ReferenceTable):
        return ReferenceTable(table.model_name, entries)
    from closed_loop.shield.table import ShieldTable

    return ShieldTable(table.model_name, entries, table.priority, table.meta)


class BeforeA12(CrusherReference):
    """The crusher as it stood at 0003 A11: ``ISSUE`` only in ``ISO``, a handover always sends a
    live permit to ``REVAL`` and clears the announcement. Kept so the history of the counts can be
    reproduced."""

    def _issue_applies(self, st: Any) -> bool:
        return bool(st.op == "ISO" and st.permit == "NONE")

    def _permit_after_handover(self, permit: str, ends: bool) -> str:
        return "NONE" if permit == "NONE" else "REVAL"

    def _announcement(self, before: Any, env: Any) -> int:
        return 0 if before.soon == 1 else int(env.soon_next)


class RenewsAlways(BeforeA12):
    """The reading before 0003 A11 (and before A12): completing the work always renews."""

    def _renewal_choices(self, st: Any, e: str, mid: Any) -> tuple[Any, ...]:
        if mid is not None and e == "COMPLETE" and st.op == "WORK":
            return (mid._replace(h=0),)
        return (mid,)


class SpikeReading(RenewsAlways):
    """The superseded reading of 0003 A9: an alarm that comes on with someone inside is itself a
    violation, and P2 is checked only for a pending revalidation (and, as in the spike, completion
    always renews). Built here from the amendment's description, not from the spike or from the
    production flag."""

    def _environment_and_hazards(self, st: Any, mid: Any, env: Any) -> Hashable | str:
        after = self._environment_stage(st, mid, env)
        if after is None:
            return VIOL
        if after.occ == 1 and (
            after.gas == self.stale or after.permit == "REVAL" or after.alarm == 1
        ):
            return VIOL
        return tuple(after)


class LiteralIssue(CrusherReference):
    """ISSUE as the game bullet of A12.1 reads word for word: permit ``NONE`` and ``op`` in
    ``ISO`` or ``WORK``, with no condition on who is inside."""

    def _issue_applies(self, st: Any) -> bool:
        return bool(st.permit == "NONE" and st.op in ("ISO", "WORK"))


@cache
def before_a12() -> BeforeA12:
    return BeforeA12()


@cache
def renews_always() -> RenewsAlways:
    return RenewsAlways()


@cache
def spike_reading() -> SpikeReading:
    return SpikeReading()


@cache
def literal_issue() -> LiteralIssue:
    return LiteralIssue()
