"""The six properties P1-P6, their parameters and the signals they read.

Interface text: docs/adr/0002-interfaces.md, section 3.4; properties: docs/adr/0001-architecture.md,
section 5. Every parameter that comes from a clause lives in ``SpecParams`` (see ``clauses.py``).

Roles, not asset ids
--------------------
The formulas name four fixed roles, ``crusher``, ``pump_a``, ``pump_b`` and ``conveyor``, and
``signals_from_trace`` maps the asset ids it is given onto them. Every atom reads one component,
except ``pumps.available`` (the pump pair), which is the A_dec assumption of section 4.2. P1 reads
all four (ADR 0003 A12.3: it used to leave the conveyor out, and the plant recorded P1 events on
CV-101 that the specification could not see).

Margins (the unit is in the atom name's comment)
------------------------------------------------
* ``*.working``, ``*.iso_proven``, ``crusher.occupied``, ``crusher.alarm_occupied``: +1 when
  true, -1 when false.
* ``crusher.in_space``: ``+HOLDS_MARGIN`` when the space is occupied, ``-HOLDS_MARGIN`` when not
  (HOLDS_MARGIN = 1e9). It is the antecedent of P2 and P3. An implication is ``max(-antecedent,
  consequent)``, so an antecedent of size 1 would clip a violation to -1 and hide the minutes in
  the consequent; at 1e9 the consequent's own margin comes through. An idle property therefore
  reads 1e9, "nothing contradicts it".
* ``crusher.gas_fresh``: ``retest_q * 15 - gas_age_min`` (minutes). Positive while the last passing
  test is inside the re-test period; zero at exactly the period, which is still compliant.
* ``crusher.gas_since_break``: ``empty_min - gas_age_min`` (minutes): how much of the time since
  the space was last empty the last test covers. Negative when the test is older than the break.
  ``empty_min`` is 15 minutes per consecutive empty quarter before this one, and 1e7 when the space
  has never been occupied, so a first entry is judged by ``gas_fresh`` alone.
* ``crusher.permit_ok`` (minutes): the smaller of the plant's own ``permit_margin_min`` and
  ``permit_max_h * 60`` minus the permit's age; -1 while there is no permit or a revalidation is
  pending. The age counts from the first quarter of the unbroken run of non-NONE permit values.
  The F1 plant ends a permit at the same 12 hours from its issue (ADR 0003 A12.1), a quarter
  earlier than this cap would go negative, so on a plant trace the plant's own margin and a NONE
  permit are what read negative first; the cap is for traces that come from elsewhere.
* ``pumps.available``: pumps in ``RUN`` or ``STOP`` minus 0.5 (a stopped pump counts as available,
  0001 section 4.6; a locked pump does not: it cannot be started until the lock is removed).
* ``crusher.no_failure``: -1 on an uncontrolled-failure quarter, otherwise the true remaining life
  in hours (1 when the trace did not record it).

Convention assumed of the plant (to confirm when the plant lands): ``gas_age_min[t]`` is 0 in the
quarter a test passes and grows by 15 per quarter; ``alarm`` and ``occupied`` are the state during
quarter ``t``.

P6 is written as an equivalent past formula. ``G(alarm & occupied -> F[0,k] not occupied)`` fails
at quarter ``t`` exactly when the space is occupied at every quarter ``t .. t+k``, which a monitor
standing at ``t+k`` sees as "alarm and occupied ``k`` quarters ago, and occupied since". So the same
formula serves the streaming monitor and the offline objective, at the cost of the verdict arriving
``k`` quarters after the alarm, and of a window running past the end of the trace going unjudged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.timegrid import QUARTER_MIN
from closed_loop.core.trace import Trace, op_index, permit_index
from closed_loop.core.types import Op, Permit
from closed_loop.spec.stl import (
    Always,
    And,
    Atom,
    Formula,
    Hist,
    Implies,
    Not,
    Once,
    all_of,
    min_robustness,
)

__all__ = [
    "DEFAULT_CONVEYOR",
    "HOLDS_MARGIN",
    "PROPERTY_IDS",
    "SpecParams",
    "evaluate",
    "property_formulas",
    "signals_from_trace",
]

PROPERTY_IDS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5", "P6")
DEFAULT_CONVEYOR = "CV-101"  # the asset id P1's fourth conjunct reads unless the caller names one
ROLES: tuple[str, ...] = ("crusher", "pump_a", "pump_b", "conveyor")  # the assets P1 covers
HOLDS_MARGIN = 1.0e9  # size of a Boolean antecedent, above any margin the trace can produce
_NEVER_EMPTY_MIN = 1.0e7  # empty_min when the space has not been occupied yet


@dataclass(frozen=True)
class SpecParams:
    """Clause-backed parameters of the specification (quarters unless stated)."""

    retest_q: int = 2  # gas re-test period, CORP-HSE-002 clause 4.3
    break_q: int = 1  # a break in work longer than this needs a fresh test, same clause
    permit_max_h: float = 12.0  # permit validity, CORP-HSE-004 clause 4.1
    alarm_exit_q: int = 1  # quarters allowed between a gas alarm and exit, CORP-HSE-002 clause 4.4
    provenance: Mapping[str, str] = field(default_factory=dict)  # id -> verbatim clause + source

    def __post_init__(self) -> None:
        for name in ("retest_q", "break_q", "alarm_exit_q"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
                raise ValueError(f"{name} must be an int of at least 1 quarter, got {value!r}")
        if not self.permit_max_h > 0.0 or not np.isfinite(self.permit_max_h):
            raise ValueError(f"permit_max_h must be positive and finite, got {self.permit_max_h!r}")
        object.__setattr__(self, "provenance", dict(self.provenance))


def property_formulas(p: SpecParams) -> dict[str, Formula]:
    """The six formulas, keyed ``"P1"`` to ``"P6"``."""

    def atom(name: str) -> Formula:
        return Atom(name)

    occupied = atom("crusher.occupied")
    in_space = atom("crusher.in_space")
    working_proven = [
        Implies(atom(f"{role}.working"), atom(f"{role}.iso_proven")) for role in ROLES
    ]
    k = p.alarm_exit_q
    return {
        # P1: no work on an energised asset (CORP-HSE-001 clause 5.1), on the crusher, both pumps
        # and the conveyor
        "P1": Always(all_of(*working_proven)),
        # P2: occupied only under a valid, revalidated permit (CORP-HSE-004 clauses 4.1, 5.1)
        "P2": Always(Implies(in_space, atom("crusher.permit_ok"))),
        # P3: re-test inside the period, and after a longer break before re-entry
        "P3": Always(
            And(
                Implies(in_space, atom("crusher.gas_fresh")),
                Implies(
                    And(in_space, Hist(Not(in_space), 1, p.break_q + 1)),
                    atom("crusher.gas_since_break"),
                ),
            )
        ),
        # P4: at least one slurry pump available (plant design assumption)
        "P4": Always(atom("pumps.available")),
        # P5: no uncontrolled failure of the critical asset (plant design assumption)
        "P5": Always(atom("crusher.no_failure")),
        # P6: a gas alarm with the space occupied is followed by exit within k quarters
        "P6": Always(
            Implies(
                Once(atom("crusher.alarm_occupied"), k, k),
                Not(Hist(occupied, 0, k)),
            )
        ),
    }


def _sign(mask: NDArray[np.bool_]) -> NDArray[np.float32]:
    return np.where(mask, np.float32(1.0), np.float32(-1.0)).astype(np.float32)


def _run_start_age(active: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Quarters since the current unbroken run of ``active`` began (0 on its first quarter)."""
    n = active.shape[0]
    idx = np.arange(n)
    started = active & ~np.concatenate([[False], active[:-1]])
    last_start = np.maximum.accumulate(np.where(started, idx, -1))
    return np.asarray(np.where(active, idx - last_start, 0), dtype=np.int64)


def _empty_minutes(occupied: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Minutes the space had been empty just before each quarter (15 per empty quarter)."""
    n = occupied.shape[0]
    idx = np.arange(n)
    last_occ = np.maximum.accumulate(np.where(occupied, idx, -1))
    before = np.concatenate([[-1], last_occ[:-1]])  # last occupied quarter strictly before t
    empty_q = idx - 1 - before
    out = np.where(before < 0, _NEVER_EMPTY_MIN, empty_q * float(QUARTER_MIN))
    return out.astype(np.float32)


def signals_from_trace(
    tr: Trace,
    p: SpecParams,
    crusher: str,
    pumps: tuple[str, str],
    conveyor: str = DEFAULT_CONVEYOR,
) -> dict[str, NDArray[np.float32]]:
    """Margin signals for every atom of ``property_formulas(p)``, one value per quarter."""
    if len(pumps) != 2 or pumps[0] == pumps[1]:
        raise ValueError(f"pumps must be two different asset ids, got {pumps!r}")
    assets = {"crusher": crusher, "pump_a": pumps[0], "pump_b": pumps[1], "conveyor": conveyor}
    for role, asset in assets.items():
        if asset not in tr.asset_ids:
            raise ValueError(f"{role} {asset!r} is not in the trace's assets {tr.asset_ids}")
    if crusher in pumps:
        raise ValueError("the crusher cannot also be a pump")
    if conveyor in (crusher, *pumps):
        raise ValueError("the conveyor cannot also be the crusher or a pump")

    work, run, stop = op_index(Op.WORK), op_index(Op.RUN), op_index(Op.STOP)
    out: dict[str, NDArray[np.float32]] = {}
    for role, asset in assets.items():
        out[f"{role}.working"] = _sign(tr.op[asset] == work)
        out[f"{role}.iso_proven"] = _sign(tr.iso_proven[asset])

    occupied = tr.occupied[crusher]
    out["crusher.occupied"] = _sign(occupied)
    out["crusher.in_space"] = np.where(
        occupied, np.float32(HOLDS_MARGIN), np.float32(-HOLDS_MARGIN)
    ).astype(np.float32)
    out["crusher.alarm_occupied"] = _sign(tr.alarm[crusher] & occupied)

    age = tr.gas_age_min[crusher].astype(np.float32)
    out["crusher.gas_fresh"] = (np.float32(p.retest_q * QUARTER_MIN) - age).astype(np.float32)
    out["crusher.gas_since_break"] = (_empty_minutes(occupied) - age).astype(np.float32)

    permit = tr.permit[crusher]
    has_permit = permit != permit_index(Permit.NONE)
    cap = np.float32(p.permit_max_h * 60.0) - _run_start_age(has_permit) * np.float32(QUARTER_MIN)
    plant_margin = tr.permit_margin_min[crusher]
    valid = permit == permit_index(Permit.VALID)
    out["crusher.permit_ok"] = np.where(
        valid, np.minimum(plant_margin, cap), np.float32(-1.0)
    ).astype(np.float32)

    available = sum((tr.op[a] == run) | (tr.op[a] == stop) for a in pumps)
    out["pumps.available"] = (np.asarray(available, dtype=np.float32) - np.float32(0.5)).astype(
        np.float32
    )

    rul_hours = tr.true_rul_days[crusher] * np.float32(24.0)
    rul_hours = np.where(np.isnan(rul_hours), np.float32(1.0), np.maximum(rul_hours, 1e-3))
    out["crusher.no_failure"] = np.where(
        tr.uncontrolled_failure[crusher], np.float32(-1.0), rul_hours
    ).astype(np.float32)
    return out


def evaluate(
    tr: Trace,
    p: SpecParams,
    crusher: str,
    pumps: tuple[str, str],
    conveyor: str = DEFAULT_CONVEYOR,
) -> dict[str, float]:
    """Min robustness per property over the whole trace; negative = violation."""
    signals = signals_from_trace(tr, p, crusher, pumps, conveyor)
    return {pid: min_robustness(f, signals) for pid, f in property_formulas(p).items()}
