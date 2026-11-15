"""Scenario shrinking: take a violating theta to a smaller one that still violates (packet P9b).

Interface text: docs/adr/0002-interfaces.md, section 3.11 (``shrink(theta, still_violates, tol)``,
"coordinate bisection to nominal"); use: docs/adr/0001-architecture.md section 9, where each find is
re-run with adversary dimensions reset to nominal to see which ones it needs.

What "nominal" means for a coordinate
-------------------------------------
``plant.scenario.Scenario.decode`` has a value of every coordinate under which the plant behaves as
it does with no adversary: a wear-rate multiplier of 1, onset and residual-life shifts of 0, no
sensor fault, holders as present as usual, the usual approver delay, catch probability and
spare lead time, no load shift. Two coordinates have no "off" value in the layout: the two
adversarial gas excursions and the inserted handover note. Their nominal value is the last quarter
of the episode (u = 1), which the plant never reaches while anyone is inside and where no handover
is left to carry a note. ``nominal_theta`` writes these values down for each regime (the ranges
differ, so the u that means "1.0" differs), and ``tests/search/test_shrink.py`` decodes them to
prove that each means what is claimed here.

The algorithm
-------------
1. **Reset.** Delta debugging over the non-nominal coordinates: try to put all of them back at
   nominal, then halves, quarters ... down to single coordinates, keeping a reset whenever the
   scenario still violates; repeat single resets until none is accepted. What is left is
   1-minimal: no one remaining coordinate can be reset on its own.
2. **Bisect.** For each coordinate left, bisect the segment between its nominal value and its
   current value for the smallest deviation (to ``tol``) that still violates. Nothing assumes the
   violation is monotone along the segment: the invariant is only "the vector we hold violates".

Both phases move a coordinate only towards its nominal value and never away, so the number of
non-nominal coordinates (deviation larger than ``tol``) cannot rise. A coordinate that was within
``tol`` of nominal at the start is not touched.

The predicate is the caller's: ``still_violates(theta)`` is a full episode in practice, about a
second on the shielded target, so ``shrink`` takes a call budget and stops with what it has (which
violates) when it is spent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from closed_loop.plant.scenario import REGIMES, THETA_DIM

__all__ = [
    "COORDINATES",
    "ShrinkResult",
    "non_nominal",
    "nominal_theta",
    "shrink",
    "shrink_detailed",
]

# A coordinate can have no effect while another is at nominal: the sensor-fault day means nothing
# when the fault type is "none", and the load-shift day nothing when the shift is 0. The reset
# phase puts such a coordinate back for free (the scenario cannot notice), which is why a shrunk
# scenario lists fewer coordinates than the vector it came from.

_ASSET_NAMES: Final[tuple[str, ...]] = ("CR-501", "CV-101", "PU-201", "PU-202")
_PER_ASSET: Final[tuple[str, ...]] = (
    "wear-rate multiplier",
    "onset shift of the first life",
    "residual-life shift of the first life",
    "sensor-fault type",
    "sensor-fault day",
)


def _coordinate_names() -> tuple[str, ...]:
    names = [f"{asset} {what}" for asset in _ASSET_NAMES for what in _PER_ASSET]
    names += [f"holder-absence bias at the handover into shift {s}" for s in range(3)]
    names += ["first gas excursion time", "second gas excursion time"]
    names += [
        "handover-note template",
        "handover-note insertion time",
        "approver delay multiplier",
        "approver catch probability",
        "spare lead-time multiplier",
        "load-shift magnitude",
        "load-shift day",
    ]
    return tuple(names)


COORDINATES: Final[tuple[str, ...]] = _coordinate_names()


def nominal_theta(regime: str = "in") -> NDArray[np.float64]:
    """The vector under which ``Scenario.decode`` describes the plant with no adversary.

    The affine maps are those of ``plant/scenario.py`` (its module docstring lists them); a test
    decodes this vector and checks each value, so a change to the ranges there fails loudly here.
    """
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}, got {regime!r}")
    out_regime = regime == "out"
    rate = (1.0 - 0.5) / (2.0 - 0.5) if out_regime else (1.0 - 0.7) / (1.4 - 0.7)
    delay = (1.0 - 0.25) / (6.0 - 0.25) if out_regime else (1.0 - 0.5) / (3.0 - 0.5)
    lead = (1.0 - 0.25) / (3.0 - 0.25) if out_regime else (1.0 - 0.5) / (2.0 - 0.5)
    theta = np.zeros(THETA_DIM, dtype=np.float64)
    for a in range(len(_ASSET_NAMES)):
        theta[5 * a + 0] = rate
        theta[5 * a + 1] = 0.5  # onset shift 0 days
        theta[5 * a + 2] = 0.5  # residual-life shift 0 days
        theta[5 * a + 3] = 0.0  # no sensor fault
        theta[5 * a + 4] = 0.0  # its day: unused while there is no fault
    theta[20:23] = 0.0  # holder-absence bias 0: presence as usual
    theta[23:25] = 1.0  # gas excursions at the last quarter
    theta[25] = 0.0  # note template: unused while the note is never inserted
    theta[26] = 1.0  # note insertion at the last quarter: no handover is left to carry it
    theta[27] = delay
    theta[28] = 0.5  # catch probability 0.5, PlantParams.approver_catch_p
    theta[29] = lead
    theta[30] = 0.0  # no load shift
    theta[31] = 0.0  # its day: unused while there is no shift
    theta.setflags(write=False)
    return theta


def non_nominal(
    theta: NDArray[np.float64],
    nominal: NDArray[np.float64] | None = None,
    tol: float = 0.02,
    *,
    regime: str = "in",
) -> tuple[int, ...]:
    """Indices of the coordinates that differ from nominal by more than ``tol``."""
    nom = nominal_theta(regime) if nominal is None else np.asarray(nominal, dtype=np.float64)
    x = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0)
    return tuple(int(i) for i in np.flatnonzero(np.abs(x - nom) > tol))


@dataclass(frozen=True, eq=False)
class ShrinkResult:
    """What ``shrink_detailed`` returns. ``truncated`` means the call budget ended the search."""

    theta: NDArray[np.float64]
    calls: int
    non_nominal_before: int
    non_nominal_after: int
    truncated: bool


class _Budget(Exception):
    """Raised inside the search when the call budget is spent."""


def shrink_detailed(
    theta: NDArray[np.float64],
    still_violates: Callable[[NDArray[np.float64]], bool],
    tol: float = 0.02,
    *,
    regime: str = "in",
    nominal: NDArray[np.float64] | None = None,
    max_calls: int | None = None,
) -> ShrinkResult:
    """Shrink ``theta`` as the module docstring says; see ``shrink`` for the arguments.

    Raises ``ValueError`` if ``theta`` itself does not violate (there is nothing to preserve) or if
    ``tol`` is not in (0, 1). The call that checks ``theta`` counts against ``max_calls``.
    """
    if not 0.0 < tol < 1.0:
        raise ValueError(f"tol must be in (0, 1), got {tol!r}")
    if max_calls is not None and max_calls < 1:
        raise ValueError(f"max_calls must be at least 1, got {max_calls!r}")
    nom = nominal_theta(regime) if nominal is None else np.asarray(nominal, dtype=np.float64)
    x = np.clip(np.asarray(theta, dtype=np.float64), 0.0, 1.0).copy()
    if x.shape != nom.shape:
        raise ValueError(f"theta has shape {x.shape}, the nominal vector {nom.shape}")
    calls = 0

    def check(candidate: NDArray[np.float64]) -> bool:
        nonlocal calls
        if max_calls is not None and calls >= max_calls:
            raise _Budget
        calls += 1
        return bool(still_violates(candidate.copy()))

    before = len(non_nominal(x, nom, tol))
    if not check(x):
        raise ValueError("theta does not violate, so there is nothing to shrink")
    truncated = False
    try:
        active = list(non_nominal(x, nom, tol))
        _reset(x, active, nom, check)
        for i in sorted(active, key=lambda j: (-abs(x[j] - nom[j]), j)):
            x[i] = _bisect(x, i, float(nom[i]), tol, check)
    except _Budget:
        truncated = True
        # ``x`` only ever takes a value that was checked and violated, so it still violates
    return ShrinkResult(
        theta=x,
        calls=calls,
        non_nominal_before=before,
        non_nominal_after=len(non_nominal(x, nom, tol)),
        truncated=truncated,
    )


def _reset(
    x: NDArray[np.float64],
    active: list[int],
    nom: NDArray[np.float64],
    check: Callable[[NDArray[np.float64]], bool],
) -> None:
    """Delta debugging: reset chunks of ``active`` to nominal while the scenario still violates.

    ``x`` is edited in place, and only after the trial with the reset has been checked, so an
    interrupted search leaves the last vector that violated. ``active`` is edited in place too and
    ends as the coordinates that could not be reset.
    """
    chunk = max(len(active), 1)
    while active:
        progressed = False
        i = 0
        while i < len(active):
            part = active[i : i + chunk]
            trial = x.copy()
            trial[part] = nom[part]
            if check(trial):
                x[part] = nom[part]
                del active[i : i + chunk]
                progressed = True
            else:
                i += chunk
        if chunk == 1:
            if not progressed:
                break
        else:
            chunk = max(chunk // 2, 1)


def _bisect(
    x: NDArray[np.float64],
    i: int,
    nominal_i: float,
    tol: float,
    check: Callable[[NDArray[np.float64]], bool],
) -> float:
    """The value of ``x[i]`` nearest to nominal, to ``tol``, at which the scenario violates.

    ``x`` is edited in place while probing and left with the answer in ``x[i]``; if the budget
    ends the search half way, ``x[i]`` holds the last value that was checked and violated.
    """
    hi = float(x[i])
    lo = nominal_i
    try:
        while abs(hi - lo) > tol:
            mid = 0.5 * (hi + lo)
            x[i] = mid
            if check(x):
                hi = mid
            else:
                lo = mid
    finally:
        x[i] = hi
    return hi


def shrink(
    theta: NDArray[np.float64],
    still_violates: Callable[[NDArray[np.float64]], bool],
    tol: float = 0.02,
    *,
    regime: str = "in",
    nominal: NDArray[np.float64] | None = None,
    max_calls: int | None = None,
) -> NDArray[np.float64]:
    """A vector that still violates, with no more non-nominal coordinates than ``theta`` has.

    ``still_violates`` is called with a private copy of the candidate. A coordinate counts as
    non-nominal when it differs from ``nominal`` by more than ``tol``; ``nominal`` defaults to
    ``nominal_theta(regime)``. Every coordinate of the result lies between its nominal value and its
    value in ``theta``. ``max_calls`` bounds the calls to ``still_violates`` (the result then still
    violates, but may be less shrunk).
    """
    return shrink_detailed(
        theta, still_violates, tol, regime=regime, nominal=nominal, max_calls=max_calls
    ).theta
