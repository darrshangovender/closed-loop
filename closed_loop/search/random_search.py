"""Uniform random search over the unit box, and the ``SearchRun`` record both optimisers return.

Interface text: docs/adr/0002-interfaces.md, section 3.11; design: docs/adr/0001-architecture.md,
section 9. Random search is the baseline CMA-ES is compared with: same box, same encoding, same
budget.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for

__all__ = ["SearchRun", "random_search", "summarise_run"]


@dataclass(frozen=True, eq=False)
class SearchRun:
    """The outcome of one search run. Lower ``f`` is worse for the system under test.

    ``first_violation_eval`` is the 1-based number of the first evaluation with ``f < 0``, or None
    if there was none within ``evals`` (a right-censored value). ``history[i]`` is the best ``f``
    found in the first ``i + 1`` evaluations, so it never rises. ``theta_best`` is the argument
    of the best ``f``. ``regime`` is a label ("R_in", "R_out", ...) set by the caller.
    Equality is identity because the fields hold arrays.
    """

    algo: str
    regime: str
    seed: int
    best_f: float
    first_violation_eval: int | None
    evals: int
    theta_best: NDArray[np.float64]
    history: NDArray[np.float64]


def _check_dim_budget(dim: int, budget: int) -> None:
    for name, value in (("dim", dim), ("budget", budget)):
        if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < 1:
            raise ValueError(f"{name} must be an int of at least 1, got {value!r}")


def summarise_run(
    algo: str,
    regime: str,
    seed: int,
    values: NDArray[np.float64],
    points: NDArray[np.float64],
) -> SearchRun:
    """Build a ``SearchRun`` from every evaluation in order (``values[i]`` at ``points[i]``)."""
    best_idx = int(np.argmin(values))  # first of the equal minima
    negative = np.flatnonzero(values < 0.0)
    return SearchRun(
        algo=algo,
        regime=regime,
        seed=int(seed),
        best_f=float(values[best_idx]),
        first_violation_eval=int(negative[0]) + 1 if negative.size else None,
        evals=int(values.shape[0]),
        theta_best=np.array(points[best_idx], dtype=np.float64),
        history=np.minimum.accumulate(values.astype(np.float64)),
    )


def random_search(
    f: Callable[[NDArray[np.float64]], float],
    dim: int,
    budget: int,
    seed: int,
    *,
    regime: str = "",
) -> SearchRun:
    """Evaluate ``f`` at ``budget`` points drawn uniformly from ``[0, 1)^dim``; keep the minimum.

    The points come from ``rng_for(seed, "random_search")`` and do not depend on ``f``, so two
    objectives see the same points. Raises ``ValueError`` if ``f`` returns NaN.
    """
    _check_dim_budget(dim, budget)
    points = rng_for(seed, "random_search").random((int(budget), int(dim)))
    values = np.empty(int(budget), dtype=np.float64)
    for i in range(int(budget)):
        value = float(f(points[i].copy()))
        if math.isnan(value):
            raise ValueError(f"f returned NaN at evaluation {i + 1}")
        values[i] = value
    return summarise_run("random", regime, seed, values, points)
