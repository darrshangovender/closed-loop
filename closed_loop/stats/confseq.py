"""Betting confidence sequences for a mean in [0, 1] (Waudby-Smith and Ramdas style).

Interface text: docs/adr/0002-interfaces.md, section 3.3; method: docs/adr/0001-architecture.md,
section 11.1.

For every candidate mean ``m`` on a grid, two capital processes bet against ``m``::

    K+_t(m) = prod_i (1 + lam_i (x_i - m))        K-_t(m) = prod_i (1 - lam_i (x_i - m))

with one predictable bet ``lam_i`` per step, shared by every ``m`` and capped at ``c = 0.5`` so
that every factor stays at or above 0.5. The hedged capital is ``K = (K+ + K-) / 2``. If ``m`` is
the true mean, ``K`` is a non-negative supermartingale started at 1, so by Ville's inequality
``P(K_t(m) >= 1/alpha for some t) <= alpha``. The confidence sequence is the set of ``m`` whose
capital has stayed below ``1/alpha``, valid at all times at once.

Discretisation is handled rigorously. ``K+`` falls and ``K-`` rises as ``m`` grows (the bets do
not depend on ``m``), so for ``m`` inside a grid cell ``[g_k, g_k+1]`` the capital is at least
``(K+(g_k+1) + K-(g_k)) / 2``. A cell is excluded only when that lower bound reaches ``1/alpha``.
The reported interval runs from the left edge of the first surviving cell to the right edge of
the last one, so it can be wider than the exact set by up to one grid step on each side.

All capital is kept in log space, so long streams cannot overflow.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

# Largest bet. With |x - m| <= 1 every factor 1 +/- lam (x - m) is then at least 0.5.
BET_CAP = 0.5
_BLOCK = 256  # rows of (time x grid) held in memory at once

# The spread-scaled sequence of ``paired_cs`` (0003 A16, prereg section 6).
SPREAD_MIN_PAIRS = 100  # fewer pairs than this go to the betting sequence
SPREAD_KURTOSIS_DIVISOR = 50.0  # sample kurtosis above n / 50 goes to the betting sequence
_SPREAD_LEVEL = 0.05 / 9  # the level r is tuned to: one primary claim of nine
_SPREAD_LOG = -2.0 * math.log(_SPREAD_LEVEL)
# ``r`` of Theorem 2.2: ONE constant of ours, prespecified, the same at every call and every level.
# The theorem takes any prespecified value, and the choice of this one is not the paper's: it is
# written down here, tuned by hand to n = 300 (the planner arms' size) at the level 0.05 / 9. The
# ``alpha`` of a call enters the width only through the logarithm.
SPREAD_R = (_SPREAD_LOG + math.log(_SPREAD_LOG + 1.0)) / 300.0

__all__ = [
    "SPREAD_KURTOSIS_DIVISOR",
    "SPREAD_MIN_PAIRS",
    "SPREAD_R",
    "CSAccumulator",
    "betting_cs",
    "paired_betting_cs",
    "paired_cs",
    "zero_event_upper",
]


def _check_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def _check_grid(grid: int) -> None:
    if isinstance(grid, bool) or not isinstance(grid, int | np.integer) or grid < 3:
        raise ValueError(f"grid must be an int of at least 3, got {grid!r}")


def _check_cap(c: float) -> None:
    if not 0.0 < c < 1.0:
        raise ValueError(f"the bet cap c must be in (0, 1), got {c!r}")


def _as_unit_array(x: Sequence[float] | NDArray) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError("every x must be a finite number in [0, 1]")
    return arr


def _predictable_bets(x: NDArray[np.float64], alpha: float, c: float) -> NDArray[np.float64]:
    """Bet ``lam_t`` for every step of ``x`` (along the last axis), each using only ``x[:t]``.

    ``lam_t = min(c, sqrt(2 ln(2/alpha) / (var_{t-1} * t * ln(1 + t))))`` with the running
    variance ``var_t = (1/4 + sum_{i<=t} (x_i - mu_i)^2) / (t + 1)`` and running mean
    ``mu_t = (1/2 + sum_{i<=t} x_i) / (t + 1)``, both shrunk towards a prior of (1/2, 1/4).
    """
    t = np.arange(1, x.shape[-1] + 1, dtype=np.float64)
    mu = (0.5 + np.cumsum(x, axis=-1)) / (t + 1.0)
    sq = np.cumsum((x - mu) ** 2, axis=-1)
    sq_before = np.concatenate([np.zeros_like(sq[..., :1]), sq[..., :-1]], axis=-1)
    var_before = (0.25 + sq_before) / t
    lam = np.sqrt(2.0 * math.log(2.0 / alpha) / (var_before * t * np.log1p(t)))
    return np.minimum(c, lam)


def _surviving_cells(
    log_up: NDArray[np.float64], log_down: NDArray[np.float64], alpha: float
) -> NDArray[np.bool_]:
    """Mask of cells not yet excluded; ``log_up`` and ``log_down`` are log K+ and log K-."""
    # Lower bound of log K inside cell k: K+ at its right edge and K- at its left edge.
    log_floor = np.logaddexp(log_up[..., 1:], log_down[..., :-1]) - math.log(2.0)
    return log_floor < -math.log(alpha)


def _interval_of(
    alive: NDArray[np.bool_], grid_pts: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Left edge of the first and right edge of the last surviving cell; empty gives (1, 0)."""
    any_alive = alive.any(axis=-1)
    first = np.argmax(alive, axis=-1)
    last = alive.shape[-1] - 1 - np.argmax(alive[..., ::-1], axis=-1)
    lower = np.where(any_alive, grid_pts[first], 1.0)
    upper = np.where(any_alive, grid_pts[last + 1], 0.0)
    return lower, upper


def betting_cs(
    x: Sequence[float] | NDArray,
    alpha: float = 0.05,
    grid: int = 1001,
    *,
    c: float = BET_CAP,
    intersect: bool = True,
) -> tuple[NDArray, NDArray]:
    """Anytime-valid confidence sequence for the mean of ``x`` in [0, 1].

    Returns ``(lower, upper)``, each of length ``len(x)``: the bounds after every prefix. With
    ``intersect=True`` (the default) the running intersection is returned, which is still a valid
    sequence and never widens; ``intersect=False`` gives the raw per-prefix sets. A pair with
    ``lower > upper`` means the set is empty, which has probability at most ``alpha``. Raises
    ``ValueError`` on values outside [0, 1], NaN or infinity.
    """
    _check_alpha(alpha)
    _check_grid(grid)
    _check_cap(c)
    return _bounds_on_points(
        _as_unit_array(x), alpha, c, np.linspace(0.0, 1.0, int(grid)), intersect
    )


def _bounds_on_points(
    xs: NDArray[np.float64],
    alpha: float,
    c: float,
    pts: NDArray[np.float64],
    intersect: bool,
) -> tuple[NDArray, NDArray]:
    """``betting_cs`` on an arbitrary increasing set of candidate means that spans [0, 1]."""
    n = xs.shape[0]
    lam = _predictable_bets(xs, alpha, c)
    lower = np.empty(n)
    upper = np.empty(n)
    carry_up = np.zeros(pts.shape[0])
    carry_down = np.zeros(pts.shape[0])
    for start in range(0, n, _BLOCK):
        stop = min(start + _BLOCK, n)
        step = lam[start:stop, None] * (xs[start:stop, None] - pts[None, :])
        log_up = carry_up + np.cumsum(np.log1p(step), axis=0)
        log_down = carry_down + np.cumsum(np.log1p(-step), axis=0)
        carry_up, carry_down = log_up[-1], log_down[-1]
        lower[start:stop], upper[start:stop] = _interval_of(
            _surviving_cells(log_up, log_down, alpha), pts
        )
    if intersect and n:
        lower = np.maximum.accumulate(lower)
        upper = np.minimum.accumulate(upper)
    return lower, upper


class CSAccumulator:
    """Online confidence sequence: feed values one at a time, read the current bounds.

    ``bounds()`` is the running intersection of the per-step sets, so it equals the last entry of
    ``betting_cs(xs, alpha, grid)`` for the values seen so far. Before any value it is (0, 1).
    """

    def __init__(self, alpha: float = 0.05, grid: int = 1001, *, c: float = BET_CAP) -> None:
        _check_alpha(alpha)
        _check_grid(grid)
        _check_cap(c)
        self._alpha = alpha
        self._c = c
        self._pts = np.linspace(0.0, 1.0, int(grid))
        self._log_up = np.zeros(self._pts.shape[0])
        self._log_down = np.zeros(self._pts.shape[0])
        self._sum = 0.0
        self._sq_sum = 0.0
        self._lower = 0.0
        self._upper = 1.0
        self.n = 0

    def update(self, x: float) -> None:
        """Add one observation. Raises ``ValueError`` if ``x`` is not a finite number in [0, 1]."""
        value = float(x)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"x must be a finite number in [0, 1], got {x!r}")
        t = float(self.n + 1)
        # The same formulas as _predictable_bets, with the sums carried instead of recomputed.
        var_before = (0.25 + self._sq_sum) / t
        lam = min(
            self._c, math.sqrt(2.0 * math.log(2.0 / self._alpha) / (var_before * t * np.log1p(t)))
        )
        self._sum += value
        mu = (0.5 + self._sum) / (t + 1.0)
        self._sq_sum += (value - mu) ** 2
        step = lam * (value - self._pts)
        self._log_up += np.log1p(step)
        self._log_down += np.log1p(-step)
        lo, hi = _interval_of(
            _surviving_cells(self._log_up, self._log_down, self._alpha), self._pts
        )
        self._lower = max(self._lower, float(lo))
        self._upper = min(self._upper, float(hi))
        self.n += 1

    def bounds(self) -> tuple[float, float]:
        """Current ``(lower, upper)``; ``lower > upper`` means the set is empty."""
        return self._lower, self._upper


def _checked_pairs(
    a: NDArray, b: NDArray, lo: float, hi: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """The two arms as float arrays, checked, and the width ``w = hi - lo`` of their range."""
    if not hi > lo:
        raise ValueError(f"need lo < hi, got lo={lo!r}, hi={hi!r}")
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    if arr_a.shape != arr_b.shape or arr_a.ndim != 1:
        raise ValueError(
            f"a and b must be one-dimensional and equal in shape: {arr_a.shape}, {arr_b.shape}"
        )
    for name, arr in (("a", arr_a), ("b", arr_b)):
        if not np.all(np.isfinite(arr)) or np.any(arr < lo) or np.any(arr > hi):
            raise ValueError(f"every value of {name} must be finite and within [{lo}, {hi}]")
    return arr_a, arr_b, float(hi) - float(lo)


def paired_betting_cs(
    a: NDArray, b: NDArray, lo: float, hi: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Confidence sequence for ``mean(a - b)`` with ``a``, ``b`` in ``[lo, hi]``, after all pairs.

    The differences lie in ``[-w, w]`` with ``w = hi - lo``; they are rescaled to [0, 1], the
    betting sequence is run, and its final bounds are mapped back to the original scale. Valid at
    every sample size and every distribution, and its half-width has a floor set by ``w`` and not
    by the spread of the differences (0003 A16): 0.080 at n = 300 for ``w = 1`` at the level
    ``0.05 / 9``, whatever the data. This is the function ``paired_cs`` falls back to.
    """
    arr_a, arr_b, w = _checked_pairs(a, b, lo, hi)
    if arr_a.shape[0] == 0:
        return -w, w
    unit = (arr_a - arr_b + w) / (2.0 * w)
    low, up = betting_cs(np.clip(unit, 0.0, 1.0), alpha)
    return float(low[-1]) * 2.0 * w - w, float(up[-1]) * 2.0 * w - w


def _spread_halfwidth(n: int, s: float, alpha: float) -> float:
    """``s * sqrt(2 (n r + 1) / (n^2 r) * log(sqrt(n r + 1) / alpha))`` with the module's ``r``."""
    r = SPREAD_R
    scale = 2.0 * (n * r + 1.0) / (n * n * r)
    return s * math.sqrt(scale * math.log(math.sqrt(n * r + 1.0) / alpha))


def paired_cs(
    a: NDArray, b: NDArray, lo: float, hi: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Confidence sequence for ``mean(a - b)`` with ``a``, ``b`` in ``[lo, hi]``, after all pairs.

    **Spread-scaled sequence.** With at least ``SPREAD_MIN_PAIRS`` (100) pairs and a sample
    kurtosis of the differences ``d`` of at most ``n / SPREAD_KURTOSIS_DIVISOR`` (``n / 50``) the
    result is the time-uniform sequence of Waudby-Smith, Arbour, Sinha, Kennedy and Ramdas,
    "Time-uniform central limit theory and asymptotic confidence sequences", Annals of
    Statistics 52(6), 2613-2640, 2024, Theorem 2.2::

        mean(d) +/- s * sqrt(2 (n r + 1) / (n^2 r) * log(sqrt(n r + 1) / alpha))

    clipped to ``[-w, w]`` with ``w = hi - lo``. ``s`` is the sample standard deviation (``n - 1``
    in the denominator); the kurtosis is the fourth central moment over the square of the second,
    both with ``n`` in the denominator. Its width follows the spread of the differences and not the
    range, which is why it can resolve effects the betting sequence cannot at the registered sizes.

    **Its validity is asymptotic**, not finite-sample: the theorem is a statement about a limit,
    and the repository checks the coverage at its sizes by simulation (``tests/stats`` and
    ``scripts/reachability.py``), not by proof. The kurtosis guard exists because the sequence
    alone missed at 67 times its level on an event of rate 0.02 in simulation (0003 A16).

    ``r`` is one fixed constant of ours (``SPREAD_R``), written down in advance and the same at
    every call. The theorem takes any prespecified value, and the paper does not choose this one.
    The ``alpha`` of the call enters only through the logarithm.

    **Fallback.** Under 100 pairs, a kurtosis above the limit, or no spread at all (every
    difference equal: ``s`` would be 0 and the interval a point) the result is
    ``paired_betting_cs``, which is valid at every size.
    """
    arr_a, arr_b, w = _checked_pairs(a, b, lo, hi)
    n = arr_a.shape[0]
    if n >= SPREAD_MIN_PAIRS:
        _check_alpha(alpha)
        d = arr_a - arr_b
        # a spread of exactly zero needs the extremes, not the variance: the mean of n equal
        # floats can differ from them by an ulp, and the variance of that is not 0
        if float(np.max(d)) != float(np.min(d)):
            mean = float(d.mean())
            dev = d - mean
            # kurtosis does not depend on the scale, and powers of a tiny or huge deviation
            # underflow or overflow (a difference of 1e-91 has a fourth power of 0): work on
            # deviations scaled to a peak of 1
            peak = float(np.max(np.abs(dev)))
            z = dev / peak
            m2 = float(np.mean(z**2))
            m4 = float(np.mean(z**4))
            if m2 > 0.0 and m4 / (m2 * m2) <= n / SPREAD_KURTOSIS_DIVISOR:
                s = peak * math.sqrt(float(np.sum(z**2)) / (n - 1))
                half = _spread_halfwidth(n, s, alpha)
                return max(-w, mean - half), min(w, mean + half)
    return paired_betting_cs(arr_a, arr_b, lo, hi, alpha)


def zero_event_upper(n: int, alpha: float = 0.05) -> float:
    """Upper bound on a rate after ``n`` trials with no event, from the same betting sequence.

    Anytime-valid, so it costs more than the fixed-``n`` rule of three (about ``3 / n`` at 5%);
    the ratio is reported by the tests and the progress note. ``n = 0`` gives 1. The candidate
    means are packed into ``[0, 4 ln(2/alpha) / n]`` (plus the point 1), because the answer is
    of order ``1 / n`` and a uniform grid of step 0.001 would dominate it for ``n`` in the
    thousands; the cell rule keeps the result rigorous whatever the spacing.
    """
    if isinstance(n, bool) or not isinstance(n, int | np.integer) or n < 0:
        raise ValueError(f"n must be a non-negative int, got {n!r}")
    _check_alpha(alpha)
    if n == 0:
        return 1.0
    reach = min(1.0, 4.0 * math.log(2.0 / alpha) / n)
    pts = np.append(np.linspace(0.0, reach, 1001), 1.0) if reach < 1.0 else np.linspace(0, 1, 1001)
    _, upper = _bounds_on_points(np.zeros(int(n)), alpha, BET_CAP, pts, True)
    return float(upper[-1])
