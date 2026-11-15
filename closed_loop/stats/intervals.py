"""Wilson score interval and the exact (Clopper-Pearson) interval for a proportion.

Interface text: docs/adr/0002-interfaces.md, section 3.3. ``wilson`` states the Monte Carlo error of
a rate measured in a test or in a report (``k`` events out of ``n``), never as a stopping rule.
``clopper_pearson`` is the interval C-SEARCH is decided on (0003 A16): exact, so valid at ten runs.
"""

from __future__ import annotations

import math
from numbers import Integral

__all__ = ["clopper_pearson", "wilson"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for ``k`` successes in ``n`` trials (two-sided, default 95%).

    ``n = 0`` gives the uninformative ``(0, 1)``. Raises ``ValueError`` unless ``0 <= k <= n``
    and ``z > 0``. The bounds are exactly 0 for ``k = 0`` and exactly 1 for ``k = n``.
    """
    if z <= 0.0 or not math.isfinite(z):
        raise ValueError(f"z must be positive and finite, got {z!r}")
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k!r}, n={n!r}")
    if n == 0:
        return 0.0, 1.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided interval for ``k`` successes in ``n`` trials, coverage at least ``1 - alpha``
    (``alpha / 2`` in each tail), from the beta quantiles.

    The ends are the usual ones: the lower bound is exactly 0 for ``k = 0`` and the upper exactly
    1 for ``k = n``. ``n = 0`` gives the uninformative ``(0, 1)``. Raises ``ValueError`` unless
    ``0 <= k <= n`` (integers) and ``0 < alpha < 1``. It is a fixed-``n`` interval, not a
    sequence: look at it once.
    """
    for name, value in (("k", k), ("n", n)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an int, got {value!r}")
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k!r}, n={n!r}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
    k, n = int(k), int(n)
    if n == 0:
        return 0.0, 1.0
    from scipy.stats import beta  # type: ignore[import-untyped]  # heavy import

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi
