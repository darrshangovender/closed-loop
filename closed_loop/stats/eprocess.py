"""Anytime-valid e-process for a bounded stream whose conditional mean may exceed ``p0``.

Interface text: docs/adr/0002-interfaces.md, section 3.3, as amended by docs/adr/0003-review-
amendments.md, A6 (``BoundedEProcess`` replaces ``BernoulliEProcess``).

Null hypothesis H0: ``E[x_i | x_1..x_{i-1}] <= p0`` for every i, with each ``x_i`` in [0, 1].
Under H0 the capital of a bet of fixed size ``lam >= 0``, ``prod_i (1 + lam (x_i - p0))``, is a
non-negative supermartingale started at 1. The process averages 20 such bets, ``lam = u / p0``
with ``u`` spaced geometrically from 0.005 to 0.99 (so every factor stays positive for any
``x`` in [0, 1]). An average of supermartingales is a supermartingale, so Ville's inequality
gives ``P(E_n >= 1/delta for some n) <= delta`` however long the stream runs and whatever stopping
rule the user applies. The fixed grid trades some power against the best single bet for not
having to choose it.

The unit of the sequence matters. The guarantee needs the conditional-mean condition, which
holds when the items are independent given the past, for example one value per life with lives
independent. Per-checkpoint labels from one life resolve together and are strongly dependent;
fed to this test one by one they break the condition and inflate the false-alarm rate (the test
in tests/stats/test_eprocess.py measures how much). That is why ``CoverageMonitor`` pushes one
bounded value per life.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["BoundedEProcess"]

N_BETS = 20
U_MIN = 0.005
U_MAX = 0.99
_LOG_CAP = 700.0  # exp(700) is finite; e_value saturates here instead of overflowing


class BoundedEProcess:
    """E-process against H0: conditional mean of x <= ``p0``, with ``x`` in [0, 1].

    ``update(x)`` returns the current e-value. ``fired`` latches once the e-value reaches
    ``1 / delta`` and stays set; the e-value keeps updating afterwards. ``fired_at`` is the value
    of ``n`` at the first firing, or None.
    """

    def __init__(self, p0: float, delta: float = 0.01) -> None:
        if not 0.0 < p0 < 1.0:
            raise ValueError(f"p0 must be in (0, 1), got {p0!r}")
        if not 0.0 < delta < 1.0:
            raise ValueError(f"delta must be in (0, 1), got {delta!r}")
        self.p0 = float(p0)
        self.delta = float(delta)
        u = np.geomspace(U_MIN, U_MAX, N_BETS)
        self._bets = u / self.p0
        self._log_bets = np.zeros(N_BETS)  # log capital of each bet
        self._log_threshold = math.log(1.0 / self.delta)
        self._log_e = 0.0
        self.fired = False
        self.fired_at: int | None = None
        self.n = 0

    @property
    def e_value(self) -> float:
        """Current e-value (the average capital), saturating at ``exp(700)``."""
        return math.exp(min(self._log_e, _LOG_CAP))

    @property
    def log_e_value(self) -> float:
        """Natural log of the current e-value, exact for any length of stream."""
        return self._log_e

    def update(self, x: float) -> float:
        """Add one value in [0, 1] and return the current e-value.

        Raises ``ValueError`` for NaN, infinity or a value outside [0, 1]; ``bool`` is accepted
        as 0 or 1.
        """
        value = float(x)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"x must be a finite number in [0, 1], got {x!r}")
        self._log_bets += np.log1p(self._bets * (value - self.p0))
        top = float(np.max(self._log_bets))
        total = float(np.exp(self._log_bets - top).sum())
        self._log_e = top + math.log(total) - math.log(N_BETS)
        self.n += 1
        if not self.fired and self._log_e >= self._log_threshold:
            self.fired = True
            self.fired_at = self.n
        return self.e_value
