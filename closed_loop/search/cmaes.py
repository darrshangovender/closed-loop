"""(mu/mu_w, lambda)-CMA-ES with box repair and IPOP restarts, in numpy only.

Interface text: docs/adr/0002-interfaces.md, section 3.11; design: docs/adr/0001-architecture.md,
section 9. The update equations are those of Hansen's tutorial (2016) with positive weights:
log-weights, cumulative step-size adaptation, rank-one and rank-mu covariance updates.

Box handling. The search space is ``[0, 1]^dim``. ``ask`` returns the sampled points clipped into
the box (so the objective is never evaluated outside it), and ``tell`` ranks the batch by
``f + range * (d / sigma)^2``, where ``d`` is how far the sample was from its clipped point and
``range`` is the spread of ``f`` in the batch. A point one step size outside the box therefore
pays one full range of ``f``, whatever the scale of ``f``. The distribution update uses the
unclipped samples, so the search can still push towards a face of the box.

IPOP restarts. When a stopping criterion fires (tolerance on ``f`` or on the step, an
ill-conditioned covariance, a coordinate the step no longer moves, or stagnation), the run
restarts from a new random mean with the population doubled, up to ``restarts`` times. After the
last restart the criteria are logged but ignored: the caller's budget decides when to stop.

``f`` is minimised. Infinite values are allowed (they rank last or first); NaN is refused.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.search.random_search import SearchRun, summarise_run

__all__ = ["CMAES", "cmaes_search"]

_TOLFUN = 1e-12
_TOLX = 1e-12
_MAX_COND = 1e14
_SIGMA_MAX = 2.0  # twice the width of the box
_SIGMA_MIN = 1e-16


class CMAES:
    """Ask-and-tell CMA-ES on ``[0, 1]^dim``.

    ``ask()`` returns ``popsize`` points; ``tell(x, f)`` takes those same points back with their
    objective values. ``best_x``/``best_f`` hold the best point told so far (``best_f`` is ``inf``
    before the first ``tell``), ``evals`` the number of values told. The population size doubles on
    every restart, so ``popsize`` is the current one. Same ``seed``, same sequence of points.
    """

    def __init__(
        self,
        dim: int,
        seed: int,
        popsize: int | None = None,
        sigma0: float = 0.3,
        restarts: int = 3,
    ) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int | np.integer) or dim < 1:
            raise ValueError(f"dim must be an int of at least 1, got {dim!r}")
        if popsize is not None and (
            isinstance(popsize, bool) or not isinstance(popsize, int | np.integer) or popsize < 4
        ):
            raise ValueError(f"popsize must be an int of at least 4, got {popsize!r}")
        if not (math.isfinite(sigma0) and sigma0 > 0.0):
            raise ValueError(f"sigma0 must be positive and finite, got {sigma0!r}")
        if isinstance(restarts, bool) or not isinstance(restarts, int | np.integer) or restarts < 0:
            raise ValueError(f"restarts must be a non-negative int, got {restarts!r}")
        self.dim = int(dim)
        self.seed = int(seed)
        self.sigma0 = float(sigma0)
        self.max_restarts = int(restarts)
        self.base_popsize = int(popsize) if popsize is not None else 4 + int(3 * math.log(dim))
        self.restarts_used = 0
        self.stop_log: list[tuple[int, str]] = []  # (evals, reason) each time a criterion fired
        self.best_x: NDArray[np.float64] = np.full(self.dim, 0.5)
        self.best_f: float = math.inf
        self.evals = 0
        self._pending: tuple[NDArray, NDArray, NDArray] | None = None  # clipped, raw y, unclipped x
        self._start_run()

    # ------------------------------------------------------------------ one run

    def _start_run(self) -> None:
        n = self.dim
        self.popsize = self.base_popsize * 2**self.restarts_used
        self._rng = rng_for(self.seed, "cmaes", self.restarts_used)
        mu = self.popsize // 2
        raw = math.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self._weights = raw / raw.sum()
        self._mu_eff = float(1.0 / np.sum(self._weights**2))
        me = self._mu_eff
        self._cs = (me + 2.0) / (n + me + 5.0)
        self._ds = 1.0 + 2.0 * max(0.0, math.sqrt((me - 1.0) / (n + 1.0)) - 1.0) + self._cs
        self._cc = (4.0 + me / n) / (n + 4.0 + 2.0 * me / n)
        self._c1 = 2.0 / ((n + 1.3) ** 2 + me)
        self._cmu = min(1.0 - self._c1, 2.0 * (me - 2.0 + 1.0 / me) / ((n + 2.0) ** 2 + me))
        self._chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.mean = self._rng.random(n)
        self.sigma = self.sigma0
        self._pc = np.zeros(n)
        self._ps = np.zeros(n)
        self._cov = np.eye(n)
        self._basis = np.eye(n)  # B
        self._scales = np.ones(n)  # D, the square roots of the eigenvalues
        self.generation = 0
        window = 10 + math.ceil(30 * n / self.popsize)
        self._best_hist: deque[float] = deque(maxlen=window)
        self._stagnation_hist: list[float] = []
        self._pending = None

    def _decompose(self) -> None:
        self._cov = (self._cov + self._cov.T) / 2.0
        values, vectors = np.linalg.eigh(self._cov)
        if np.min(values) <= 0.0:  # numerical breakdown: shift to a valid matrix
            values = np.maximum(values, 1e-30)
            self._cov = (vectors * values) @ vectors.T
        self._basis = vectors
        self._scales = np.sqrt(values)

    def ask(self) -> NDArray[np.float64]:
        """The next ``popsize`` points, shape ``(popsize, dim)``, all inside ``[0, 1]``."""
        if self._pending is not None:
            raise RuntimeError("tell() the previous batch before asking for another")
        z = self._rng.standard_normal((self.popsize, self.dim))
        y = (z * self._scales) @ self._basis.T
        x_raw = self.mean + self.sigma * y
        x = np.clip(x_raw, 0.0, 1.0)
        self._pending = (x, y, x_raw)
        return x.copy()

    def tell(self, x: NDArray[np.float64], f: NDArray[np.float64]) -> None:
        """Report the objective values of the batch returned by the last ``ask``."""
        if self._pending is None:
            raise RuntimeError("ask() before tell()")
        asked, y, x_raw = self._pending
        xs = np.asarray(x, dtype=np.float64)
        fs = np.asarray(f, dtype=np.float64)
        if xs.shape != asked.shape or not np.array_equal(xs, asked):
            raise ValueError("x must be the array returned by the last ask()")
        if fs.shape != (self.popsize,):
            raise ValueError(f"f must have shape ({self.popsize},), got {fs.shape}")
        if np.isnan(fs).any():
            raise ValueError("f contains NaN")
        self._pending = None
        self.evals += self.popsize
        best = int(np.argmin(fs))
        if fs[best] < self.best_f:
            self.best_f = float(fs[best])
            self.best_x = asked[best].copy()

        ranked = self._ranking_values(fs, x_raw, asked)
        order = np.argsort(ranked, kind="stable")
        mu = self._weights.shape[0]
        chosen = order[:mu]
        y_w = self._weights @ y[chosen]
        old_mean = self.mean
        self.mean = old_mean + self.sigma * y_w
        self._update_paths_and_covariance(y[chosen], y_w)
        self._best_hist.append(float(fs[best]))
        self._stagnation_hist.append(float(fs[best]))
        self.generation += 1
        self._decompose()
        self._maybe_restart(fs)

    def _ranking_values(
        self, fs: NDArray[np.float64], x_raw: NDArray[np.float64], x_box: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        finite = fs[np.isfinite(fs)]
        low = float(np.min(finite)) if finite.size else 0.0
        high = float(np.max(finite)) if finite.size else 0.0
        spread = high - low
        values = np.where(np.isposinf(fs), high + spread + 1.0, fs)
        values = np.where(np.isneginf(fs), low - spread - 1.0, values)
        scale = spread if spread > 0.0 else 1e-12 * (1.0 + abs(high))
        outside = np.sum((x_raw - x_box) ** 2, axis=1) / (self.sigma**2)
        return np.asarray(values + scale * outside, dtype=np.float64)

    def _update_paths_and_covariance(
        self, y_sel: NDArray[np.float64], y_w: NDArray[np.float64]
    ) -> None:
        n, cs, cc = self.dim, self._cs, self._cc
        inv_sqrt_c_y = self._basis @ ((self._basis.T @ y_w) / self._scales)
        self._ps = (1.0 - cs) * self._ps + math.sqrt(cs * (2.0 - cs) * self._mu_eff) * inv_sqrt_c_y
        ps_norm = float(np.linalg.norm(self._ps))
        decay = math.sqrt(1.0 - (1.0 - cs) ** (2 * (self.generation + 1)))
        h_sigma = 1.0 if ps_norm / decay < (1.4 + 2.0 / (n + 1.0)) * self._chi_n else 0.0
        self._pc = (1.0 - cc) * self._pc + h_sigma * math.sqrt(cc * (2.0 - cc) * self._mu_eff) * y_w
        rank_mu = (y_sel.T * self._weights) @ y_sel
        keep = 1.0 - self._c1 - self._cmu
        self._cov = (
            keep * self._cov
            + self._c1
            * (np.outer(self._pc, self._pc) + (1.0 - h_sigma) * cc * (2.0 - cc) * self._cov)
            + self._cmu * rank_mu
        )
        exponent = (cs / self._ds) * (ps_norm / self._chi_n - 1.0)
        self.sigma = float(
            min(max(self.sigma * math.exp(min(exponent, 1.0)), _SIGMA_MIN), _SIGMA_MAX)
        )

    # ------------------------------------------------------------------ restarts

    def _stop_reason(self, fs: NDArray[np.float64]) -> str | None:
        diag = np.diag(self._cov)
        step = self.sigma * math.sqrt(float(np.max(diag)))
        if step < _TOLX * self.sigma0:
            return "tolx"
        if np.any(self.mean + 0.2 * self.sigma * np.sqrt(diag) == self.mean):
            return "noeffectcoord"
        if float(np.max(self._scales) / np.min(self._scales)) ** 2 > _MAX_COND:
            return "condition"
        window = self._best_hist.maxlen or 1
        if len(self._best_hist) == window:
            finite = np.isfinite(fs)
            if (
                max(self._best_hist) - min(self._best_hist) < _TOLFUN
                and finite.any()
                and float(np.max(fs[finite]) - np.min(fs[finite])) < _TOLFUN
            ):
                return "tolfun"
        horizon = int(100 + 100 * self.dim**1.5 / self.popsize)
        hist = self._stagnation_hist
        if len(hist) > horizon and np.median(hist[-20:]) >= np.median(hist[-horizon:][:20]):
            return "stagnation"
        return None

    def _maybe_restart(self, fs: NDArray[np.float64]) -> None:
        reason = self._stop_reason(fs)
        if reason is None:
            return
        self.stop_log.append((self.evals, reason))
        if self.restarts_used < self.max_restarts:
            self.restarts_used += 1
            self._start_run()


def cmaes_search(
    f: Callable[[NDArray[np.float64]], float],
    dim: int,
    budget: int,
    seed: int,
    *,
    regime: str = "",
    **kwargs: float,
) -> SearchRun:
    """Run ``CMAES`` for ``budget`` evaluations of ``f`` and return a ``SearchRun``.

    The last generation is cut short if it would pass ``budget``; those points are not evaluated.
    Extra keyword arguments (``popsize``, ``sigma0``, ``restarts``) go to ``CMAES``.
    """
    if isinstance(budget, bool) or not isinstance(budget, int | np.integer) or budget < 1:
        raise ValueError(f"budget must be an int of at least 1, got {budget!r}")
    opt = CMAES(dim, seed, **kwargs)  # type: ignore[arg-type]
    points: list[NDArray[np.float64]] = []
    values: list[float] = []
    while len(values) < budget:
        batch = opt.ask()
        scores = np.empty(batch.shape[0])
        take = min(batch.shape[0], budget - len(values))
        for i in range(batch.shape[0]):
            # points past the budget are told as +inf: ranked last, never counted or returned
            scores[i] = float(f(batch[i].copy())) if i < take else math.inf
            if i < take:
                points.append(batch[i].copy())
                values.append(float(scores[i]))
        opt.tell(batch, scores)
    return summarise_run("cmaes", regime, seed, np.array(values), np.array(points))
