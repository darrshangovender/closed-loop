"""The structural causal model of the plant's degradation, as a diagnostician could write it.

*Structure handed to the model.* The equations are those of 0001 section 7, restricted to what an
engineer could write down from the asset's design, not from the simulator's code:

* a wear clock ``tau`` advances at ``rho(L) = exp(beta (L - 1))`` per day under load ``L``;
* severity is ``s = clip((tau - tau_on) / D, 0, 1) ** k`` once the clock passes the onset
  ``tau_on``, and the component fails when ``tau`` reaches ``tau_on + D``;
* the health index is ``hi = h0 + c_h (L - 1) + (hf - h0) s + noise``;
* a DERATE multiplies the load by a retained fraction.

*Handed in, not learned*: this graph, and ``hf``, the health-index level at failure (the class's
published failure threshold, 9.0). Everything else is estimated from the training lives:
``h0``, ``c_h`` and the noise level by regression on the first days of each life (before any onset);
the retained fraction of the operator's derate from the step it makes in the observed load; and
``beta`` by profile least squares over lives, each fitted by abduction. So ``C-CAUSE`` is **favoured
by construction**: the model is given the right structure, and the pre-registered claim is read with
that in mind. The misspecified model (``misspecified=True``) drops the interaction of load with age,
that is, the clock rate no longer depends on load, so a derate changes the health-index offset only
and cannot postpone the failure; it shows how much the result leans on the structure.

*Abduction-action-prediction.* For one observed life:

1. **Abduction.** Rebuild the load the asset really saw (undo the operator's derate), integrate the
   clock, and fit the life's exogenous terms ``(tau_on, D, k)`` to its health-index path by
   nonlinear least squares. A failure that was seen fixes ``tau_on + D`` to the clock at that
   failure.
2. **Action.** Replace the load from the query's day on by ``load_cap`` times the base load (a
   ``load_cap`` of 1 removes any derate, the world where nobody acts).
3. **Prediction.** Integrate the clock under the new load, with the load beyond the record forecast
   by its recent mean, and read off the day the clock reaches ``tau_on + D``. The crossing is
   rounded up to the plant's 15-minute grid, as the plant does.

Without step 1 the model could only guess the clock total from the training population; the mutation
test removes it and the noise-free exactness test fails.

Interfaces beyond the frozen text of 0002 section 3.9: ``abduct``, ``beta``, ``derate_factor`` and
the other fitted attributes, and the keyword options of the constructor. Everything is SYNTHETIC.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop.causal.records import LifeRecord

DAY_Q = 96
MAX_DAY = 400.0
BASELINE_DAYS = 20  # a new component's first days carry no wear (21 d baseline, S: FeatureConfig)
FORECAST_DAYS = 14  # recent load window used to forecast beyond the record
_EPS = 1e-9
_FLOOR = 1e-9  # a residual spread below this is numerical noise, not data
_N_ON = 70  # onset grid points of the abduction search
_N_EXT = 14  # grid points for how far past the record a censored life's failure lies
_K_GRID = np.exp(np.linspace(math.log(0.8), math.log(6.5), 22))


def _best_cells(sse: NDArray[np.float64], n: int) -> list[tuple[int, int]]:
    """Indices of the ``n`` smallest cells of a 2-D array, best first."""
    order = np.argsort(sse, axis=None)[:n]
    return [(int(i // sse.shape[1]), int(i % sse.shape[1])) for i in order]


@dataclass(frozen=True)
class Abduction:
    """The exogenous terms of one life inferred from its record (clock days)."""

    tau_on: float
    dur: float
    shape: float
    tau_fail: float  # tau_on + dur
    sse: float
    n: int


def norm_cdf(x: float) -> float:
    """The standard normal distribution function (``math.erf``, so no scipy.stats import)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class SCMEffectModel:
    """SCM with abduction (see the module docstring); ``name`` is ``scm`` or its variant."""

    def __init__(
        self,
        *,
        misspecified: bool = False,
        hi_fail: float = 9.0,
        default_derate_factor: float = 0.7,
        n_beta_lives: int = 60,
        beta_tol: float = 0.002,
        trim: float = 0.8,
    ) -> None:
        self.misspecified = misspecified
        self.hi_fail = float(hi_fail)
        self.default_derate_factor = float(default_derate_factor)
        self.n_beta_lives = int(n_beta_lives)
        if not 0.5 <= trim <= 1.0:
            raise ValueError("trim must be in [0.5, 1]")
        self.beta_tol = float(beta_tol)
        self.trim = float(trim)
        self.name = "scm_misspecified" if misspecified else "scm"
        self.fitted = False
        # fitted values
        self.h0 = 1.0
        self.c_h = 0.0
        self.sigma_h = 0.3
        self.beta = 0.0
        self.beta_se = 0.0
        self.derate_factor = self.default_derate_factor
        self.derate_factor_source = "default"
        self.fresh_clock: NDArray[np.float64] = np.zeros(0)
        self.load_sd = 0.05
        self._cache: dict[tuple[int, float], tuple[LifeRecord, Abduction]] = {}

    # ------------------------------------------------------------------------------- the clock
    @property
    def _a(self) -> float:
        return self.hi_fail - self.h0

    def _base_load(self, life: LifeRecord, factor: float) -> NDArray[np.float64]:
        """The load the asset would have seen undisturbed: the observed load, derate undone."""
        base = life.load.copy()
        if life.derate_days.size:
            idx = np.searchsorted(life.days, life.derate_days)
            base[idx] = base[idx] / factor
        return base

    def _rho(
        self,
        base: NDArray[np.float64],
        beta: float,
        n_total: int,
        derate_from: float | None,
        cap: float,
    ) -> NDArray[np.float64]:
        """Clock advance per day for days ``0 .. n_total - 1``: the record, then a load forecast."""
        n = base.size
        recent = base[-FORECAST_DAYS:]
        mu = float(recent.mean())
        var = float(np.var(base)) if n > 1 else 0.0
        ext = np.concatenate([base, np.full(max(n_total - n, 0), mu)])[:n_total]
        jensen = np.concatenate([np.zeros(n), np.full(max(n_total - n, 0), 1.0)])[:n_total]
        undisturbed = np.exp(beta * (ext - 1.0) + 0.5 * beta**2 * var * jensen)
        if derate_from is None:
            return undisturbed
        derated = np.exp(beta * (cap * ext - 1.0) + 0.5 * (beta * cap) ** 2 * var * jensen)
        frac = np.clip(np.arange(n_total) + 1.0 - derate_from, 0.0, 1.0)
        return (1.0 - frac) * undisturbed + frac * derated

    def _factual_clock(self, life: LifeRecord, beta: float) -> NDArray[np.float64]:
        """Clock at the start of each recorded day, under the load and derate that happened."""
        self._need_contiguous(life)
        base = self._base_load(life, self.derate_factor)
        n = life.days.size
        # The factual derate is in the observed load already; undone above, it is reapplied here.
        start = life.derate_start
        if start is None and life.derate_days.size:
            start = float(life.derate_days[0])
        rho = self._rho(base, beta, n, start, self.derate_factor)
        return np.concatenate([[0.0], np.cumsum(rho)])

    @staticmethod
    def _need_contiguous(life: LifeRecord) -> None:
        d = life.days
        if d.size == 0:
            raise ValueError("the SCM needs the daily load: this record has no days")
        if d[0] != life.life_start_day or np.any(np.diff(d) != 1.0):
            raise ValueError("the SCM needs consecutive daily readings from the life's start")

    # --------------------------------------------------------------------------- abduction
    def abduct(self, life: LifeRecord, beta: float | None = None) -> Abduction:
        """Fit ``(tau_on, D, k)`` to the life's health-index path (step 1 of the module doc)."""
        b = self.beta if beta is None else beta
        key = (id(life), round(b, 9))
        hit = self._cache.get(key)
        if hit is not None and hit[0] is life:
            return hit[1]
        out = self._abduct(life, b)
        if len(self._cache) > 4000:
            self._cache.clear()
        self._cache[key] = (life, out)
        return out

    def _abduct(self, life: LifeRecord, b: float) -> Abduction:
        cum = self._factual_clock(life, 0.0 if self.misspecified else b)
        idx = (life.hi_days - life.days[0]).astype(int)
        tau_h = cum[idx]
        dl = life.load[idx] - 1.0
        res = life.hi - self.h0 - self.c_h * dl
        tau_end = float(cum[-1])
        a = self._a
        if life.fail_day is not None:
            tau_fail_obs = self._clock_at(life, cum, life.fail_day)
            return self._abduct_failed(tau_h, res, a, tau_fail_obs)
        return self._abduct_censored(tau_h, res, a, tau_end)

    def _abduct_failed(self, tau_h: NDArray, res: NDArray, a: float, tau_fail: float) -> Abduction:
        """A seen failure fixes ``tau_on + D``: search ``(tau_on, k)``, then polish."""
        from scipy.optimize import least_squares  # type: ignore[import-untyped]  # heavy import

        on = np.linspace(0.0, max(tau_fail - 0.5, 0.0), _N_ON)
        ks = _K_GRID
        x = np.clip((tau_h[None, :] - on[:, None]) / (tau_fail - on)[:, None], 0.0, 1.0)
        sse = ((res[None, None, :] - a * x[:, None, :] ** ks[None, :, None]) ** 2).sum(axis=2)
        starts = _best_cells(sse, 2)

        def resid(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            xx = np.clip((tau_h - theta[0]) / (tau_fail - theta[0]), 0.0, 1.0)
            return res - a * np.power(xx, math.exp(theta[1]))

        lo = np.array([0.0, math.log(_K_GRID[0] * 0.5)])
        hi = np.array([max(tau_fail - 0.5, 1e-6), math.log(_K_GRID[-1] * 1.5)])
        best = None
        for ia, ib in starts:
            x0 = np.clip(np.array([on[ia], math.log(ks[ib])]), lo, hi)
            sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=60)
            if best is None or sol.cost < best.cost:
                best = sol
        assert best is not None
        tau_on = float(best.x[0])
        return Abduction(
            tau_on, tau_fail - tau_on, math.exp(best.x[1]), tau_fail, 2.0 * best.cost, int(res.size)
        )

    def _abduct_censored(self, tau_h: NDArray, res: NDArray, a: float, tau_end: float) -> Abduction:
        """No failure seen: the failure is later than the record, ``tau_on + D > tau_end``."""
        from scipy.optimize import least_squares  # heavy import

        on = np.linspace(0.0, tau_end, _N_ON)
        ext = np.exp(
            np.linspace(math.log(0.5), math.log(300.0), _N_EXT)
        )  # D = (tau_end - on) + ext
        ks = _K_GRID[::2]
        dur = (tau_end - on)[:, None] + ext[None, :]  # (A, E)
        x = np.clip((tau_h[None, None, :] - on[:, None, None]) / dur[:, :, None], 0.0, 1.0)
        pred = a * x[:, :, None, :] ** ks[None, None, :, None]  # (A, E, K, n)
        sse = ((res - pred) ** 2).sum(axis=3).reshape(on.size * ext.size, ks.size)
        flat = _best_cells(sse, 2)

        def resid(theta: NDArray[np.float64]) -> NDArray[np.float64]:
            d = max(tau_end - theta[0], 0.0) + math.exp(theta[1])
            xx = np.clip((tau_h - theta[0]) / d, 0.0, 1.0)
            return res - a * np.power(xx, math.exp(theta[2]))

        lo = np.array([0.0, math.log(0.05), math.log(_K_GRID[0] * 0.5)])
        hi = np.array([tau_end + 1.0, math.log(400.0), math.log(_K_GRID[-1] * 1.5)])
        best = None
        for cell, ib in flat:
            ia, ie = divmod(cell, ext.size)
            x0 = np.clip(np.array([on[ia], math.log(ext[ie]), math.log(ks[ib])]), lo, hi)
            sol = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=80)
            if best is None or sol.cost < best.cost:
                best = sol
        assert best is not None
        tau_on = float(best.x[0])
        dur_fit = max(tau_end - tau_on, 0.0) + math.exp(best.x[1])
        return Abduction(
            tau_on, dur_fit, math.exp(best.x[2]), tau_on + dur_fit, 2.0 * best.cost, int(res.size)
        )

    @staticmethod
    def _clock_at(life: LifeRecord, cum: NDArray[np.float64], day: float) -> float:
        """Clock at ``day`` (linear inside a day); a failure quarter counts from its start."""
        d = int(min(math.floor(day + _EPS), cum.size - 2))
        rho = cum[d + 1] - cum[d]
        # the plant fails at the first quarter whose start has the clock at the target, so the clock
        # at the failure quarter's start is at most one quarter above the target: take the middle
        return float(cum[d] + (day - d) * rho - 0.5 * rho / DAY_Q)

    # ----------------------------------------------------------------------------- fitting
    def fit(self, lives: Sequence[LifeRecord]) -> None:
        if not lives:
            raise ValueError("fit needs at least one life")
        self._fit_derate_factor(lives)
        self._fit_observation(lives)
        if self.misspecified:
            self.beta = 0.0
            self.beta_se = 0.0
        else:
            self._fit_beta(lives)
        self._fit_fresh_and_load(lives)
        self.fitted = True

    def _fit_derate_factor(self, lives: Sequence[LifeRecord]) -> None:
        ratios = []
        for lf in lives:
            if lf.derate_days.size and lf.days.size:
                d1 = int(lf.derate_days[0])
                i = int(np.searchsorted(lf.days, d1))
                if i >= 1 and lf.days[i] == d1 and lf.days[i - 1] == d1 - 1:
                    ratios.append(lf.load[i] / lf.load[i - 1])
        if ratios:
            self.derate_factor = float(np.median(ratios))
            self.derate_factor_source = f"median of {len(ratios)} derate steps in the observed load"
        else:
            self.derate_factor = self.default_derate_factor
            self.derate_factor_source = "default (no derated life in the training data)"

    def _fit_observation(self, lives: Sequence[LifeRecord]) -> None:
        xs, ys = [], []
        for lf in lives:
            if not lf.days.size:
                continue
            m = (lf.hi_days >= lf.life_start_day) & (lf.hi_days < lf.life_start_day + BASELINE_DAYS)
            if not m.any():
                continue
            idx = (lf.hi_days[m] - lf.days[0]).astype(int)
            xs.append(np.column_stack([np.ones(idx.size), lf.load[idx] - 1.0]))
            ys.append(lf.hi[m])
        if not xs:
            raise ValueError("no baseline days in the training lives")
        X, y = np.vstack(xs), np.concatenate(ys)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        sd = float(res.std())
        if sd > _FLOOR:  # one trimmed refit against the odd gain-drift life
            keep = np.abs(res) <= 3.0 * sd
            if keep.sum() >= 3:
                coef, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
                res = y[keep] - X[keep] @ coef
        self.h0, self.c_h = float(coef[0]), float(coef[1])
        self.sigma_h = max(float(res.std(ddof=2)), 1e-6) if res.size > 2 else 1e-6

    def _beta_sse(self, subset: Sequence[LifeRecord], beta: float) -> float:
        """Least-trimmed sum of the lives' squared residuals at ``beta``: the worst ``1 - trim`` of
        the lives are left out, so a life whose health index carries an unmodelled sensor drift
        cannot drag the load sensitivity (an un-trimmed profile gave beta near 0 with 30% such
        lives)."""
        sse = np.sort([self.abduct(lf, beta).sse for lf in subset])
        return float(sse[: max(int(math.ceil(self.trim * sse.size)), 1)].sum())

    def _fit_beta(self, lives: Sequence[LifeRecord]) -> None:
        from scipy.optimize import minimize_scalar  # heavy import

        a = self._a
        usable = [
            lf for lf in lives if lf.days.size and np.count_nonzero(lf.hi > self.h0 + 0.1 * a) >= 8
        ]
        if len(usable) < 3:
            raise ValueError("too few degrading lives to estimate the load sensitivity")
        # derated lives identify beta best; take them first, then spread the rest evenly
        usable.sort(key=lambda lf: (lf.derate_start is None,))
        step = max(len(usable) // self.n_beta_lives, 1)
        subset = usable[::step][: self.n_beta_lives]
        r = minimize_scalar(
            lambda b: self._beta_sse(subset, b),
            bounds=(0.0, 2.0),
            method="bounded",
            options={"xatol": self.beta_tol},
        )
        self.beta = float(r.x)
        h = 0.1
        lo, mid, hi = (self._beta_sse(subset, b) for b in (self.beta - h, self.beta, self.beta + h))
        curv = (lo - 2.0 * mid + hi) / h**2
        n_pts = sum(int(np.count_nonzero(np.isfinite(lf.hi))) for lf in subset)
        s2 = max(mid / max(n_pts - 3 * len(subset), 1), self.sigma_h**2)
        self.beta_se = float(math.sqrt(2.0 * s2 / curv)) if curv > 0.0 else 0.0

    def _fit_fresh_and_load(self, lives: Sequence[LifeRecord]) -> None:
        totals, sds = [], []
        for lf in lives:
            if not lf.days.size:
                continue
            ab = self.abduct(lf)
            totals.append(ab.tau_fail)
            sds.append(float(np.std(self._base_load(lf, self.derate_factor))))
        self.fresh_clock = np.sort(np.array(totals))
        self.load_sd = float(np.median(sds)) if sds else 0.05

    # ------------------------------------------------------------------------- the queries
    def _counterfactual(
        self, life: LifeRecord, derate_from_day: float, load_cap: float, beta: float
    ) -> float:
        ab = self.abduct(life, beta)
        base = self._base_load(life, self.derate_factor)
        n_total = base.size + int(MAX_DAY)
        use_beta = 0.0 if self.misspecified else beta
        derate_from = derate_from_day if load_cap < 1.0 else None
        rho = self._rho(base, use_beta, n_total, derate_from, load_cap)
        cum = np.concatenate([[0.0], np.cumsum(rho)])
        if ab.tau_fail >= cum[-1]:
            return MAX_DAY
        m = int(np.searchsorted(cum, ab.tau_fail, side="left")) - 1
        m = max(m, 0)
        quarters = DAY_Q * m + math.ceil((ab.tau_fail - cum[m]) * DAY_Q / rho[m] - 1e-7)
        return float(min(quarters / DAY_Q, MAX_DAY))

    def counterfactual_failure_day(
        self, life: LifeRecord, derate_from_day: float, load_cap: float
    ) -> float:
        """The day this life would have failed had the load been ``load_cap`` times its base load
        from ``derate_from_day`` on (``load_cap = 1``: nobody acts). Needs ``fit`` first."""
        self._need_fitted()
        return self._counterfactual(life, derate_from_day, load_cap, self.beta)

    def failure_sd(self, life: LifeRecord, derate_from_day: float, load_cap: float) -> float:
        """Standard deviation of the counterfactual failure day: the sampling error of ``beta`` and
        the load beyond the record, which the record cannot show."""
        self._need_fitted()
        t = self._counterfactual(life, derate_from_day, load_cap, self.beta)
        h = 0.05
        dt = (
            self._counterfactual(life, derate_from_day, load_cap, self.beta + h)
            - self._counterfactual(life, derate_from_day, load_cap, max(self.beta - h, 0.0))
        ) / (2.0 * h)
        unseen = max(t - life.days.size, 0.0)
        s_load = self.beta * self.load_sd * unseen
        return float(math.sqrt((dt * self.beta_se) ** 2 + s_load**2 + (1.0 / DAY_Q) ** 2))

    def p_fail_before(
        self, life: LifeRecord, replace_day: float | None, horizon_day: float
    ) -> float:
        """Probability of an uncontrolled failure before ``horizon_day`` if it is replaced
        on ``replace_day`` (``None``: nobody acts). A replacement installs a fresh component whose
        lifetime is drawn from the training lives' abducted clock totals."""
        self._need_fitted()
        t_hat = self.counterfactual_failure_day(life, 0.0, 1.0)
        sd = self.failure_sd(life, 0.0, 1.0)
        if replace_day is None:
            return float(norm_cdf((horizon_day - t_hat) / sd))
        before = float(norm_cdf((min(replace_day, horizon_day) - t_hat) / sd))
        survive = 1.0 - float(norm_cdf((replace_day - t_hat) / sd))
        rho_bar = math.exp(
            self.beta * (float(np.mean(self._base_load(life, self.derate_factor))) - 1.0)
        )
        if self.misspecified:
            rho_bar = 1.0
        new_by = float(np.mean(self.fresh_clock / rho_bar <= horizon_day - replace_day))
        return float(before + survive * new_by)

    def _need_fitted(self) -> None:
        if not self.fitted:
            raise RuntimeError("call fit() first")
