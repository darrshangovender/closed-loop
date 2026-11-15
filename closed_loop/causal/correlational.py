"""The correlation-only models: an effect model (ridge) and a multinomial logistic regression.

*Effect model.* ``CorrelationalEffectModel`` is what an analyst without a causal model would fit to
the same observational lives (0001 section 7: "ridge on HI, derate flag, interactions"). At each
life's reference day, its first alert, it takes the health index, the slope of the last week, the
mean load, a flag "this life was ever derated", the delay of the derate after the reference day,
and the flag's interactions with the index and the slope, and regresses the remaining life (to the
failure, or to the end of observation when the life was not seen to fail) on them by ridge
regression. It knows nothing of the clock, of what a derate does to the load, or of why a life was
derated. The flag is the usual one a practitioner would build, "was derated at some point", and it
carries two biases a causal model avoids: the operator derates when the degradation shows, and a
life that fails quickly has less time to be derated at all.

Its counterfactual keeps the individual's residual (an additive-noise model):
``T_cf = T_obs + f(x, a_cf, delay_cf) - f(x, a_obs, delay_obs)``, the best use of the factual
outcome a model without a mechanism can make. It cannot represent a retained load other than the
one in the training data, so every ``load_cap`` below 1 is the flag, and ``load_cap = 1`` is
no flag.

*Logistic regression.* ``SoftmaxRegression`` is multinomial logistic regression written in numpy
(analytic gradient, L-BFGS from scipy to minimise). The correlational diagnoser uses it.

Everything is SYNTHETIC.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop.causal.records import LifeRecord
from closed_loop.causal.scm import DAY_Q, MAX_DAY, norm_cdf

N_FEATURES = 7


@dataclass(frozen=True)
class RefPoint:
    """A life seen at its reference day (first alert)."""

    t_ref: int
    hi: float
    slope: float
    load: float
    flag: float  # 1.0 if the life was ever derated
    delay: float  # days from the reference day to the derate; 0 when not derated
    outcome: float  # days from the reference day to the failure (or to the end of observation)


def reference_point(life: LifeRecord, hi_alert: float, blind_days: float) -> RefPoint | None:
    """The features of ``life`` at its first alert, or ``None`` if it never alerted."""
    t_ref = life.first_alert_day(hi_alert, blind_days)
    if t_ref is None or not life.days.size:
        return None
    last = (life.hi_days <= t_ref) & (life.hi_days >= t_ref - 6)
    days, hi = life.hi_days[last], life.hi[last]
    slope = float(np.polyfit(days, hi, 1)[0]) if days.size >= 2 else 0.0
    hi_ref = float(hi[days == t_ref][0])
    window = (life.days <= t_ref) & (life.days >= t_ref - 6)
    flag = 0.0 if life.derate_start is None else 1.0
    delay = 0.0 if life.derate_start is None else max(life.derate_start - t_ref, 0.0)
    return RefPoint(
        t_ref, hi_ref, slope, float(life.load[window].mean()), flag, delay, life.end_day - t_ref
    )


def design_row(pt: RefPoint, flag: float, delay: float) -> NDArray[np.float64]:
    """``[hi, slope, load, flag, flag * delay, flag * hi, flag * slope]``."""
    return np.array(
        [pt.hi, pt.slope, pt.load, flag, flag * delay, flag * pt.hi, flag * pt.slope], dtype=float
    )


class CorrelationalEffectModel:
    """Ridge regression of the remaining life on alert-day features and a derate flag."""

    name = "correlational"

    def __init__(
        self, *, ridge: float = 1.0, hi_alert: float = 2.0, blind_days: float = 23.0
    ) -> None:
        if ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        self.ridge = ridge
        self.hi_alert = hi_alert
        self.blind_days = blind_days
        self.fitted = False
        self.fresh_days: NDArray[np.float64] = np.zeros(0)

    # --------------------------------------------------------------------------- fitting
    def fit(self, lives: Sequence[LifeRecord]) -> None:
        pts = [p for lf in lives if (p := reference_point(lf, self.hi_alert, self.blind_days))]
        if len(pts) < N_FEATURES + 2:
            raise ValueError(f"need at least {N_FEATURES + 2} lives with an alert, got {len(pts)}")
        X = np.vstack([design_row(p, p.flag, p.delay) for p in pts])
        y = np.array([p.outcome for p in pts])
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0)
        self.sd = np.where(self.sd > 0.0, self.sd, 1.0)
        Z = (X - self.mu) / self.sd
        self.y_mean = float(y.mean())
        gram = Z.T @ Z + self.ridge * np.eye(N_FEATURES)
        self._gram_inv = np.linalg.inv(gram)
        self.coef = self._gram_inv @ Z.T @ (y - self.y_mean)
        resid = y - (self.y_mean + Z @ self.coef)
        self.sigma = float(math.sqrt(resid @ resid / max(len(y) - N_FEATURES - 1, 1)))
        self.fresh_days = np.sort(
            np.array([lf.fail_day for lf in lives if lf.fail_day is not None], dtype=float)
        )
        self.n_train = len(pts)
        self.fitted = True

    def predict_row(self, pt: RefPoint, flag: float, delay: float) -> float:
        z = (design_row(pt, flag, delay) - self.mu) / self.sd
        return float(self.y_mean + z @ self.coef)

    # -------------------------------------------------------------------------- queries
    def _adjustment(
        self, life: LifeRecord, derate_from_day: float, load_cap: float
    ) -> tuple[RefPoint, float, float]:
        """The reference point, the counterfactual shift of the outcome and its standard error."""
        if not self.fitted:
            raise RuntimeError("call fit() first")
        pt = reference_point(life, self.hi_alert, self.blind_days)
        if pt is None:
            raise ValueError("this life never alerted, so it has no reference day")
        flag_cf = 1.0 if load_cap < 1.0 else 0.0
        delay_cf = max(derate_from_day - pt.t_ref, 0.0) if flag_cf else 0.0
        d_row = (design_row(pt, flag_cf, delay_cf) - design_row(pt, pt.flag, pt.delay)) / self.sd
        shift = float(d_row @ self.coef)
        se = float(self.sigma * math.sqrt(d_row @ self._gram_inv @ d_row))
        return pt, shift, se

    def counterfactual_failure_day(
        self, life: LifeRecord, derate_from_day: float, load_cap: float
    ) -> float:
        pt, shift, _ = self._adjustment(life, derate_from_day, load_cap)
        return float(min(life.end_day + shift, MAX_DAY))

    def p_fail_before(
        self, life: LifeRecord, replace_day: float | None, horizon_day: float
    ) -> float:
        """Probability of a failure before ``horizon_day``; a replacement installs a component whose
        lifetime is drawn from the failure days of the training lives."""
        pt, shift, se = self._adjustment(life, 0.0, 1.0)
        t_hat = life.end_day + shift
        sd = max(se, 1.0 / DAY_Q)
        if replace_day is None:
            return float(norm_cdf((horizon_day - t_hat) / sd))
        before = float(norm_cdf((min(replace_day, horizon_day) - t_hat) / sd))
        survive = 1.0 - float(norm_cdf((replace_day - t_hat) / sd))
        new_by = float(np.mean(self.fresh_days <= horizon_day - replace_day))
        return float(before + survive * new_by)


# ----------------------------------------------------------------------- multinomial logistic
class SoftmaxRegression:
    """Multinomial logistic regression with an L2 penalty on the weights (not the intercepts).

    ``fit(X, y)`` takes features ``(n, p)`` and integer labels in ``0 .. n_classes - 1``;
    features are standardised inside. ``predict_proba`` returns ``(n, n_classes)``. A class that
    never occurs in training keeps a probability near zero.
    """

    def __init__(self, n_classes: int, *, l2: float = 1.0, max_iter: int = 400) -> None:
        if n_classes < 2:
            raise ValueError("need at least two classes")
        self.n_classes = n_classes
        self.l2 = l2
        self.max_iter = max_iter
        self.fitted = False

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> SoftmaxRegression:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or y.shape != (X.shape[0],):
            raise ValueError("X must be (n, p) and y (n,)")
        if np.min(y) < 0 or np.max(y) >= self.n_classes:
            raise ValueError("labels out of range")
        n, p = X.shape
        self.mu = X.mean(axis=0)
        self.sd = np.where(X.std(axis=0) > 0.0, X.std(axis=0), 1.0)
        from scipy.optimize import minimize  # type: ignore[import-untyped]  # heavy import

        Z = (X - self.mu) / self.sd
        onehot = np.zeros((n, self.n_classes))
        onehot[np.arange(n), y] = 1.0
        k = self.n_classes

        def unpack(theta: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
            return theta[: p * k].reshape(p, k), theta[p * k :]

        def objective(theta: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            W, b = unpack(theta)
            logits = Z @ W + b
            logits -= logits.max(axis=1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
            loss = -float((onehot * logp).sum()) / n + 0.5 * self.l2 * float((W * W).sum()) / n
            prob = np.exp(logp)
            gw = Z.T @ (prob - onehot) / n + self.l2 * W / n
            gb = (prob - onehot).sum(axis=0) / n
            return loss, np.concatenate([gw.ravel(), gb])

        theta0 = np.zeros(p * k + k)
        sol = minimize(
            objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": self.max_iter}
        )
        self.W, self.b = unpack(sol.x)
        self.converged = bool(sol.success)
        self.fitted = True
        return self

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.fitted:
            raise RuntimeError("call fit() first")
        Z = (np.atleast_2d(np.asarray(X, dtype=float)) - self.mu) / self.sd
        logits = Z @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)
