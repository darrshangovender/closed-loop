"""The streaming RUL estimator of the SYNTHETIC F1 plant (packet P4a).

Interface text: docs/adr/0002-interfaces.md, section 3.6; design: docs/adr/0001-architecture.md,
section 2 ("Observation chain") and 6.1.

``TheilSenEstimator`` mirrors the structure of ``asset_health.rul`` without copying it and without
importing it (0002 rule 4): a Theil-Sen slope of the health index over the last 8 daily points
(at least 5 of them present), extrapolated to the failure level and capped at the horizon. It is
fast (a few microseconds per update, standard library and numpy only) and strictly causal: the
output on day ``d`` depends on the readings up to ``d`` and on nothing else.

Four statuses:

``estimated``      a trend extrapolation, with a scale from the delta method on the regression
                   standard errors (the daily noise is ``PlantParams.hi_noise_sd``);
``prior``          the blind window after a life starts (``blind_days``, 21 d baseline plus 48 h
                   settle): the class prior is reported;
``insufficient``   fewer than ``min_points`` readings in the window: the class prior;
``not_degrading``  the fitted level is below the degradation threshold ``hi_alert`` or the slope is
                   below ``min_slope`` per day: the class prior.

The class prior is a Weibull law of the total life of the asset class, conditioned on the age of the
current life (0001 section 6.1). Its parameters are fitted, with no randomness, to the life law the
catalogue and ``PlantParams.renewal_onset_days`` imply (onset offset plus duration, mode-weighted);
the median and the 10-90 spread of the remaining life give ``m_days`` and ``sigma_days``. The
estimator knows the plant's structure, as the planner does; that is a stated limitation
(0001 section 15), not a leak: it never reads truth.

Blind days are not stored as readings: after a replacement the first estimate needs ``min_points``
fresh readings once the window has passed, as a pipeline that has to re-learn its baseline would.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import numpy as np

from closed_loop.plant.params import ASSET_CLASS, MODES_BY_CLASS, PlantParams

__all__ = [
    "STATUS_ESTIMATED",
    "STATUS_INSUFFICIENT",
    "STATUS_NOT_DEGRADING",
    "STATUS_PRIOR",
    "ClassPrior",
    "EstimatorOutput",
    "StreamingEstimator",
    "TheilSenEstimator",
    "Trend",
    "class_prior",
    "theil_sen",
]

STATUS_ESTIMATED = "estimated"
STATUS_PRIOR = "prior"  # blind window
STATUS_INSUFFICIENT = "insufficient"
STATUS_NOT_DEGRADING = "not_degrading"

_Z90 = 1.2815515655446004  # standard normal 90% point, to turn a 10-90 spread into a scale


@dataclass(frozen=True)
class EstimatorOutput:
    asset_id: str
    day: int
    status: str  # "estimated" | "prior" | "not_degrading" | "insufficient"
    m_days: float
    sigma_days: float  # prior values when status != "estimated"
    hi: float | None


class StreamingEstimator(Protocol):
    def update(
        self, asset_id: str, day: int, hi: float | None, life_id: int
    ) -> EstimatorOutput: ...

    def reset_life(self, asset_id: str, day: int) -> None: ...  # blind window starts


# ----------------------------------------------------------------------------- the class prior


@dataclass(frozen=True)
class ClassPrior:
    """Weibull law ``P(W > w) = exp(-(w / scale)^shape)`` of the total life of an asset class."""

    shape: float
    scale: float

    def remaining_quantile(self, age: float, p: float) -> float:
        """The ``p`` quantile of the remaining life of a component of this age that still runs."""
        if not 0.0 < p < 1.0:
            raise ValueError(f"p must be in (0, 1), got {p!r}")
        if age < 0.0:
            raise ValueError(f"age must be non-negative, got {age!r}")
        z = (age / self.scale) ** self.shape - math.log1p(-p)
        return self.scale * z ** (1.0 / self.shape) - age

    def point_and_scale(self, age: float) -> tuple[float, float]:
        """Median remaining life and its scale ``(q90 - q10) / (2 z90)``, at least 1e-6."""
        q10 = self.remaining_quantile(age, 0.10)
        q50 = self.remaining_quantile(age, 0.50)
        q90 = self.remaining_quantile(age, 0.90)
        return q50, max((q90 - q10) / (2.0 * _Z90), 1e-6)


@cache
def class_prior(asset_class: str, onset_days: tuple[float, float]) -> ClassPrior:
    """Weibull fitted to the total-life law of ``asset_class`` (no randomness).

    The law is ``onset offset + duration`` with the offset uniform on ``onset_days``
    (``PlantParams.renewal_onset_days``) and the duration uniform on the failure mode's catalogue
    range, mixed by the mode weights. Its distribution function is computed on a fine grid by
    convolving the two uniform densities; the Weibull comes from least squares of
    ``ln(-ln(1 - F))`` on ``ln(w)`` over the 2% to 98% quantiles.
    """
    modes = MODES_BY_CLASS.get(asset_class)
    if not modes:
        raise ValueError(f"unknown asset class {asset_class!r}")
    step = 0.05
    top = onset_days[1] + max(m.duration_days[1] for m in modes) + 5.0
    n = int(top / step) + 1

    def uniform_density(lo: float, hi: float) -> np.ndarray:
        d = np.zeros(n)
        i0, i1 = int(round(lo / step)), int(round(hi / step))
        d[i0 : i1 + 1] = 1.0 / (hi - lo)
        return d

    onset = uniform_density(*onset_days)
    pdf = np.zeros(n)
    total = sum(m.weight for m in modes)
    for m in modes:
        dur = uniform_density(*m.duration_days)
        pdf += (m.weight / total) * np.convolve(onset, dur)[:n] * step
    cdf = np.cumsum(pdf) * step
    cdf /= cdf[-1]
    grid = np.arange(n) * step
    ps = np.linspace(0.02, 0.98, 49)
    w = np.interp(ps, cdf, grid)
    x, y = np.log(w), np.log(-np.log1p(-ps))
    shape, intercept = np.polyfit(x, y, 1)
    return ClassPrior(shape=float(shape), scale=float(math.exp(-intercept / shape)))


# ----------------------------------------------------------------------------- Theil-Sen


def theil_sen(days: list[int], values: list[float]) -> tuple[float, float]:
    """Theil-Sen ``(slope, level at the last day)``: median pairwise slope and median intercept.

    ``days`` must be strictly increasing and have at least two entries.
    """
    n = len(days)
    if n < 2 or n != len(values):
        raise ValueError("theil_sen needs at least two points and equal-length inputs")
    slopes: list[float] = []
    for i in range(n - 1):
        di, vi = days[i], values[i]
        for j in range(i + 1, n):
            dd = days[j] - di
            if dd <= 0:
                raise ValueError("days must be strictly increasing")
            slopes.append((values[j] - vi) / dd)
    slope = _median(slopes)
    last = days[-1]
    level = _median([values[i] + slope * (last - days[i]) for i in range(n)])
    return slope, level


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    k = len(xs)
    mid = k // 2
    return xs[mid] if k % 2 else 0.5 * (xs[mid - 1] + xs[mid])


@dataclass(frozen=True)
class Trend:
    """The fitted trend of one asset after a day's update, for forecasting the next readings."""

    day: int
    level: float  # fitted health index at ``day``
    slope: float  # per day
    n: int  # readings in the window
    sxx: float  # sum of squared day deviations of the window
    dbar: float  # mean day of the window
    noise_sd: float  # daily noise of the health index (PlantParams.hi_noise_sd)

    def forecast(self, ahead: int = 1) -> float:
        return self.level + self.slope * ahead

    def predictive_sd(self, ahead: int = 1) -> float:
        """Standard deviation of the forecast error of a new reading ``ahead`` days on."""
        far = self.day + ahead - self.dbar
        return self.noise_sd * math.sqrt(1.0 + 1.0 / self.n + far * far / self.sxx)


# ----------------------------------------------------------------------------- the estimator


class _AssetState:
    __slots__ = ("days", "last_day", "life_id", "life_start", "trend", "values")

    def __init__(self, life_start: float) -> None:
        self.life_id: int | None = None
        self.life_start = life_start
        self.days: deque[int] = deque()
        self.values: deque[float] = deque()
        self.last_day: int | None = None
        self.trend: Trend | None = None


class TheilSenEstimator:
    """Theil-Sen trend of the health index extrapolated to the failure level.

    ``window_days`` calendar days, at least ``min_points`` readings in them, slope at least
    ``min_slope`` per day, level at least ``hi_alert``; failure level ``hi_fail`` and daily noise
    ``hi_noise_sd`` from ``PlantParams``. The point is capped at ``horizon_cap_days`` and the scale
    never falls below ``sigma_floor_days``.

    A new ``life_id`` for an asset is an implicit ``reset_life`` on that day. Days must increase
    strictly within a life. ``hi`` may be ``None`` or NaN (a masked day).
    """

    def __init__(
        self,
        params: PlantParams | None = None,
        *,
        window_days: int = 8,
        min_points: int = 5,
        min_slope: float = 0.05,
        horizon_cap_days: float = 90.0,
        sigma_floor_days: float = 1.0,
        class_of: Mapping[str, str] | None = None,
    ) -> None:
        self.params = params if params is not None else PlantParams()
        if window_days < 2 or not 2 <= min_points <= window_days:
            raise ValueError("need window_days >= 2 and 2 <= min_points <= window_days")
        if min_slope <= 0.0 or horizon_cap_days <= 0.0 or sigma_floor_days <= 0.0:
            raise ValueError("min_slope, horizon_cap_days and sigma_floor_days must be positive")
        self.window_days = window_days
        self.min_points = min_points
        self.min_slope = min_slope
        self.horizon_cap_days = horizon_cap_days
        self.sigma_floor_days = sigma_floor_days
        self._class_of: Mapping[str, str] = class_of if class_of is not None else ASSET_CLASS
        self._state: dict[str, _AssetState] = {}

    # -------------------------------------------------------------------- state
    def _asset(self, asset_id: str) -> _AssetState:
        st = self._state.get(asset_id)
        if st is None:
            if asset_id not in self._class_of:
                raise ValueError(f"unknown asset {asset_id!r}: pass class_of to name its class")
            st = self._state[asset_id] = _AssetState(life_start=0.0)
        return st

    def reset_life(self, asset_id: str, day: int) -> None:
        """A new component starts on ``day``: the blind window begins, the readings are dropped."""
        _check_day(day)
        st = self._asset(asset_id)
        st.life_start = float(day)
        st.life_id = None  # adopted from the next update
        st.days.clear()
        st.values.clear()
        st.last_day = None
        st.trend = None

    def life_start_day(self, asset_id: str) -> float:
        return self._asset(asset_id).life_start

    def trend(self, asset_id: str) -> Trend | None:
        """The trend fitted on the latest update, or ``None`` if there was none (blind, too few)."""
        return self._asset(asset_id).trend

    # -------------------------------------------------------------------- update
    def update(self, asset_id: str, day: int, hi: float | None, life_id: int) -> EstimatorOutput:
        _check_day(day)
        value: float | None = None
        if hi is not None:
            value = float(hi)
            if math.isnan(value):
                value = None
            elif math.isinf(value):
                raise ValueError("hi must be finite, None or NaN")
        st = self._asset(asset_id)
        if st.life_id is None:
            st.life_id = life_id
        elif st.life_id != life_id:
            self.reset_life(asset_id, day)
            st.life_id = life_id
        if st.last_day is not None and day <= st.last_day:
            raise ValueError(
                f"{asset_id}: day {day} does not follow day {st.last_day}; updates must be causal"
            )
        st.last_day = day
        p = self.params
        age = day - st.life_start
        if age < p.blind_days:
            st.trend = None
            return self._prior_output(asset_id, day, STATUS_PRIOR, age, value)
        if value is not None:
            st.days.append(day)
            st.values.append(value)
        while st.days and st.days[0] <= day - self.window_days:
            st.days.popleft()
            st.values.popleft()
        n = len(st.days)
        if n < self.min_points:
            st.trend = None
            return self._prior_output(asset_id, day, STATUS_INSUFFICIENT, age, value)
        days = list(st.days)
        slope, level = theil_sen(days, list(st.values))
        dbar = sum(days) / n
        sxx = sum((d - dbar) ** 2 for d in days)
        st.trend = Trend(day, level, slope, n, sxx, dbar, p.hi_noise_sd)
        if level < p.hi_alert or slope < self.min_slope:
            return self._prior_output(asset_id, day, STATUS_NOT_DEGRADING, age, value)
        m_raw = (p.hi_fail - level) / slope
        se_slope = p.hi_noise_sd / math.sqrt(sxx)
        se_level = p.hi_noise_sd * math.sqrt(1.0 / n + (day - dbar) ** 2 / sxx)
        sigma = math.hypot(max(m_raw, 0.0) * se_slope / slope, se_level / slope)
        return EstimatorOutput(
            asset_id=asset_id,
            day=day,
            status=STATUS_ESTIMATED,
            m_days=min(max(m_raw, 0.0), self.horizon_cap_days),
            sigma_days=max(sigma, self.sigma_floor_days),
            hi=value,
        )

    def _prior_output(
        self, asset_id: str, day: int, status: str, age: float, hi: float | None
    ) -> EstimatorOutput:
        prior = class_prior(self._class_of[asset_id], self.params.renewal_onset_days)
        m, sigma = prior.point_and_scale(age)
        return EstimatorOutput(
            asset_id=asset_id,
            day=day,
            status=status,
            m_days=m,
            sigma_days=max(sigma, self.sigma_floor_days),
            hi=hi,
        )


def _check_day(day: int) -> None:
    if isinstance(day, bool) or not isinstance(day, int | np.integer) or day < 0:
        raise ValueError(f"day must be a non-negative int, got {day!r}")
