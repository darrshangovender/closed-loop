"""The coverage monitor: an e-process on the perception contract (packet P4c). SYNTHETIC.

Interface text: docs/adr/0002-interfaces.md, section 3.6, as amended by docs/adr/0003 A6;
design: docs/adr/0001-architecture.md, section 6.4.

Four modes, one e-process (``stats.BoundedEProcess``, ``H0: E[x | past] <= p0``, ``x`` in [0, 1]):

``opt``    label modes. One value per *life*, pushed when the life's labels resolve: the fraction of
``pess``   that life's checkpoints in miss status under the mode. ``opt`` counts only ``MISS``
``inf``    (UNKNOWN and LOWER_HIT are taken as covered, so a firing is a true violation and the
           power is low); ``pess`` also counts ``UNKNOWN`` as a miss (fast, with false alarms,
           because a label that says nothing is scored as a failure); ``inf`` drops ``UNKNOWN`` from
           the life's fraction and pushes nothing for a life with no informative label.
``proxy``  no label. A one-day-ahead conformal band on the health index ``hi`` built from the
           estimator's trend, with a split quantile from development lives; every day the next
           day's reading is compared with yesterday's band and the miss indicator goes in.

The unit of the sequence is the life, not the checkpoint (docs/adr/0003 A6): the checkpoints of one
life resolve together and are strongly dependent, so a per-checkpoint Bernoulli test would break
Ville's condition. The guarantee needs the conditional-mean assumption: the per-life values are
independent of one another given the past (lives are independent) and have mean at most ``p0``
under H0. ``p0`` is the nominal miss level of the set being monitored: 0.01 for a per-step 99% set,
and ``alpha_life`` (0.05) for the per-life band, whose per-life any-miss probability is at most that
and so bounds the mean of the per-life fraction of misses.

``proxy`` mode tests whether the degradation path is leaving the calibrated model: a band on the
next day's health index, not a band on RUL. It is a proxy for the contract and nothing more. A
firing does not mean C1 failed (a process upset or a sensor fault moves ``hi`` without touching the
bin of the true RUL), and silence does not mean C1 held (a degradation that the trend extrapolates
well, a slow drift for instance, is forecast one day ahead without a miss). Its unit is a day of one
asset, and days of one life are autocorrelated, so the conditional-mean condition is an
*assumption* there and not a property: the false-alarm rate is measured on unshifted episodes and
reported, not assumed to be at most ``delta``.

On firing, the caller widens every perception set to the whole range (``shield.runtime`` does this
from ``Observation.monitor_fired``); this class only decides when.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop.perception.conformal import require_calibration_seeds
from closed_loop.perception.estimator import TheilSenEstimator, Trend
from closed_loop.perception.labels import Label, LabelStatus
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import ASSET_IDS, PlantParams
from closed_loop.plant.scenario import Scenario
from closed_loop.stats.eprocess import BoundedEProcess

__all__ = [
    "LABEL_MODES",
    "MODES",
    "PROXY_ALPHA",
    "CoverageMonitor",
    "ProxyBand",
    "life_fraction",
    "proxy_development_scores",
]

LABEL_MODES = ("opt", "pess", "inf")
MODES = (*LABEL_MODES, "proxy")
PROXY_ALPHA = 0.05  # nominal miss rate of the one-day-ahead band


def life_fraction(labels: Sequence[Label], mode: str) -> float | None:
    """The value one life pushes: the fraction of its checkpoints in miss status under ``mode``.

    ``None`` for mode ``inf`` when no label of the life is informative (all ``UNKNOWN``) or when
    there are no labels at all.
    """
    if mode not in LABEL_MODES:
        raise ValueError(f"mode must be one of {LABEL_MODES}, got {mode!r}")
    n = len(labels)
    if n == 0:
        return None
    misses = sum(lab.status is LabelStatus.MISS for lab in labels)
    unknown = sum(lab.status is LabelStatus.UNKNOWN for lab in labels)
    if mode == "opt":
        return misses / n
    if mode == "pess":
        return (misses + unknown) / n
    informative = n - unknown
    return misses / informative if informative else None


# ----------------------------------------------------------------------------- the proxy band


@dataclass(frozen=True)
class ProxyBand:
    """One-day-ahead band on the health index: ``forecast +- quantile * predictive_sd``.

    ``quantile`` is the split-conformal quantile (level ``1 - alpha``) of the development
    residuals ``|hi - forecast| / predictive_sd``, so a reading on a path that stays inside the
    calibrated model lies outside the band with probability about ``alpha``.
    """

    quantile: float
    alpha: float = PROXY_ALPHA

    @classmethod
    def fit(cls, residual_scores: NDArray[np.float64], alpha: float = PROXY_ALPHA) -> ProxyBand:
        z = np.sort(np.asarray(residual_scores, dtype=np.float64))
        if z.ndim != 1 or z.size == 0 or not np.all(np.isfinite(z)) or np.any(z < 0.0):
            raise ValueError("residual_scores must be a non-empty 1-d array of finite values >= 0")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        k = math.ceil((len(z) + 1) * (1.0 - alpha) - 1e-9)
        if k > len(z):
            raise ValueError(f"{len(z)} residuals are too few for a band at alpha {alpha}")
        return cls(quantile=float(z[k - 1]), alpha=alpha)

    def interval(self, trend: Trend) -> tuple[float, float]:
        """The band for the reading one day after ``trend.day``."""
        centre, half = trend.forecast(1), self.quantile * trend.predictive_sd(1)
        return centre - half, centre + half


def proxy_development_scores(
    params: PlantParams,
    seeds: Sequence[int],
    scenario: Scenario | None = None,
    estimator_factory: Callable[[], TheilSenEstimator] | None = None,
) -> NDArray[np.float64]:
    """Normalised one-day-ahead residuals on open-loop first lives of ``dev`` or ``calib`` seeds.

    For each asset and day with a fitted trend on the day before and a reading today (not
    masked), ``|hi - forecast| / predictive_sd``. Only days while the component is running count.
    """
    require_calibration_seeds(seeds)
    out: list[float] = []
    for seed in seeds:
        table = simulate_lives(params, seed, scenario if scenario is not None else Scenario())
        for aid in ASSET_IDS:
            est = (
                estimator_factory() if estimator_factory is not None else TheilSenEstimator(params)
            )
            last = min(table.n_days, math.ceil(float(table.failure_day[aid])))
            prev: Trend | None = None
            for d in range(last):
                h = table.hi[aid][d]
                value = None if math.isnan(h) else float(h)
                est.update(aid, d, value, 0)
                if prev is not None and prev.day == d - 1 and value is not None:
                    out.append(abs(value - prev.forecast(1)) / prev.predictive_sd(1))
                prev = est.trend(aid)
    if not out:
        raise ValueError("the seeds gave no one-day-ahead residuals")
    return np.array(out)


# ----------------------------------------------------------------------------- the monitor


class CoverageMonitor:
    """E-process on the coverage contract; see the module docstring.

    ``observe(labels)`` is for the label modes and ``observe_day(...)`` for ``proxy``. After a
    firing, ``fired`` stays set; ``fired_at_label`` is the number of labels (or, in proxy mode,
    daily indicators) seen when it first fired, ``fired_unit`` the number of e-process steps (lives
    or days) and ``fired_day`` the day of the labels that completed the evidence (or of the
    reading). ``include_prior=False`` leaves labels from prior-status checkpoints out.
    """

    def __init__(
        self,
        mode: str,
        p0: float = 0.01,
        delta: float = 0.01,
        *,
        band: ProxyBand | None = None,
        include_prior: bool = True,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if mode == "proxy":
            if band is None:
                raise ValueError("proxy mode needs a ProxyBand (see proxy_development_scores)")
            if not math.isclose(p0, band.alpha, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"p0 must be the band's nominal miss rate {band.alpha}, got {p0!r}"
                )
        elif band is not None:
            raise ValueError("only proxy mode takes a band")
        self.mode = mode
        self.p0 = float(p0)
        self.delta = float(delta)
        self.band = band
        self.include_prior = include_prior
        self._process = BoundedEProcess(p0, delta)
        self.fired = False
        self.fired_at_label: int | None = None
        self.fired_unit: int | None = None
        self.fired_day: float | None = None
        self.n_labels = 0
        self.n_lives = 0
        self.n_days = 0
        self.n_skipped_lives = 0
        self._last_resolved = -(10**9)
        self._seen: set[tuple[str, int]] = set()
        self._forecast: dict[str, tuple[int, int, float, float]] = {}  # asset -> day, life, lo, hi

    @classmethod
    def proxy(cls, band: ProxyBand, delta: float = 0.01) -> CoverageMonitor:
        """Proxy mode with ``p0`` set to the band's nominal miss rate."""
        return cls("proxy", band.alpha, delta, band=band)

    @property
    def e_value(self) -> float:
        return self._process.e_value

    @property
    def log_e_value(self) -> float:
        return self._process.log_e_value

    @property
    def n_units(self) -> int:
        """Values pushed so far: lives (label modes) or days (proxy)."""
        return self._process.n

    def _push(self, x: float, seen: int, day: float) -> None:
        self._process.update(x)
        if self._process.fired and not self.fired:
            self.fired = True
            self.fired_at_label = seen
            self.fired_unit = self._process.n
            self.fired_day = day

    # ------------------------------------------------------------------ label modes
    def observe(self, labels: Sequence[Label]) -> None:
        """Group ``labels`` by life and push one value per life (modes ``opt``, ``pess``, ``inf``).

        Labels arrive in non-decreasing ``resolved_day``. A life's labels must all be in one call
        (they resolve together); a second call naming the same life raises ``ValueError``.
        """
        if self.mode == "proxy":
            raise ValueError("proxy mode uses no labels: call observe_day")
        by_life: dict[tuple[str, int], list[Label]] = {}
        for lab in labels:
            if lab.resolved_day < self._last_resolved:
                raise ValueError(
                    f"label resolved on day {lab.resolved_day} arrives after day "
                    f"{self._last_resolved}: the monitor reads labels in time order"
                )
            self._last_resolved = lab.resolved_day
            if lab.prior and not self.include_prior:
                continue
            by_life.setdefault((lab.asset_id, lab.life_id), []).append(lab)
        for key, group in by_life.items():
            if key in self._seen:
                raise ValueError(
                    f"the labels of life {key} were already observed: send a life once"
                )
            self._seen.add(key)
            self.n_labels += len(group)
            x = life_fraction(group, self.mode)
            if x is None:
                self.n_skipped_lives += 1
                continue
            self.n_lives += 1
            self._push(x, self.n_labels, float(max(lab.resolved_day for lab in group)))

    # ------------------------------------------------------------------ proxy mode
    def observe_day(
        self, asset_id: str, day: int, hi: float | None, trend: Trend | None, life_id: int = 0
    ) -> None:
        """One asset's day, in proxy mode: score today's reading against yesterday's band, then
        form tomorrow's band from ``trend`` (the estimator's trend after today's update).

        ``hi`` is today's reading (``None`` or NaN when masked: no indicator), ``trend`` the
        estimator's trend or ``None`` while it has none (blind window, too few points), which
        leaves no band for tomorrow. A new ``life_id`` drops the stored band.
        """
        if self.mode != "proxy" or self.band is None:
            raise ValueError("observe_day is for proxy mode")
        value = None if hi is None or math.isnan(hi) else float(hi)
        stored = self._forecast.get(asset_id)
        if stored is not None and stored[0] == day and stored[1] == life_id and value is not None:
            lo, up = stored[2], stored[3]
            miss = not (lo <= value <= up)
            self.n_days += 1
            self._push(1.0 if miss else 0.0, self.n_days, float(day))
        if trend is not None and trend.day == day:
            lo, up = self.band.interval(trend)
            self._forecast[asset_id] = (day + 1, life_id, lo, up)
        else:
            self._forecast.pop(asset_id, None)
