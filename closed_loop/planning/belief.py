"""Vectorised belief over particles, one set per asset (packet P8a).

``BeliefTracker`` keeps, for every asset, a weighted particle set from a ``WorldModel`` and updates
it once a day from what the controller can see: the day number, the health-index reading (or
``None`` when the gate masked it), whether the reading came from an inspection, and the observed
load. **It reads observations only.** Nothing in this module or in ``worldmodel.py`` imports the
plant, ``TruthView`` or a ``Trace``; ``tests/planning/test_belief_truth_free.py`` checks the
imports,
the names used and the signatures, and replays an episode with the plant's truth poisoned.

One update, for an asset whose belief stands at day ``last``:

1. propagate the particles ``day - last`` days with ``model.step`` (continue), using the observed
   load for the final day when the model accepts one;
2. if the day is past the blind window and ``hi`` is not ``None``, add ``model.loglik`` to the log
   weights (an inspection reading uses ``inspect_noise_scale``, a quarter of the noise by default)
   and normalise;
3. if the effective sample size is below ``ess_fraction * n``, resample systematically and let the
   model jitter the result (``model.resample``; Liu-West shrinkage for the plant model).

*Blind window.* For ``blind_days`` after a life starts (23: 21 days of baseline plus 48 hours of
settle, PlantParams) the asset-health pipeline reports nothing. The tracker mirrors that so F1 and
F2 agree: it propagates the particles but leaves their weights alone, so the belief at the end of
the window is the prior carried forward by the dynamics. An inspection reading, taken by a person,
is not blind and is used. A missing reading (``hi is None``) propagates and does not reweight.

*Determinism.* Every random draw comes from ``rng_for(seed, "belief", asset_id, life, day)`` (the
prior from ``"belief_prior"``), so the result depends on the seed and the observation sequence of
that asset alone, not on the order in which assets are updated.

*Degeneracy.* If an observation is impossible for every particle (all log weights non-finite), the
tracker redraws a first-component prior, weights it by the observation, and counts the event in
``degenerate_events``; if that fails too the weights are made uniform. This also recovers a belief
that a silent replacement has made wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.planning.worldmodel import (
    ACTION_CONTINUE,
    Particles,
    WorldModel,
    failure_days,
    weighted_quantile,
)

ModelSource = WorldModel | Mapping[str, WorldModel] | Callable[[str], WorldModel]
_TINY = 1e-300  # a zero weight is kept at this floor in log space; the likelihood decides the rest


@dataclass
class _Track:
    model: WorldModel
    accepts_load: bool
    particles: Particles
    last_day: int
    life_start_day: int
    life: int
    same_day: int = 0  # readings already folded in on last_day
    n_updates: int = 0
    n_resamples: int = 0
    ess: float = 0.0
    load_ema: float | None = None
    last_load: float | None = None  # load observed at the previous update, drives the next step


def systematic_resample(weights: NDArray[np.float64], rng: np.random.Generator) -> NDArray[np.intp]:
    """Systematic resampling: one uniform draw, ``n`` evenly spaced positions on the weight CDF."""
    n = weights.size
    positions = (rng.random() + np.arange(n)) / n
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    return np.minimum(np.searchsorted(cdf, positions, side="left"), n - 1)


class BeliefTracker:
    """Per-asset particle beliefs updated from observations only (see the module docstring)."""

    def __init__(
        self,
        model: ModelSource,
        *,
        n_particles: int = 128,
        seed: int = 0,
        ess_fraction: float = 0.5,
        shrink_delta: float = 0.95,
        blind_days: float = 23.0,
        inspect_noise_scale: float = 0.25,
        load_ema_alpha: float = 0.2,
    ) -> None:
        if n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if not 0.0 < ess_fraction <= 1.0:
            raise ValueError("ess_fraction must be in (0, 1]")
        if blind_days < 0.0 or inspect_noise_scale <= 0.0 or not 0.0 < load_ema_alpha <= 1.0:
            raise ValueError("blind_days >= 0, inspect_noise_scale > 0 and 0 < load_ema_alpha <= 1")
        self._source = model
        self.n_particles = n_particles
        self.seed = seed
        self.ess_fraction = ess_fraction
        self.shrink_delta = shrink_delta
        self.blind_days = blind_days
        self.inspect_noise_scale = inspect_noise_scale
        self.load_ema_alpha = load_ema_alpha
        self.degenerate_events = 0
        self._tracks: dict[str, _Track] = {}

    # ------------------------------------------------------------------------------- models
    def model_for(self, asset_id: str) -> WorldModel:
        src = self._source
        if isinstance(src, Mapping):
            return src[asset_id]
        if callable(src) and not isinstance(src, WorldModel):
            return src(asset_id)
        return src

    # ------------------------------------------------------------------------------- tracks
    def _track(self, asset_id: str) -> _Track:
        tr = self._tracks.get(asset_id)
        if tr is None:
            model = self.model_for(asset_id)
            rng = rng_for(self.seed, "belief_prior", asset_id, 0)
            tr = _Track(
                model,
                bool(getattr(model, "accepts_load", False)),
                model.prior(rng, self.n_particles),
                0,
                0,
                0,
            )
            tr.ess = tr.particles.ess()
            self._tracks[asset_id] = tr
        return tr

    def reset_life(self, asset_id: str, day: int) -> None:
        """A new component starts on ``day``: draw its prior and restart the blind window."""
        old = self._tracks.get(asset_id)
        life = 0 if old is None else old.life + 1
        model = self.model_for(asset_id)
        rng = rng_for(self.seed, "belief_prior", asset_id, life)
        renew = getattr(model, "renewal", None)
        particles = (
            renew(rng, self.n_particles)
            if renew is not None
            else model.prior(rng, self.n_particles)
        )
        tr = _Track(model, bool(getattr(model, "accepts_load", False)), particles, day, day, life)
        tr.ess = particles.ess()
        self._tracks[asset_id] = tr

    # ------------------------------------------------------------------------------- update
    def update(
        self,
        asset_id: str,
        day: int,
        hi: float | None,
        inspected: bool = False,
        *,
        load: float | None = None,
    ) -> None:
        """Move the belief of ``asset_id`` to ``day`` and fold in the reading ``hi``.

        ``day`` may repeat (a second reading on the same day, an inspection) but may not go back.
        ``hi is None`` is a masked day. ``load`` is the observed load of the day, if any.
        """
        tr = self._track(asset_id)
        if day < tr.last_day:
            raise ValueError(f"day {day} is before the belief's day {tr.last_day} for {asset_id}")
        model = tr.model
        accepts_load = tr.accepts_load
        p = tr.particles
        # One keyed generator per day stepped, and the last one also serves the weighting and the
        # resampling of this reading. A reading is taken at the start of its day, so the clock
        # between the previous reading and this one ran under the load seen at the previous one.
        steps = day - tr.last_day
        rng = None
        for d in range(tr.last_day + 1, day + 1):
            rng = rng_for(self.seed, "belief", asset_id, tr.life, d, 0)
            if accepts_load and tr.last_load is not None:
                p = model.step(p, ACTION_CONTINUE, rng, load=tr.last_load)  # type: ignore[call-arg]
            else:
                p = model.step(p, ACTION_CONTINUE, rng)
        if steps > 0:
            tr.same_day = 0
        else:
            tr.same_day += 1
            rng = rng_for(self.seed, "belief", asset_id, tr.life, day, tr.same_day)
        assert rng is not None
        if steps > 0 or tr.n_updates == 0:
            tr.last_load = load if load is not None else tr.last_load
        tr.last_day = day
        tr.n_updates += 1
        if load is not None:
            a = self.load_ema_alpha
            tr.load_ema = load if tr.load_ema is None else (1.0 - a) * tr.load_ema + a * load
        blind = day < tr.life_start_day + self.blind_days
        if hi is None or (blind and not inspected):
            tr.particles = p
            tr.ess = p.ess()
            return
        scale = self.inspect_noise_scale if inspected else 1.0
        if accepts_load:
            ll = model.loglik(p, hi, scale, load=1.0 if load is None else load)  # type: ignore[call-arg]
        else:
            ll = model.loglik(p, hi, scale)
        p = self._reweight(model, p, ll, hi, scale, rng)
        ess = p.ess()
        if ess < self.ess_fraction * p.n:
            idx = systematic_resample(p.weights, rng)
            resample = getattr(model, "resample", None)
            p = resample(p, idx, rng, self.shrink_delta) if resample is not None else p.take(idx)
            tr.n_resamples += 1
            ess = p.ess()
        tr.particles = p
        tr.ess = ess

    def _reweight(
        self,
        model: WorldModel,
        p: Particles,
        ll: NDArray[np.float64],
        hi: float,
        scale: float,
        rng: np.random.Generator,
    ) -> Particles:
        logw = np.log(np.maximum(p.weights, _TINY)) + ll
        w = _normalise(logw)
        if w is not None:
            return p.with_weights(w, trusted=True)
        # Every particle rules the reading out: start again from the first-component prior.
        self.degenerate_events += 1
        fresh = model.prior(rng, p.n)
        w = _normalise(model.loglik(fresh, hi, scale))
        return fresh.with_weights(w if w is not None else np.full(p.n, 1.0 / p.n))

    # ------------------------------------------------------------------------------- queries
    def particles(self, asset_id: str) -> Particles:
        return self._track(asset_id).particles

    def day(self, asset_id: str) -> int:
        return self._track(asset_id).last_day

    def ess(self, asset_id: str) -> float:
        return self._track(asset_id).ess

    def n_resamples(self, asset_id: str) -> int:
        return self._track(asset_id).n_resamples

    def life(self, asset_id: str) -> int:
        return self._track(asset_id).life

    def load_forecast(self, asset_id: str) -> float | None:
        """Moving average of the observed load, for forward rollouts; ``None`` if none seen."""
        return self._track(asset_id).load_ema

    def remaining_life_days(self, asset_id: str) -> NDArray[np.float64]:
        """Per-particle days to failure at nominal load, for models that provide them."""
        tr = self._track(asset_id)
        fn = getattr(tr.model, "remaining_life_days", None)
        if fn is None:
            raise NotImplementedError(f"{type(tr.model).__name__} has no remaining_life_days")
        if tr.accepts_load and tr.load_ema is not None:
            return fn(tr.particles, load=tr.load_ema)
        return fn(tr.particles)

    def remaining_life_quantiles(
        self, asset_id: str, qs: tuple[float, ...] = (0.05, 0.5, 0.95)
    ) -> NDArray[np.float64]:
        p = self.particles(asset_id)
        return weighted_quantile(self.remaining_life_days(asset_id), p.weights, np.array(qs))

    def failure_day_quantiles(
        self,
        asset_id: str,
        qs: tuple[float, ...] = (0.05, 0.5, 0.95),
        *,
        max_days: int = 300,
    ) -> NDArray[np.float64]:
        """Quantiles of the days to failure under ``continue``, from a rollout of the model.

        The rollout uses the model's stochastic dynamics around the forecast load, so unlike
        ``remaining_life_quantiles`` it includes the load and rate uncertainty of the future. It
        is deterministic: the generator is keyed by the asset, the life and the day. Particles
        that survive ``max_days`` count as ``max_days + 1``.
        """
        tr = self._track(asset_id)
        rng = rng_for(self.seed, "belief_rollout", asset_id, tr.life, tr.last_day)
        kwargs = {"load_mean": tr.load_ema} if tr.accepts_load and tr.load_ema is not None else {}
        days = failure_days(tr.model, tr.particles, rng, max_days, **kwargs)
        days = np.where(np.isinf(days), max_days + 1.0, days)
        return weighted_quantile(days, tr.particles.weights, np.array(qs))

    def p_failed(self, asset_id: str) -> float:
        """Posterior probability that the asset has already failed."""
        tr = self._track(asset_id)
        return float(np.dot(tr.particles.weights, tr.model.failed(tr.particles)))


def _normalise(logw: NDArray[np.float64]) -> NDArray[np.float64] | None:
    """Normalised weights from log weights, or ``None`` when none is finite."""
    top = float(np.max(logw))
    if not np.isfinite(top):
        return None
    w = np.exp(logw - top)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return None
    return w / total
