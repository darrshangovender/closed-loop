"""World models for the belief planner (packet P8a).

A world model is a small generative model of one asset that a particle filter can run: it draws a
prior, moves particles one day forward under an action, scores an observation, and says which
particles have failed. The planner (packet P8b) and the belief tracker (``belief.py``) use only this
interface, so a different model can be dropped in. Everything here is SYNTHETIC and deterministic:
all randomness comes from the ``numpy.random.Generator`` the caller passes in (the tracker builds it
with ``rng_for``), and nothing reads a file, a key or the network.

Two models ship:

``PlantWorldModel``
    The plant's own structure (0001 section 2) with parameters *fitted on development seeds*
    (``planning/fitting.py``), deliberately not the generating parameters, plus a ``mismatch`` knob
    for a wrong wear rate (0001 section 8). Particles carry (mode, progress x, duration D, shape k,
    rate multiplier):

    * the wear clock advances ``rho = rate * exp(beta (load - 1)) * (1 - mismatch)`` per day and
      progress is ``x = (clock - onset) / D``, negative before the onset;
    * severity is ``s = clip(x, 0, 1) ** k`` and the asset has failed when ``x >= 1``;
    * **observation model**: the daily health index is
      ``hi ~ N(hi_base + (hi_fail - hi_base) s + hi_load_coef (load - 1), (hi_noise_sd * scale)^2)``
      where ``scale`` is 1 for the daily reading and ``inspect_noise_factor`` (0.25) for an
      inspection. Only ``hi`` and the observed load enter. The signature channels of 0003 A3 and
      sensor gain drift are not modelled here; the diagnoser (``closed_loop.causal.diagnoser``)
      decides whether a rise in ``hi`` is degradation.

``TabularWorldModel``
    A finite model with an explicit transition array per action and an observation confusion
    matrix. It is the tiny instance of 0001 section 8 on which P8b validates the planner against
    exact dynamic programming, and it has an exact Bayes filter (``predict``, ``posterior``) that
    the particle filter is tested against.

Actions are integers: ``ACTION_CONTINUE = 0``, ``ACTION_INSPECT = 1``, ``ACTION_REPLACE = 2``
(``ACTION_NAMES``). Continue and inspect move the state identically; they differ in the observation
the caller scores afterwards (``noise_scale``) and in cost, which is the planner's business.

Interfaces beyond the frozen text of docs/adr/0002 section 3.10, all additive: ``Particles`` (the
type the text leaves open), a ``renewal`` method (a fresh component after a replacement), the
optional ``load`` keyword on ``PlantWorldModel.step`` and ``loglik``, and ``resample``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

ACTION_CONTINUE = 0
ACTION_INSPECT = 1
ACTION_REPLACE = 2
ACTION_NAMES: tuple[str, str, str] = ("continue", "inspect", "replace")
N_ACTIONS = len(ACTION_NAMES)

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


# --------------------------------------------------------------------------------------- particles
@dataclass(frozen=True, eq=False)
class Particles:
    """A weighted particle set. The arrays are read-only: a step returns a new set.

    ``state`` maps a variable name to an array of length ``n``; ``weights`` is non-negative and sums
    to one. The arrays handed in are frozen (``writeable = False``) on construction.
    """

    state: Mapping[str, NDArray]
    weights: NDArray[np.float64]

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("weights must be a non-empty vector")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and non-negative")
        total = float(w.sum())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1, got {total}")
        if not self.state:
            raise ValueError("a particle set needs at least one state variable")
        for name, arr in self.state.items():
            if arr.shape != w.shape:
                raise ValueError(f"state {name!r} has shape {arr.shape}, expected {w.shape}")
            arr.setflags(write=False)
        w.setflags(write=False)
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "state", MappingProxyType(dict(self.state)))

    @classmethod
    def trusted(cls, state: dict[str, NDArray], weights: NDArray[np.float64]) -> Particles:
        """Build a set from arrays the caller has just computed, skipping the validation.

        For the models' own ``step`` and ``resample``, where the shapes and the weights come from
        code in this package; the validation is a measurable share of one belief update. Use the
        constructor for anything that comes from outside.
        """
        obj = object.__new__(cls)
        for arr in state.values():
            arr.setflags(write=False)
        weights.setflags(write=False)
        object.__setattr__(obj, "state", MappingProxyType(state))
        object.__setattr__(obj, "weights", weights)
        return obj

    @property
    def n(self) -> int:
        return int(self.weights.size)

    def ess(self) -> float:
        """Effective sample size ``1 / sum(w^2)``: ``n`` when uniform, 1 for a point mass."""
        return float(1.0 / np.dot(self.weights, self.weights))

    def with_weights(self, weights: NDArray[np.float64], *, trusted: bool = False) -> Particles:
        """The same particles under new weights (``trusted``: already normalised, no check)."""
        if trusted:
            return Particles.trusted(dict(self.state), weights)
        return Particles(self.state, weights)

    def take(self, idx: NDArray[np.intp]) -> Particles:
        """The particles ``idx`` (repeats allowed), uniformly weighted: a resampling."""
        n = int(idx.size)
        return Particles.trusted({k: v[idx] for k, v in self.state.items()}, np.full(n, 1.0 / n))

    def mean(self, name: str) -> float:
        return float(np.dot(self.weights, self.state[name]))

    def quantile(self, values: NDArray[np.float64], q: float | NDArray[np.float64]) -> NDArray:
        """Weighted quantile(s) of ``values`` (one per particle) under this set's weights."""
        return weighted_quantile(values, self.weights, q)


def weighted_quantile(
    values: NDArray[np.float64], weights: NDArray[np.float64], q: float | NDArray[np.float64]
) -> NDArray[np.float64]:
    """Quantile(s) ``q`` in [0, 1] of a weighted sample (interpolating the midpoint CDF)."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    w = weights[order]
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("weights sum to zero")
    cdf = (np.cumsum(w) - 0.5 * w) / total
    return np.atleast_1d(np.interp(q, cdf, v))


# ------------------------------------------------------------------------------------- the protocol
@runtime_checkable
class WorldModel(Protocol):
    """What a particle filter and a planner need from a model of one asset.

    A real model adapter (a learned simulator, say) fills these four methods; nothing else is
    required. ``obs`` is whatever the model's observation is (a health-index reading for
    ``PlantWorldModel``, a symbol index for ``TabularWorldModel``); ``None`` means no observation
    and must score zero.
    """

    def prior(self, rng: np.random.Generator, n: int) -> Particles: ...

    def step(self, p: Particles, action: int, rng: np.random.Generator) -> Particles: ...

    def loglik(self, p: Particles, obs: float | None, noise_scale: float = 1.0) -> NDArray: ...

    def failed(self, p: Particles) -> NDArray[np.bool_]: ...


@runtime_checkable
class RenewableWorldModel(WorldModel, Protocol):
    """A world model that can also draw the state of a freshly installed component."""

    def renewal(self, rng: np.random.Generator, n: int) -> Particles: ...


# ------------------------------------------------------------------------------- the plant's model
_REQUIRED_GLOBAL = (
    "hi_base",
    "hi_fail",
    "hi_noise_sd",
    "hi_load_coef",
    "beta_load",
    "horizon_days",
    "first_p_in",
    "first_onset_min",
    "first_fail_hi",
    "renew_onset_lo",
    "renew_onset_hi",
)
_OPTIONAL_GLOBAL = {"rate_sd": 0.02, "rate_walk_sd": 0.004, "load_sd": 0.05}
_MODE_FIELDS = ("weight", "dur_lo", "dur_hi", "k_lo", "k_hi")


def mode_keys(asset_class: str, fitted: Mapping[str, float]) -> tuple[str, ...]:
    """The failure-mode names the fitted mapping holds for ``asset_class``, sorted."""
    prefix = f"{asset_class}."
    names = {k[len(prefix) :].rsplit(".", 1)[0] for k in fitted if k.startswith(prefix)}
    return tuple(sorted(names))


class PlantWorldModel:
    """The plant's structure with fitted parameters, for one asset class (see the module doc).

    ``fitted`` holds floats under these keys (``planning.fitting.fit_world_model`` produces them):
    ``hi_base``, ``hi_fail``, ``hi_noise_sd``, ``hi_load_coef``, ``beta_load``; ``horizon_days``,
    ``first_p_in`` (share of first components that fail inside the horizon), ``first_onset_min``
    (earliest onset of a first life, day) and ``first_fail_hi`` (latest failure day of a first
    life); ``renew_onset_lo``, ``renew_onset_hi`` (onset offset after a replacement, clock days);
    and, for every mode of the class, ``"<CLASS>.<MODE>.<field>"`` with fields ``weight``,
    ``dur_lo``, ``dur_hi``, ``k_lo``, ``k_hi``. Optional: ``rate_sd`` (spread of the per-asset wear
    multiplier, default 0.02), ``rate_walk_sd`` (daily drift of that multiplier, 0.004) and
    ``load_sd`` (day-to-day load wander used when no load is observed, 0.05).

    ``mismatch`` is the relative error of the wear rate the model believes in: the clock runs at
    ``(1 - mismatch)`` of the fitted speed, so a positive value is an *optimistic* model (it
    underestimates wear), the harder case for a safety constraint. Negative values are allowed.
    The pre-registered ablation uses 0, 0.10 and 0.25.
    """

    name = "plant"
    n_actions = N_ACTIONS
    action_names = ACTION_NAMES
    accepts_load = True  # step and loglik take the observed load

    def __init__(
        self,
        fitted: Mapping[str, float],
        mismatch: float = 0.0,
        *,
        asset_class: str = "SLURRY_PUMP",
    ) -> None:
        if not -1.0 < mismatch < 1.0:
            raise ValueError(f"mismatch must be in (-1, 1), got {mismatch}")
        missing = [k for k in _REQUIRED_GLOBAL if k not in fitted]
        modes = mode_keys(asset_class, fitted)
        if not modes:
            missing.append(f"{asset_class}.<MODE>.<field>")
        for m in modes:
            missing += [
                f"{asset_class}.{m}.{f}"
                for f in _MODE_FIELDS
                if f"{asset_class}.{m}.{f}" not in fitted
            ]
        if missing:
            raise KeyError(f"fitted parameters missing: {', '.join(missing)}")
        g = {k: float(fitted[k]) for k in _REQUIRED_GLOBAL}
        g.update({k: float(fitted.get(k, d)) for k, d in _OPTIONAL_GLOBAL.items()})
        if g["hi_fail"] <= g["hi_base"]:
            raise ValueError("hi_fail must exceed hi_base")
        if g["hi_noise_sd"] <= 0.0:
            raise ValueError("hi_noise_sd must be positive")
        self.asset_class = asset_class
        self.mismatch = float(mismatch)
        self.modes = modes
        self._g = g
        table = np.array(
            [[float(fitted[f"{asset_class}.{m}.{f}"]) for f in _MODE_FIELDS] for m in modes]
        )
        w = table[:, 0]
        if np.any(w <= 0.0):
            raise ValueError("mode weights must be positive")
        self._cum_w = np.cumsum(w / w.sum())
        self._cum_w[-1] = 1.0
        self._dur_lo, self._dur_hi = table[:, 1], table[:, 2]
        self._k_lo, self._k_hi = table[:, 3], table[:, 4]
        if np.any(self._dur_lo <= 0.0) or np.any(self._dur_hi < self._dur_lo):
            raise ValueError("durations must satisfy 0 < dur_lo <= dur_hi")
        if np.any(self._k_lo <= 0.0) or np.any(self._k_hi < self._k_lo):
            raise ValueError("shapes must satisfy 0 < k_lo <= k_hi")
        self._speed = 1.0 - self.mismatch

    # ------------------------------------------------------------------------------- draws
    def _draw(self, rng: np.random.Generator, n: int, *, first: bool) -> Particles:
        u = rng.random((5, n))
        z = rng.standard_normal(n)
        g = self._g
        mode = np.minimum(np.searchsorted(self._cum_w, u[0], side="right"), len(self.modes) - 1)
        k = self._k_lo[mode] + (self._k_hi[mode] - self._k_lo[mode]) * u[1]
        d_lo, d_hi = self._dur_lo[mode], self._dur_hi[mode]
        if first:
            # A first life fails inside the horizon with the fitted share ``first_p_in``: it then
            # fails after its onset floor plus its duration and before the horizon, which caps its
            # duration. Otherwise it fails between the horizon and the fitted latest failure day.
            # The onset is the failure day less the duration.
            h = g["horizon_days"]
            inside = u[4] < g["first_p_in"]
            d_hi = np.where(inside, np.minimum(d_hi, h - g["first_onset_min"] - 1.0), d_hi)
            d_lo = np.minimum(d_lo, d_hi)
            dur = d_lo + (d_hi - d_lo) * u[2]
            lo_in = g["first_onset_min"] + dur
            hi_in = np.maximum(h - 0.5, lo_in)
            lo_out = np.maximum(h + 1.0, lo_in)
            hi_out = np.maximum(g["first_fail_hi"], lo_out)
            fail_day = np.where(
                inside, lo_in + (hi_in - lo_in) * u[3], lo_out + (hi_out - lo_out) * u[3]
            )
            onset = fail_day - dur
        else:
            dur = d_lo + (d_hi - d_lo) * u[2]
            onset = g["renew_onset_lo"] + (g["renew_onset_hi"] - g["renew_onset_lo"]) * u[3]
        rate = np.exp(g["rate_sd"] * z)
        return Particles.trusted(
            {
                "mode": mode.astype(np.int64),
                "x": -onset / dur,
                "dur": dur,
                "k": k,
                "rate": rate,
            },
            np.full(n, 1.0 / n),
        )

    def prior(self, rng: np.random.Generator, n: int) -> Particles:
        """Particles for a first component at day 0 (the episode start)."""
        return self._draw(rng, n, first=True)

    def renewal(self, rng: np.random.Generator, n: int) -> Particles:
        """Particles for a freshly installed component (onset a fitted number of days away)."""
        return self._draw(rng, n, first=False)

    # ----------------------------------------------------------------------------- dynamics
    def step(
        self,
        p: Particles,
        action: int,
        rng: np.random.Generator,
        *,
        load: float | None = None,
        load_mean: float = 1.0,
    ) -> Particles:
        """One day forward under ``action``.

        Continue and inspect advance the wear clock. Replace discards the particles and draws a
        fresh component. With ``load`` given (the observed load) the clock follows it exactly;
        without it the load wanders around ``load_mean`` (a forecast; nominal 1.0) with the fitted
        ``load_sd``, which is what a forward rollout uses.
        """
        if action == ACTION_REPLACE:
            return self.renewal(rng, p.n)
        if action not in (ACTION_CONTINUE, ACTION_INSPECT):
            raise ValueError(f"unknown action {action!r}")
        g = self._g
        s = p.state
        eps = rng.standard_normal((2, p.n))
        rate = s["rate"] * np.exp(g["rate_walk_sd"] * eps[0])
        dl = (load - 1.0) if load is not None else (load_mean - 1.0) + g["load_sd"] * eps[1]
        rho = rate * np.exp(g["beta_load"] * dl) * self._speed
        x = np.minimum(s["x"] + rho / s["dur"], 1.0)
        return Particles.trusted({**s, "x": x, "rate": rate}, p.weights)

    # --------------------------------------------------------------------------- observation
    def severity(self, p: Particles) -> NDArray[np.float64]:
        return np.power(np.clip(p.state["x"], 0.0, 1.0), p.state["k"])

    def hi_mean(self, p: Particles, load: float = 1.0) -> NDArray[np.float64]:
        g = self._g
        return (
            g["hi_base"]
            + (g["hi_fail"] - g["hi_base"]) * self.severity(p)
            + g["hi_load_coef"] * (load - 1.0)
        )

    def loglik(
        self,
        p: Particles,
        obs: float | None,
        noise_scale: float = 1.0,
        *,
        load: float = 1.0,
    ) -> NDArray[np.float64]:
        """Log density of the health-index reading ``obs`` for every particle (0 when ``None``)."""
        if noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive")
        if obs is None:
            return np.zeros(p.n)
        sd = self._g["hi_noise_sd"] * noise_scale
        z = (obs - self.hi_mean(p, load)) / sd
        return -0.5 * z * z - (math.log(sd) + _HALF_LOG_2PI)

    def failed(self, p: Particles) -> NDArray[np.bool_]:
        return p.state["x"] >= 1.0

    # ------------------------------------------------------------------------ remaining life
    def remaining_life_days(self, p: Particles, load: float = 1.0) -> NDArray[np.float64]:
        """Days to failure per particle at a constant ``load`` and its rate; 0 once failed."""
        s = p.state
        rem = np.maximum(1.0 - s["x"], 0.0) * s["dur"]
        return rem / (s["rate"] * math.exp(self._g["beta_load"] * (load - 1.0)) * self._speed)

    # ------------------------------------------------------------------------- resampling
    def resample(
        self,
        p: Particles,
        idx: NDArray[np.intp],
        rng: np.random.Generator,
        shrink_delta: float = 0.95,
    ) -> Particles:
        """The particles ``idx`` with small Liu-West jitter on the continuous variables.

        Kernel shrinkage (Liu and West 2001): each variable moves toward the weighted mean by
        ``1 - a`` and gets noise of standard deviation ``h`` times the weighted spread, with
        ``h^2 = 1 - ((3 delta - 1) / (2 delta))^2`` and ``a^2 = 1 - h^2``, which preserves the mean
        and variance of the cloud. The duration, shape and rate are jittered in log space; the mode
        is never changed; a failed particle stays failed.
        """
        if not 0.5 < shrink_delta < 1.0:
            raise ValueError("shrink_delta must be in (0.5, 1)")
        h2 = 1.0 - ((3.0 * shrink_delta - 1.0) / (2.0 * shrink_delta)) ** 2
        a = math.sqrt(1.0 - h2)
        h = math.sqrt(h2)
        s = p.state
        n = int(idx.size)
        eps = rng.standard_normal((4, n))
        out = {"mode": s["mode"][idx]}
        for j, name in enumerate(("dur", "k", "rate")):
            v = np.log(s[name])
            m = float(np.dot(p.weights, v))
            var = float(np.dot(p.weights, (v - m) ** 2))
            out[name] = np.exp(a * v[idx] + (1.0 - a) * m + h * math.sqrt(var) * eps[j])
        x = s["x"]
        mx = float(np.dot(p.weights, x))
        vx = float(np.dot(p.weights, (x - mx) ** 2))
        x_new = a * x[idx] + (1.0 - a) * mx + h * math.sqrt(vx) * eps[3]
        out["x"] = np.where(x[idx] >= 1.0, 1.0, np.minimum(x_new, 1.0))
        return Particles.trusted(out, np.full(n, 1.0 / n))


def failure_days(
    model: WorldModel,
    p: Particles,
    rng: np.random.Generator,
    max_days: int = 300,
    **step_kwargs: float,
) -> NDArray[np.float64]:
    """Days until each particle fails, by rolling the model forward under ``continue``.

    The stochastic ``model.step`` is applied day by day (``step_kwargs`` are passed on, e.g.
    ``load_mean`` for the plant model) and a particle's failure is stamped at the middle of the day
    on which ``model.failed`` first holds, so the stamp is unbiased to within half a day. A
    particle already failed gets 0; one still alive after ``max_days`` gets ``inf``. This is the
    model's posterior predictive for the failure time, the quantity a planner integrates over.
    """
    days = np.full(p.n, np.inf)
    days[model.failed(p)] = 0.0
    cur = p
    for d in range(1, max_days + 1):
        if not np.isinf(days).any():
            break
        cur = model.step(cur, ACTION_CONTINUE, rng, **step_kwargs)
        newly = model.failed(cur) & np.isinf(days)
        days[newly] = d - 0.5
    return days


# ----------------------------------------------------------------------------- the tabular model
class TabularWorldModel:
    """A finite world model: transition arrays per action and an observation confusion matrix.

    ``T`` has shape ``(3, S, S)`` (actions in ``ACTION_NAMES`` order, ``T[a, s, s']``), ``O`` has
    shape ``(S, K)`` (``O[s, y]`` is the chance of symbol ``y`` in state ``s``). Every row sums to
    one. ``O_inspect`` is the sharper confusion matrix of an inspection; without it an inspection
    reading uses ``O`` raised to ``1 / noise_scale`` and renormalised. ``cost`` must hold the keys
    ``"continue"``, ``"inspect"``, ``"replace"`` (paid when that action is taken) and ``"failure"``
    (paid per step spent in the failure state); the model only stores them, the planner uses them.
    ``initial`` is the prior over states (default: all mass on state 0) and ``failure_state`` the
    absorbing failed state (default: the last).

    ``predict`` and ``posterior`` are the exact Bayes filter; the particle filter is tested against
    them. An observation is a symbol index passed as a float, as ``loglik`` takes ``float | None``.
    """

    name = "tabular"
    n_actions = N_ACTIONS
    action_names = ACTION_NAMES
    accepts_load = False

    def __init__(
        self,
        T: NDArray,
        O: NDArray,  # noqa: E741 - the usual name of the observation matrix
        cost: Mapping[str, float],
        *,
        O_inspect: NDArray | None = None,
        initial: NDArray | None = None,
        failure_state: int | None = None,
    ) -> None:
        T = np.array(T, dtype=np.float64)
        O = np.array(O, dtype=np.float64)  # noqa: E741
        if T.ndim != 3 or T.shape[0] != N_ACTIONS or T.shape[1] != T.shape[2]:
            raise ValueError(f"T must have shape ({N_ACTIONS}, S, S), got {T.shape}")
        n_s = T.shape[1]
        if O.ndim != 2 or O.shape[0] != n_s:
            raise ValueError(f"O must have shape ({n_s}, K), got {O.shape}")
        _check_stochastic(T, "T")
        _check_stochastic(O, "O")
        if O_inspect is not None:
            O_inspect = np.array(O_inspect, dtype=np.float64)
            if O_inspect.shape != O.shape:
                raise ValueError("O_inspect must have the shape of O")
            _check_stochastic(O_inspect, "O_inspect")
        missing = [k for k in (*ACTION_NAMES, "failure") if k not in cost]
        if missing:
            raise KeyError(f"cost is missing keys: {', '.join(missing)}")
        if any(not math.isfinite(float(v)) for v in cost.values()):
            raise ValueError("costs must be finite")
        if initial is None:
            initial = np.zeros(n_s)
            initial[0] = 1.0
        initial = np.array(initial, dtype=np.float64)
        if initial.shape != (n_s,):
            raise ValueError(f"initial must have shape ({n_s},)")
        _check_stochastic(initial[None, :], "initial")
        fs = n_s - 1 if failure_state is None else int(failure_state)
        if not 0 <= fs < n_s:
            raise ValueError("failure_state is out of range")
        self.T, self.O, self.O_inspect = T, O, O_inspect
        self.cost = dict(cost)
        self.initial = initial
        self.failure_state = fs
        self.n_states = n_s
        self.n_symbols = O.shape[1]
        self._cdf = np.cumsum(T, axis=2)
        self._cdf[:, :, -1] = 1.0
        self._init_cdf = np.cumsum(initial)
        self._init_cdf[-1] = 1.0
        self._o_cache: dict[float, NDArray] = {}

    def observation_matrix(self, noise_scale: float = 1.0) -> NDArray[np.float64]:
        """The confusion matrix for a reading of the given noise: ``O``, or the inspection one."""
        if noise_scale <= 0.0:
            raise ValueError("noise_scale must be positive")
        if noise_scale >= 1.0:
            return self.O
        if self.O_inspect is not None:
            return self.O_inspect
        hit = self._o_cache.get(noise_scale)
        if hit is None:
            hit = self.O ** (1.0 / noise_scale)
            hit = hit / hit.sum(axis=1, keepdims=True)
            self._o_cache[noise_scale] = hit
        return hit

    # --------------------------------------------------------------------- WorldModel methods
    def prior(self, rng: np.random.Generator, n: int) -> Particles:
        s = np.searchsorted(self._init_cdf, rng.random(n), side="left")
        return Particles(
            {"s": np.minimum(s, self.n_states - 1).astype(np.int64)}, np.full(n, 1 / n)
        )

    def renewal(self, rng: np.random.Generator, n: int) -> Particles:
        return Particles({"s": np.zeros(n, dtype=np.int64)}, np.full(n, 1.0 / n))

    def step(self, p: Particles, action: int, rng: np.random.Generator) -> Particles:
        if action not in (0, 1, 2):
            raise ValueError(f"unknown action {action!r}")
        s = p.state["s"]
        u = rng.random(p.n)
        nxt = (self._cdf[action][s] < u[:, None]).sum(axis=1)
        return Particles({"s": np.minimum(nxt, self.n_states - 1).astype(np.int64)}, p.weights)

    def loglik(self, p: Particles, obs: float | None, noise_scale: float = 1.0) -> NDArray:
        if obs is None:
            return np.zeros(p.n)
        y = int(obs)
        if y != obs or not 0 <= y < self.n_symbols:
            raise ValueError(
                f"observation must be a symbol index in [0, {self.n_symbols}), got {obs}"
            )
        col = self.observation_matrix(noise_scale)[:, y]
        with np.errstate(divide="ignore"):
            return np.log(col)[p.state["s"]]

    def failed(self, p: Particles) -> NDArray[np.bool_]:
        return p.state["s"] == self.failure_state

    # --------------------------------------------------------------------- the exact filter
    def predict(self, belief: NDArray[np.float64], action: int) -> NDArray[np.float64]:
        """The belief one step later under ``action``, before any observation."""
        return belief @ self.T[action]

    def posterior(
        self, belief: NDArray[np.float64], symbol: int, inspected: bool = False
    ) -> tuple[NDArray[np.float64], float]:
        """Bayes update on a symbol: the new belief and the probability of seeing the symbol."""
        o = self.observation_matrix(0.25 if inspected else 1.0)[:, symbol]
        joint = belief * o
        z = float(joint.sum())
        return (joint / z if z > 0.0 else belief.copy()), z


def _check_stochastic(a: NDArray[np.float64], name: str) -> None:
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(a.sum(axis=-1), 1.0, atol=1e-9):
        raise ValueError(f"every row of {name} must sum to 1")
