"""The belief planner (packet P8b): a vectorised rollout over a particle belief.

*What it decides.* For one asset, given a weighted particle belief and a world model, the first
action out of ``continue``, ``inspect`` and ``replace`` (the ``ACTION_*`` of ``worldmodel``). A plan
is a **job choice**, not a command list: ``replace`` becomes ``RAISE_WO`` with a target shift and
the work order is carried out by ``plant.jobs.JobRunner``, which is where the shield sits too.

*How.* The belief is rolled forward under ``continue`` for ``horizon_days`` steps with the world
model's own ``step`` (one set of random draws shared by every candidate, so differences between
candidates are not sampling noise). From the first step at which each particle has failed, ``fs``,
a table ``M[i, j]`` gives the cost of particle ``i`` under "replace at step ``j``" for ``j < H`` and
under "do not replace within the window" for ``j = H``, and ``Pf[i, j]`` the indicator that it
fails before the replacement. ``PlanCosts`` says what a failure, a replacement and a day of
discarded life cost. With ``w`` the particle weights, ``w @ M[:, j]`` is the expected cost of a plan
and ``w @ Pf[:, j]`` its failure probability.

*Value of information.* The first action is ``replace`` (a plan ``j = 0``), or ``continue`` or
``inspect`` followed by what the readings that arrive afterwards allow. Those readings are not
known when the plan is made, so their effect is averaged over what they could say: hypothetical
readings are drawn from the predictive distribution of the state after the step (or, for a finite
model, every symbol is taken with its probability), the weights are reweighted by each reading's
likelihood (a preposterior on the same particles), and the node is the cheaper of replacing now,
given what has been read, and going on to the next reading. After ``lookahead_readings`` readings
(``DEFAULT_LOOKAHEAD``) the best open-loop plan from there is taken. ``continue`` reads through the
daily noise at every step; ``inspect`` reads through a quarter of it at the first, and costs
``PlanCosts.inspect``. The inspection is chosen when its information is worth more than its price,
``voi > inspect - cont``. The ADR's version is one reading. It was measured against exact dynamic
programming first (``validation.py``): with one reading ``continue`` is credited with a single free
reading and ``inspect`` with a sharp one, so inspections are over-valued and the planner chooses
``inspect`` where ``continue`` is best; every extra reading narrows that gap, at a cost in time.

*Chance constraint.* Plans are charged ``M + lambda * Pf``. With ``lambda = 0`` the planner
minimises expected cost. If the failure probability of the chosen first action exceeds
``delta_plan`` the multiplier is raised, by doubling from a small value until the constraint holds
and then by at most ``bisection_steps`` (6) halvings of the bracket, so the returned plan satisfies
it. When even the most cautious action cannot (the asset may already have failed) the result is
marked ``feasible = False`` and the action with the smallest failure probability is returned.

*Truth.* The planner reads the belief (``BeliefTracker.particles``), the observation and the
constraints it is handed. It imports no plant module and never sees ``TruthView`` or the trace;
``tests/planning/test_planner_truth_free.py`` checks the imports, the names, and an episode with
the plant's truth poisoned. Everything is SYNTHETIC and deterministic: all randomness comes from
``rng_for(seed, "plan", asset, day)``.

Interfaces beyond the frozen text of docs/adr/0002 section 3.10, all additive: ``PlanCosts``,
``PlanConstraints``, ``AssetAssessment``, ``PlanDecision``, the optional ``mode_of`` keyword of
``plan``, and the keyword arguments of the constructor after ``seed``.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.core.timegrid import SHIFT_Q
from closed_loop.core.types import Command, Op, PlanCmd
from closed_loop.planning.belief import BeliefTracker, ModelSource
from closed_loop.planning.worldmodel import (
    ACTION_CONTINUE,
    ACTION_INSPECT,
    ACTION_NAMES,
    ACTION_REPLACE,
    Particles,
    TabularWorldModel,
    WorldModel,
)

if TYPE_CHECKING:
    from closed_loop.plant.plant import Observation

__all__ = [
    "AssetAssessment",
    "BeliefPlanner",
    "PlanConstraints",
    "PlanCosts",
    "PlanDecision",
    "mean_life_days",
    "particles_from_belief",
]

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_INSPECT_NOISE = 0.25  # an inspection reads through a quarter of the daily noise
DEFAULT_LOOKAHEAD = 2  # readings the preposterior looks ahead (the ADR's one-step version is 1)
SCREEN_WEIGHT = 1e-3  # belief weight allowed on particles that could fail inside the window
SCREEN_FACTOR = 1.5  # a particle is clear if its remaining life exceeds this many windows...
SCREEN_DAYS = 5.0  # ...plus this many days (the model's own wander is a few per cent)


# ------------------------------------------------------------------------------------------ costs
@dataclass(frozen=True)
class PlanCosts:
    """What a plan is charged. All terms are non-negative.

    ``replace`` and ``cont`` are paid per action taken (a plan "replace at step j" pays ``replace``
    once and ``cont`` for each other step). ``fail_event`` is paid once if the asset fails before it
    is replaced, ``fail_step`` for every step spent failed before the replacement, and
    ``waste_per_day`` for every day of the component's remaining life that a replacement throws
    away. ``inspect`` is the price of the inspection action in place of a ``continue`` step.
    ``lead_days`` is the part of a step during which an asset can still fail after a replacement
    has been decided at its start, because the work order has to be approved and the asset
    stopped: a particle that fails in the step after the decision counts as failed before the
    replacement with that probability (failures are taken as uniform within a step).
    """

    replace: float = 0.0
    cont: float = 0.0
    inspect: float = 0.0
    fail_event: float = 0.0
    fail_step: float = 0.0
    waste_per_day: float = 0.0
    lead_days: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "replace",
            "cont",
            "inspect",
            "fail_event",
            "fail_step",
            "waste_per_day",
            "lead_days",
        ):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {v}")
        if self.lead_days >= 1.0:
            raise ValueError("lead_days must be below one step")
        if self.inspect < self.cont:
            raise ValueError("an inspection must cost at least as much as a continued step")

    @property
    def inspect_price(self) -> float:
        """What an inspection costs over simply continuing."""
        return self.inspect - self.cont

    @classmethod
    def from_tabular(cls, model: TabularWorldModel) -> PlanCosts:
        """The costs of the exact-DP convention (``exactdp``): failure is charged per step."""
        c = model.cost
        return cls(
            replace=float(c["replace"]),
            cont=float(c["continue"]),
            inspect=float(c["inspect"]),
            fail_step=float(c["failure"]),
        )

    @classmethod
    def for_plant(
        cls,
        *,
        planned_job_cost: float,
        unplanned_cost_ratio: float,
        inspect_cost: float,
        mean_life_days: float,
        lead_days: float = 0.0,
    ) -> PlanCosts:
        """The plant's costs in the form a rolling plan needs.

        Every component life ends in one replacement, so a replacement is not charged; a plan pays
        for what differs: the unplanned failure (``unplanned_cost_ratio`` times a planned job, the
        plant's own bookkeeping) and the life thrown away by replacing early, at the rate
        ``planned_job_cost / mean_life_days`` that a replacement costs per day of life it buys
        (the long-run cost rate of a renewal process).
        """
        if mean_life_days <= 0.0:
            raise ValueError("mean_life_days must be positive")
        return cls(
            inspect=inspect_cost,
            fail_event=unplanned_cost_ratio * planned_job_cost,
            waste_per_day=planned_job_cost / mean_life_days,
            lead_days=lead_days,
        )


# ------------------------------------------------------------------------------------ constraints
@dataclass(frozen=True)
class PlanConstraints:
    """The joint limits on the work orders raised together (0001 section 8).

    ``max_open_jobs``: crew; work orders already open plus new ones may not exceed it.
    ``pump_pair``: at most one of the two may have a work order open or be raised now.
    ``confined_jobs`` holds the ``(asset, mode)`` repairs that need a confined-space entry and
    ``confined_open`` counts the ones already open; ``max_confined`` of them may be open at once
    (one confined-space job per shift: a job crosses shifts, so concurrent means the same shift).
    ``part_for`` maps ``(asset, mode)`` to the spare part it consumes, ``on_order`` the parts
    already ordered. ``busy`` assets get no new work order (cool-down, just renewed, the partner's
    repair under way). ``inspect_slots`` bounds the inspections started now (fitters free).
    """

    max_open_jobs: int = 2
    pump_pair: tuple[str, str] | None = None
    confined_jobs: frozenset[tuple[str, str]] = frozenset()
    confined_open: int = 0
    max_confined: int = 1
    part_for: Mapping[tuple[str, str], str] = field(default_factory=dict)
    on_order: frozenset[str] = frozenset()
    busy: frozenset[str] = frozenset()
    no_inspect: frozenset[str] = frozenset()
    inspect_slots: int = 1
    # stopped with no work order for a day and two quarters (``proposers.stall``, 0003 A18): such
    # an asset wants a work order like a failed one; the part is still the planner's to wait for
    stalled: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.max_open_jobs < 0 or self.max_confined < 0 or self.inspect_slots < 0:
            raise ValueError("limits must be non-negative")


@dataclass(frozen=True, eq=False)
class AssetAssessment:
    """What the planner found for one asset: values per first action, the multiplier, the verdict.

    ``q`` is the expected cost of each first action in ``ACTION_NAMES`` order (the cost terms only,
    not the multiplier's); ``q_lagrangian`` includes ``lam * p_fail``. ``p_fail`` is the failure
    probability of the policy that starts with each action. ``open_loop_cost`` and ``open_loop_p``
    are for the plans "replace at step j" (j < horizon) and "no replacement in the window" (the last
    entry), at ``lam = 0``. ``voi`` is the expected cost the inspection saves beyond its price.
    ``full`` says the preposterior was evaluated; when it was not, the verdict is the same one the
    full evaluation would reach and ``q`` holds the open-loop bound for ``continue``.
    """

    asset_id: str
    action: int
    q: NDArray[np.float64]
    q_lagrangian: NDArray[np.float64]
    p_fail: NDArray[np.float64]
    lam: float
    feasible: bool
    voi: float
    full: bool
    n_lambda: int
    open_loop_cost: NDArray[np.float64]
    open_loop_p: NDArray[np.float64]

    @property
    def action_name(self) -> str:
        return ACTION_NAMES[self.action]

    @property
    def value(self) -> float:
        """Expected cost of the chosen first action."""
        return float(self.q[self.action])

    @property
    def p_chosen(self) -> float:
        return float(self.p_fail[self.action])

    @property
    def benefit_of_replacing(self) -> float:
        """How much the best way of waiting costs over replacing now (Lagrangian units)."""
        wait = min(self.q_lagrangian[ACTION_CONTINUE], self.q_lagrangian[ACTION_INSPECT])
        return float(wait - self.q_lagrangian[ACTION_REPLACE])


@dataclass(frozen=True, eq=False)
class PlanDecision:
    """The outcome of ``BeliefPlanner.decide``: the commands and why each asset did what it did."""

    commands: tuple[Command, ...]
    assessments: Mapping[str, AssetAssessment]
    replace: tuple[str, ...]  # assets given a work order
    inspect: tuple[str, ...]
    deferred: Mapping[str, str]  # wanted to replace, could not: asset -> reason
    orders: tuple[str, ...]  # parts ordered


# ------------------------------------------------------------------------------------------ helpers
def particles_from_belief(belief: NDArray[np.float64], key: str = "s") -> Particles:
    """A weighted particle set with one particle per level of a finite model (exact)."""
    b = np.asarray(belief, dtype=np.float64)
    n = b.size
    return Particles({key: np.arange(n, dtype=np.int64)}, b / b.sum())


def mean_life_days(model: WorldModel, seed: int = 0, *, n: int = 512, max_days: int = 400) -> float:
    """Mean days to failure of a freshly installed component, by rolling the model forward.

    Used for the rate at which a day of unexpired life is worth money. A component still alive
    after ``max_days`` counts as ``max_days``.
    """
    rng = rng_for(seed, "plan-mean-life", getattr(model, "name", type(model).__name__))
    renew = getattr(model, "renewal", None)
    p = renew(rng, n) if renew is not None else model.prior(rng, n)
    days = np.full(n, float(max_days))
    alive = np.ones(n, dtype=bool)
    days[model.failed(p)] = 0.0
    alive &= ~model.failed(p)
    for d in range(1, max_days + 1):
        if not alive.any():
            break
        p = model.step(p, ACTION_CONTINUE, rng)
        newly = model.failed(p) & alive
        days[newly] = d - 0.5
        alive &= ~newly
    return float(days.mean())


def _support(model: WorldModel) -> NDArray[np.float64] | None:
    """The finite set of readings a model can produce, or None for a continuous reading."""
    if isinstance(model, TabularWorldModel):
        return np.arange(model.n_symbols, dtype=np.float64)
    fn = getattr(model, "observation_support", None)
    return None if fn is None else np.asarray(fn(), dtype=np.float64)


def _noise_sd(model: WorldModel, p: Particles, load: float | None) -> float:
    """Standard deviation of a daily reading, read off the Gaussian log density at its mean."""
    mu = float(model.hi_mean(p, 1.0 if load is None else load)[0])  # type: ignore[attr-defined]
    kw = _load_kw(model, load)
    ll = float(model.loglik(p, mu, 1.0, **kw)[0])
    return math.exp(-ll - _HALF_LOG_2PI)


def _load_kw(model: WorldModel, load: float | None) -> dict[str, float]:
    if getattr(model, "accepts_load", False) and load is not None:
        return {"load": load}
    return {}


def _step_kw(model: WorldModel, load_mean: float | None) -> dict[str, float]:
    if getattr(model, "accepts_load", False) and load_mean is not None:
        return {"load_mean": load_mean}
    return {}


def _rollout(
    model: WorldModel,
    p: Particles,
    rng: np.random.Generator,
    horizon: int,
    kw: Mapping[str, float],
    keep: int,
) -> tuple[NDArray[np.int64], list[Particles], Particles]:
    """First failed step of each particle (0 already, ``horizon + 1`` never), the sets after steps
    1 to ``keep``, and the set at the end of the window."""
    fs = np.full(p.n, horizon + 1, dtype=np.int64)
    fs[model.failed(p)] = 0
    cur = p
    kept: list[Particles] = []
    for d in range(1, horizon + 1):
        cur = model.step(cur, ACTION_CONTINUE, rng, **kw)
        if d <= keep:
            kept.append(cur)
        fs[model.failed(cur) & (fs == horizon + 1)] = d
    return fs, kept, cur


# ------------------------------------------------------------------------------------------ planner
class _Reader:
    """How the state after each step is read: shared by every tree of one assessment.

    A finite model is read through every symbol: ``lik(t, scale)`` is the ``(symbols, N)`` table of
    the chance of each symbol under each particle. A Gaussian one is read at ``mu[t]`` with
    standard deviation ``sd``. ``doomed`` marks the particles that fail inside the window: a
    handful of readings drawn from the belief would miss a few per cent of weight, so these are
    sampled as a stratum of their own (see ``BeliefPlanner._expand``).
    """

    def __init__(
        self,
        model: WorldModel,
        states: list[Particles],
        load: float | None,
        doomed: NDArray[np.bool_] | None = None,
    ) -> None:
        self.model = model
        self.doomed = doomed if doomed is not None else np.zeros(states[0].n, dtype=bool)
        self.states = states
        self.kw = _load_kw(model, load)
        self.support = _support(model)
        self.mu: list[NDArray[np.float64]] = []
        self.sd = 0.0
        self._lik: dict[tuple[int, float], NDArray[np.float64]] = {}
        if self.support is None and states:
            mean_fn = getattr(model, "hi_mean", None)
            if mean_fn is None:
                raise TypeError(
                    f"{type(model).__name__} needs observation_support (finite readings) or "
                    "hi_mean (Gaussian readings) for the value of information"
                )
            level = 1.0 if load is None else load
            self.mu = [mean_fn(p, level) for p in states]
            self.sd = _noise_sd(model, states[0], load)

    def lik(self, t: int, scale: float) -> NDArray[np.float64]:
        hit = self._lik.get((t, scale))
        if hit is None:
            assert self.support is not None
            p = self.states[t - 1]
            hit = np.exp(
                np.stack([self.model.loglik(p, float(y), scale, **self.kw) for y in self.support])
            )
            self._lik[(t, scale)] = hit
        return hit


@dataclass
class _Level:
    """One step of readings in the tree: who is read, and what the node would do by stopping."""

    cond: NDArray[np.float64]  # (parents, K): probability of each reading given its parent
    stop_m: NDArray[np.float64]  # (nodes,): expected cost of replacing at this step, from the node
    stop_p: NDArray[np.float64]


@dataclass
class _Tree:
    """The preposterior of one first action: readings for ``len(levels)`` steps, then an
    open-loop plan chosen from what has been read (columns ``depth + 1 .. H``)."""

    levels: list[_Level]
    leaf_m: NDArray[np.float64]  # (leaf nodes, H - depth)
    leaf_p: NDArray[np.float64]


@dataclass
class _Context:
    """Everything the multiplier search needs, computed once per assessment."""

    m_rep: float
    p_rep: float
    cont: _Tree
    ins: _Tree
    price: float


class BeliefPlanner:
    """Plans for each asset from its particle belief (see the module docstring).

    ``model`` is a ``WorldModel``, a mapping from asset id to one, or a callable that builds one.
    ``n_particles`` is the number of particles the plan is rolled out with (the belief is
    resampled to it if the tracker holds a different number). ``horizon_days`` is the lookahead in
    steps of the model (days). ``delta_plan`` is the chance constraint: the probability of an
    unplanned failure inside the window may not exceed it. ``costs`` is a ``PlanCosts``, a mapping
    from asset id, or a callable; it may be left out for a ``TabularWorldModel``.
    """

    name = "planner"

    def __init__(
        self,
        model: ModelSource,
        *,
        n_particles: int = 128,
        horizon_days: int = 14,
        delta_plan: float = 0.02,
        seed: int = 0,
        costs: PlanCosts | Mapping[str, PlanCosts] | Callable[[str], PlanCosts] | None = None,
        k_readings: int = 16,
        lookahead_readings: int | None = None,
        k_later: int = 4,
        full_evaluation: bool = False,
        bisection_steps: int = 6,
    ) -> None:
        if n_particles < 2:
            raise ValueError("n_particles must be at least 2")
        if horizon_days < 1:
            raise ValueError("horizon_days must be at least 1")
        if not 0.0 <= delta_plan <= 1.0:
            raise ValueError("delta_plan must be in [0, 1]")
        if k_readings < 2 or k_later < 2 or bisection_steps < 0:
            raise ValueError("k_readings and k_later must be at least 2, bisection_steps >= 0")
        lookahead = DEFAULT_LOOKAHEAD if lookahead_readings is None else lookahead_readings
        if lookahead < 1:
            raise ValueError("lookahead_readings must be at least 1")
        self._source = model
        self._costs = costs
        self.n_particles = n_particles
        self.horizon = horizon_days
        self.delta_plan = delta_plan
        self.seed = seed
        self.k_readings = k_readings
        self.k_later = k_later
        self.lookahead = lookahead
        self.full_evaluation = full_evaluation
        self.bisection_steps = bisection_steps
        self._fresh: dict[str, tuple[WorldModel, NDArray[np.float64]]] = {}
        self._cache: dict[str, tuple[Particles, int, AssetAssessment]] = {}

    # ---------------------------------------------------------------------------------- lookups
    def model_for(self, asset_id: str) -> WorldModel:
        src = self._source
        if isinstance(src, Mapping):
            return src[asset_id]
        if callable(src) and not hasattr(src, "failed"):
            return src(asset_id)
        return src  # type: ignore[return-value]

    def costs_for(self, asset_id: str, model: WorldModel) -> PlanCosts:
        c = self._costs
        if c is None:
            if isinstance(model, TabularWorldModel):
                return PlanCosts.from_tabular(model)
            raise ValueError("costs must be given for a model that is not a TabularWorldModel")
        if isinstance(c, PlanCosts):
            return c
        if isinstance(c, Mapping):
            return c[asset_id]
        return c(asset_id)

    # ---------------------------------------------------------------------------- the matrices
    def _fresh_profile(self, asset_id: str, model: WorldModel, costs: PlanCosts) -> NDArray:
        """``F[m]``: chance a freshly installed component is failed after ``m`` steps, m < H."""
        h = self.horizon
        if costs.fail_event == 0.0 and costs.fail_step == 0.0:
            return np.zeros(h)
        key = f"{asset_id}:{h}"
        hit = self._fresh.get(key)
        if hit is not None and hit[0] is model:
            return hit[1]
        rng = rng_for(self.seed, "plan-fresh", asset_id)
        n = max(self.n_particles, 256)
        renew = getattr(model, "renewal", None)
        p = renew(rng, n) if renew is not None else model.prior(rng, n)
        f = np.zeros(h)
        f[0] = float(np.dot(p.weights, model.failed(p)))
        cur = p
        for m in range(1, h):
            cur = model.step(cur, ACTION_CONTINUE, rng)
            f[m] = float(np.dot(cur.weights, model.failed(cur)))
        self._fresh[key] = (model, f)
        return f

    def _matrices(
        self,
        fs: NDArray[np.int64],
        t_fail: NDArray[np.float64],
        costs: PlanCosts,
        fresh: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``M, Pf`` of shape ``(N, H + 1)``: column ``j < H`` is "replace at step j", ``H`` is
        "no replacement in the window". See the module docstring."""
        h = self.horizon
        j = np.arange(h + 1)[None, :]
        f = fs[:, None]
        event = ((f >= 1) & (f <= j)).astype(np.float64)
        steps_failed = np.maximum(j - np.maximum(f, 1) + 1, 0).astype(np.float64)
        pf = (f <= j).astype(np.float64)
        if costs.lead_days > 0.0:
            # A failure in the step after the decision, before the asset has been stopped.
            near = ((f == j + 1) & (j < h)).astype(np.float64) * costs.lead_days
            event = event + near
            pf = pf + near
        m = costs.fail_event * event + costs.fail_step * steps_failed
        if costs.waste_per_day > 0.0:
            waste = np.where(f > j, t_fail[:, None] - j, 0.0)
            waste[:, h] = 0.0  # no replacement in the window throws nothing away
            m = m + costs.waste_per_day * waste
        # What does not depend on the particle: the actions of the plan and the fresh component
        # that follows a replacement.
        cum = np.concatenate([[0.0], np.cumsum(fresh[1:])])  # sum_{m=1..k} F[m], k = 0..H-1
        after = h - 1 - np.arange(h)  # steps continued after a replacement at j
        post = costs.fail_step * cum[after] + costs.fail_event * fresh[after]
        const = np.empty(h + 1)
        const[:h] = costs.replace + costs.cont * (h - 1) + post
        const[h] = costs.cont * h
        return m + const[None, :], pf

    # ---------------------------------------------------------------------- the preposterior
    @staticmethod
    def _sources(
        weights: NDArray[np.float64],
        doomed: NDArray[np.bool_],
        k: int,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
        """The particles whose readings are drawn, per parent, and the chance of each reading.

        Two strata: particles that fail inside the window, and the rest. A parent whose weight is
        split between them gets ``max(1, k // 4)`` readings from the first, each worth its share
        of the stratum's weight, and the others from the second; stratified inside each. This is
        what keeps a failure risk of a few tenths of a per cent in the tree: sampled in one lot,
        the risk would be absent from most trees and four times too large in a few.
        """
        parents, n = weights.shape
        w_d = np.where(doomed[None, :], weights, 0.0)
        w_s = weights - w_d
        m_d = w_d.sum(axis=1)
        m_s = w_s.sum(axis=1)
        kd = np.where((m_d > 0.0) & (m_s > 0.0), max(1, k // 4), np.where(m_s > 0.0, 0, k))
        cols = np.arange(k)[None, :]
        from_d = cols < kd[:, None]
        k_s = k - kd
        # stratified positions inside the stratum each column draws from
        local = np.where(from_d, cols, cols - kd[:, None])
        size = np.where(from_d, kd[:, None], k_s[:, None])
        pos = (local + rng.random((parents, k))) / np.maximum(size, 1)
        rows = np.arange(parents)[:, None]
        idx = np.empty((parents, k), dtype=np.intp)
        for w_part, mask in ((w_d, from_d), (w_s, ~from_d)):
            tot = w_part.sum(axis=1, keepdims=True)
            cdf = np.cumsum(w_part / np.where(tot > 0.0, tot, 1.0), axis=1)
            cdf[:, -1] = 1.0
            hit = np.searchsorted((cdf + rows).ravel(), (pos + rows).ravel()).reshape(parents, k)
            hit = np.minimum(hit - rows * n, n - 1)
            idx = np.where(mask, hit, idx)
        share = np.where(
            from_d,
            m_d[:, None] / np.maximum(kd[:, None], 1),
            m_s[:, None] / np.maximum(k_s[:, None], 1),
        )
        return idx, share

    def _expand(
        self,
        reader: _Reader,
        t: int,
        weights: NDArray[np.float64],
        rng: np.random.Generator,
        scale: float,
        k: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Read the state after step ``t`` once more, under each row of ``weights`` (parents, N).

        Returns the probability of each possible reading given its parent, ``(parents, K)``, and
        the weights each leaves, ``(parents * K, N)`` (child ``b * K + l``). A finite model is read
        through every symbol, exactly. A continuous one is read through ``k`` hypothetical
        readings per parent, drawn from the predictive distribution of the parent's belief
        (stratified over the particles, antithetic in the noise), each as likely as the others.
        """
        parents, n = weights.shape
        if reader.support is not None:
            lik = reader.lik(t, scale)  # (K, N)
            joint = weights[:, None, :] * lik[None, :, :]  # (parents, K, N)
            kk = lik.shape[0]
        else:
            mu = reader.mu[t - 1]
            sd = reader.sd * scale
            kk = k
            idx, cond = self._sources(weights, reader.doomed, k, rng)
            z = rng.standard_normal((parents, k))
            if k % 2 == 0:
                z[:, k // 2 :] = -z[:, : k // 2]
            y = mu[idx] + sd * z  # (parents, K)
            lik = np.exp(-0.5 * ((y[:, :, None] - mu[None, None, :]) / sd) ** 2)
            joint = weights[:, None, :] * lik
        mass = joint.sum(axis=2)
        ok = mass > 0.0
        post = np.where(
            ok[:, :, None], joint / np.where(ok, mass, 1.0)[:, :, None], weights[:, None, :]
        )
        if reader.support is not None:
            cond = mass
        return cond, post.reshape(parents * kk, n)

    def _tree(
        self,
        reader: _Reader,
        w: NDArray[np.float64],
        m: NDArray[np.float64],
        pf: NDArray[np.float64],
        rng: np.random.Generator,
        first_scale: float,
    ) -> _Tree:
        """Readings at steps 1 to ``len(reader.states)``, then the plans left from there."""
        levels: list[_Level] = []
        u = w[None, :]
        for t in range(1, len(reader.states) + 1):
            k = self.k_readings if t == 1 else self.k_later
            cond, u = self._expand(reader, t, u, rng, first_scale if t == 1 else 1.0, k)
            levels.append(_Level(cond, u @ m[:, t], u @ pf[:, t]))
        depth = len(levels)
        return _Tree(levels, u @ m[:, depth + 1 :], u @ pf[:, depth + 1 :])

    # ------------------------------------------------------------------------------ assessment
    def _screen(
        self,
        model: WorldModel,
        p: Particles,
        costs: PlanCosts,
        load_mean: float | None,
        delta: float,
    ) -> AssetAssessment | None:
        """A belief that cannot fail inside the window needs no rollout: the verdict is continue.

        For a model that can say how long each particle has left (``remaining_life_days``): if
        less than ``SCREEN_WEIGHT`` of the belief has a remaining life under
        ``SCREEN_FACTOR`` windows plus ``SCREEN_DAYS``, the chance of failing inside the window is
        taken as zero (the model's own wander in rate and load is a few per cent of the life, far
        under the margin; the weight left out is under a tenth of a per cent against a budget of
        two). Every cost is then a wasted-life cost and the shortcut of ``assess`` would have
        returned continue after a rollout: this returns the same verdict without one.
        """
        remaining = getattr(model, "remaining_life_days", None)
        if remaining is None or costs.waste_per_day <= 0.0 or delta < SCREEN_WEIGHT:
            return None
        rem = remaining(p, **({"load": load_mean} if load_mean is not None else {}))
        h = self.horizon
        if float(p.weights[rem <= SCREEN_FACTOR * h + SCREEN_DAYS].sum()) >= SCREEN_WEIGHT:
            return None
        mean_rem = float(np.dot(p.weights, rem))
        cost = np.append(costs.waste_per_day * np.maximum(mean_rem - np.arange(h), 0.0), 0.0)
        q = np.array([0.0, costs.inspect_price, cost[0]])
        return AssetAssessment(
            "", ACTION_CONTINUE, q, q.copy(), np.zeros(3), 0.0, True, 0.0, False, 0, cost,
            np.zeros(h + 1),
        )  # fmt: skip

    def _resample(self, p: Particles, rng: np.random.Generator) -> Particles:
        """The belief as ``n_particles`` equally weighted particles (systematic resampling)."""
        if p.n == self.n_particles:
            return p
        n = self.n_particles
        cdf = np.cumsum(p.weights)
        cdf[-1] = 1.0
        idx = np.minimum(np.searchsorted(cdf, (rng.random() + np.arange(n)) / n), p.n - 1)
        return p.take(idx)

    def assess(
        self,
        asset_id: str,
        particles: Particles,
        *,
        load_mean: float | None = None,
        day: int = 0,
        delta_plan: float | None = None,
    ) -> AssetAssessment:
        """The best first action for one asset's belief (see the module docstring)."""
        model = self.model_for(asset_id)
        costs = self.costs_for(asset_id, model)
        delta = self.delta_plan if delta_plan is None else delta_plan
        h = self.horizon
        rng = rng_for(self.seed, "plan", asset_id, day)
        p = self._resample(particles, rng)
        w = p.weights
        clear = None if self.full_evaluation else self._screen(model, p, costs, load_mean, delta)
        if clear is not None:
            return dataclasses.replace(clear, asset_id=asset_id)
        depth = min(self.lookahead, h - 1)
        fs, states, end = _rollout(model, p, rng, h, _step_kw(model, load_mean), depth)
        t_fail = np.where(fs <= h, fs - 0.5, float(h))
        if costs.waste_per_day > 0.0:
            remaining = getattr(model, "remaining_life_days", None)
            if remaining is None:
                raise ValueError("waste_per_day needs a model with remaining_life_days")
            kw = {"load": load_mean} if load_mean is not None else {}
            alive = fs > h
            t_fail = np.where(alive, h + remaining(end, **kw), t_fail)
        fresh = self._fresh_profile(asset_id, model, costs)
        m, pf = self._matrices(fs, t_fail, costs, fresh)

        m_open = w @ m
        p_open = w @ pf
        v_rep, p_rep = float(m_open[0]), float(p_open[0])
        leaf = m_open[1:]
        j_ol = int(np.argmin(leaf))
        price = costs.inspect_price
        cheap = (
            not self.full_evaluation
            and p_open[1 + j_ol] <= delta
            and leaf[j_ol] <= v_rep
            and leaf[j_ol] <= price
        )
        if cheap:
            # Waiting is already the answer: no preposterior can make an inspection pay (its value
            # cannot exceed the cost of the best plan, which is below its price), and replacing now
            # costs more than the best way of waiting.
            q = np.array([leaf[j_ol], leaf[j_ol] + price, v_rep])
            pq = np.array([p_open[1 + j_ol], p_open[1 + j_ol], p_rep])
            return AssetAssessment(
                asset_id, ACTION_CONTINUE, q, q.copy(), pq, 0.0, True, 0.0, False, 0, m_open, p_open
            )

        reader = _Reader(model, states, load_mean, fs <= h)
        ctx = _Context(
            v_rep,
            p_rep,
            self._tree(reader, w, m, pf, rng, 1.0),
            self._tree(reader, w, m, pf, rng, _INSPECT_NOISE),
            price,
        )
        return self._solve(asset_id, ctx, costs, delta, m_open, p_open)

    # ------------------------------------------------------------------- the multiplier search
    @staticmethod
    def _tree_value(tree: _Tree, lam: float) -> tuple[float, float, float]:
        """Expected Lagrangian cost, cost and failure probability of a first action's tree.

        Backward induction: at the leaves the best open-loop plan; at each earlier node the cheaper
        of replacing at that step (given what has been read) and going on to the next reading.
        """
        lag = tree.leaf_m + lam * tree.leaf_p
        j = np.argmin(lag, axis=1)
        rows = np.arange(j.size)
        v_l, v_c, v_p = lag[rows, j], tree.leaf_m[rows, j], tree.leaf_p[rows, j]
        for lv in reversed(tree.levels):
            stop_l = lv.stop_m + lam * lv.stop_p
            stop = stop_l <= v_l
            v_l = np.where(stop, stop_l, v_l)
            v_c = np.where(stop, lv.stop_m, v_c)
            v_p = np.where(stop, lv.stop_p, v_p)
            shape = lv.cond.shape
            v_l = np.sum(lv.cond * v_l.reshape(shape), axis=1)
            v_c = np.sum(lv.cond * v_c.reshape(shape), axis=1)
            v_p = np.sum(lv.cond * v_p.reshape(shape), axis=1)
        return float(v_l[0]), float(v_c[0]), float(v_p[0])

    @classmethod
    def _values(
        cls, ctx: _Context, lam: float
    ) -> tuple[int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Chosen action, Lagrangian values, cost values and failure probabilities per action."""
        vl = np.empty(3)
        vc = np.empty(3)
        pf = np.empty(3)
        for a, tree, extra in (
            (ACTION_CONTINUE, ctx.cont, 0.0),
            (ACTION_INSPECT, ctx.ins, ctx.price),
        ):
            lag, cost, prob = cls._tree_value(tree, lam)
            vl[a], vc[a], pf[a] = lag + extra, cost + extra, prob
        vc[ACTION_REPLACE] = ctx.m_rep
        pf[ACTION_REPLACE] = ctx.p_rep
        vl[ACTION_REPLACE] = ctx.m_rep + lam * ctx.p_rep
        return int(np.argmin(vl)), vl, vc, pf

    def _solve(
        self,
        asset_id: str,
        ctx: _Context,
        costs: PlanCosts,
        delta: float,
        m_open: NDArray[np.float64],
        p_open: NDArray[np.float64],
    ) -> AssetAssessment:
        n_eval = 1
        lam = 0.0
        a, vl, vc, pf = self._values(ctx, lam)
        feasible = pf[a] <= delta + 1e-12
        if not feasible:
            scale = max(costs.fail_event + costs.fail_step * self.horizon, 1e-9)
            lo, hi = 0.0, scale / 64.0
            top = scale * 64.0
            ok = False
            while hi <= top:
                n_eval += 1
                a, vl, vc, pf = self._values(ctx, hi)
                if pf[a] <= delta + 1e-12:
                    ok = True
                    break
                lo, hi = hi, hi * 2.0
            if ok:
                for _ in range(self.bisection_steps):
                    mid = 0.5 * (lo + hi)
                    n_eval += 1
                    am, vlm, vcm, pfm = self._values(ctx, mid)
                    if pfm[am] <= delta + 1e-12:
                        hi, a, vl, vc, pf = mid, am, vlm, vcm, pfm
                    else:
                        lo = mid
                lam, feasible = hi, True
            else:
                # Nothing satisfies the constraint: the most cautious action, flagged.
                a = int(np.argmin(pf + 1e-9 * vl))
                lam, feasible = hi, False
        voi = float(vc[ACTION_CONTINUE] - (vc[ACTION_INSPECT] - ctx.price))
        return AssetAssessment(
            asset_id, a, vc, vl, pf, lam, feasible, voi, True, n_eval, m_open, p_open
        )

    # ------------------------------------------------------------------------------ decisions
    def reset(self) -> None:
        """Forget cached assessments (a new episode)."""
        self._cache = {}

    def _assess_cached(self, asset_id: str, tracker: BeliefTracker) -> AssetAssessment:
        p = tracker.particles(asset_id)
        day = tracker.day(asset_id)
        hit = self._cache.get(asset_id)
        if hit is not None and hit[0] is p and hit[1] == day:
            return hit[2]
        a = self.assess(asset_id, p, load_mean=tracker.load_forecast(asset_id), day=day)
        self._cache[asset_id] = (p, day, a)
        return a

    def map_mode(self, tracker: BeliefTracker, asset_id: str) -> str | None:
        """The most probable failure mode under the belief, or None if the model has no modes."""
        model = self.model_for(asset_id)
        modes = getattr(model, "modes", None)
        p = tracker.particles(asset_id)
        idx = p.state.get("mode")
        if modes is None or idx is None:
            return None
        mass = np.bincount(idx, weights=p.weights, minlength=len(modes))
        return str(modes[int(np.argmax(mass))])

    def decide(
        self,
        obs: Observation,
        tracker: BeliefTracker,
        constraints: PlanConstraints,
        *,
        mode_of: Callable[[str], str | None] | None = None,
    ) -> PlanDecision:
        """Assess every running asset and choose the work orders, inspections and orders.

        ``mode_of(asset)`` gives the failure mode for a work order, or ``None`` to raise none (a
        diagnosis that says the alert is not degradation); the default is the most probable mode
        under the belief. Assets that have failed get a work order whenever the limits allow it.
        The joint choice is two sweeps of coordinate descent over the assets that want a work
        order, in order of what waiting costs them, subject to ``constraints``.
        """
        now_shift = obs.quarter // SHIFT_Q
        assessments: dict[str, AssetAssessment] = {}
        wants: dict[str, float] = {}
        inspect_c: list[tuple[float, str]] = []
        deferred: dict[str, str] = {}
        for aid, view in obs.assets.items():
            if aid in constraints.busy or view.job_open:
                continue
            if view.op is Op.FAIL or (view.op is Op.STOP and aid in constraints.stalled):
                wants[aid] = math.inf
                continue
            if view.op is not Op.RUN:
                continue
            a = self._assess_cached(aid, tracker)
            assessments[aid] = a
            if a.action == ACTION_REPLACE:
                wants[aid] = a.benefit_of_replacing
            elif a.action == ACTION_INSPECT and aid not in constraints.no_inspect:
                inspect_c.append((a.voi, aid))
        modes: dict[str, str] = {}
        for aid in list(wants):
            mode = mode_of(aid) if mode_of is not None else self.map_mode(tracker, aid)
            if mode is None:
                deferred[aid] = "no mode (diagnosis: not degradation)"
                del wants[aid]
            else:
                modes[aid] = mode
        selected = self._select(obs, wants, modes, constraints, deferred)
        commands: list[Command] = [
            Command(aid, PlanCmd.RAISE_WO, f"{modes[aid]}:{now_shift}", self.name)
            for aid in selected
        ]
        slots = constraints.inspect_slots
        inspected: list[str] = []
        for _, aid in sorted(inspect_c, key=lambda t: (-t[0], t[1])):
            if len(inspected) >= slots:
                break
            if aid not in selected:
                inspected.append(aid)
                commands.append(Command(aid, PlanCmd.INSPECT, None, self.name))
        ordered: list[str] = []
        for aid, view in obs.assets.items():
            for part, stock in view.spares.items():
                if stock <= 0 and part not in constraints.on_order and part not in ordered:
                    ordered.append(part)
                    commands.append(Command(aid, PlanCmd.ORDER, part, self.name))
        return PlanDecision(
            tuple(commands),
            assessments,
            tuple(selected),
            tuple(inspected),
            deferred,
            tuple(ordered),
        )

    def plan(
        self,
        obs: Observation,
        tracker: BeliefTracker,
        constraints: PlanConstraints,
        *,
        mode_of: Callable[[str], str | None] | None = None,
    ) -> list[Command]:
        """The planning commands for this decision point (frozen text of 0002 section 3.10)."""
        return list(self.decide(obs, tracker, constraints, mode_of=mode_of).commands)

    # ----------------------------------------------------------------------------- the limits
    @staticmethod
    def _violation(
        obs: Observation,
        chosen: set[str],
        modes: Mapping[str, str],
        c: PlanConstraints,
    ) -> str | None:
        """Why ``chosen`` may not all get a work order now, or None if they may."""
        views = obs.assets
        opened = {a for a, v in views.items() if v.job_open}
        if len(opened) + len(chosen) > c.max_open_jobs:
            return "crew"
        if c.pump_pair is not None:
            for a, b in (c.pump_pair, c.pump_pair[::-1]):
                if a not in chosen or b not in views:
                    continue
                if b in chosen:
                    return "pump pair"
                vb = views[b]
                if vb.op in (Op.ISO, Op.WORK):
                    return "pump pair"
                engaged = vb.job_open or vb.op is Op.FAIL
                if engaged and not (views[a].op is Op.FAIL and vb.op is not Op.FAIL):
                    return "pump pair"
        n_conf = c.confined_open + sum(1 for a in chosen if (a, modes[a]) in c.confined_jobs)
        if n_conf > c.max_confined:
            return "confined space"
        need: dict[str, int] = {}
        for a in chosen:
            part = c.part_for.get((a, modes[a]))
            if part is None:
                return "unknown mode"
            need[part] = need.get(part, 0) + 1
        for v in views.values():
            j = v.job
            if j is not None and j.part_no is not None and v.op is not Op.WORK:
                if j.status != "COMPLETED":
                    need[j.part_no] = need.get(j.part_no, 0) + 1
        for part, n in need.items():
            stock = max((v.spares.get(part, 0) for v in views.values()), default=0)
            if n > stock:
                return f"no spare {part}"
        return None

    def _select(
        self,
        obs: Observation,
        wants: Mapping[str, float],
        modes: Mapping[str, str],
        c: PlanConstraints,
        deferred: dict[str, str],
    ) -> list[str]:
        order = sorted(wants, key=lambda a: (-wants[a], a))
        chosen: set[str] = set()
        for _ in range(2):  # two sweeps of coordinate descent
            for a in order:
                if a in chosen:
                    continue
                why = self._violation(obs, chosen | {a}, modes, c)
                if why is None:
                    chosen.add(a)
                    deferred.pop(a, None)
                    continue
                deferred[a] = why
                for b in sorted(chosen, key=lambda x: (wants[x], x)):
                    swapped = (chosen - {b}) | {a}
                    if wants[b] < wants[a] and self._violation(obs, swapped, modes, c) is None:
                        chosen = swapped
                        deferred.pop(a, None)
                        deferred[b] = why
                        break
        return [a for a in order if a in chosen]
