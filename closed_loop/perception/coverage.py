"""Coverage metrics: oracle, learner-visible, Manski bounds, per-life any-miss and the two gaps.

Interface text: docs/adr/0002-interfaces.md, section 3.6; definitions:
docs/adr/0001-architecture.md, section 6.2 and docs/PREREGISTRATION.draft.md, M-COV.
Everything here is SYNTHETIC.

Two truths (docs/adr/0003 A10). Every function here that needs the true RUL reads the *realised*
one (``Checkpoint.true_rul`` or the ``true_rul`` argument), because the estimator is calibrated on
realised RUL. Contract C1 reads the bin of the rate RUL instead; ``perception.contract`` reports the
two side by side.

Populations. ``population`` selects which checkpoints or labels count:

``all``         every one;
``estimated``   the estimator produced a trend interval;
``prior``       the class prior was used (blind window, not degrading, insufficient trend);
``post_blind``  everything outside the blind window (the population of M-COV).

Learner-visible coverage ``C_lrn`` is the share of ``HIT`` among ``HIT + MISS + LOWER_HIT``
(PREREGISTRATION M-COV): a ``LOWER_HIT`` is a label whose upper side is unknown, so it is not a hit.
``UNKNOWN`` is never counted as a hit or a miss. The lower-bound version ``C_lrn^lo`` is the share
of informative labels whose lower bound is known to have held.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from closed_loop.perception.estimator import (
    EstimatorOutput,
    StreamingEstimator,
    TheilSenEstimator,
)
from closed_loop.perception.labels import Checkpoint, Label, LabelStatus
from closed_loop.plant.observe import simulate_lives
from closed_loop.plant.params import ASSET_IDS, PlantParams
from closed_loop.plant.scenario import Scenario

__all__ = [
    "POPULATIONS",
    "CoverageRun",
    "OpenLoopLife",
    "any_miss_rate",
    "checkpoint_coverage",
    "gaps",
    "learner_coverage",
    "manski_bounds",
    "open_loop_lives",
    "oracle_coverage",
    "oracle_lower_coverage",
    "per_life_any_miss",
    "per_life_any_miss_labels",
    "per_life_counts",
    "select",
]

POPULATIONS = ("all", "estimated", "prior", "post_blind")

_T = TypeVar("_T", Checkpoint, Label)


# ----------------------------------------------------------------------------- populations


def _keep(population: str, prior: bool, blind: bool) -> bool:
    if population == "all":
        return True
    if population == "estimated":
        return not prior
    if population == "prior":
        return prior
    if population == "post_blind":
        return not blind
    raise ValueError(f"population must be one of {POPULATIONS}, got {population!r}")


def select(items: Sequence[_T], population: str = "all") -> list[_T]:
    """The checkpoints or labels of one population."""
    if population not in POPULATIONS:
        raise ValueError(f"population must be one of {POPULATIONS}, got {population!r}")
    return [x for x in items if _keep(population, x.prior, x.blind)]


# ----------------------------------------------------------------------------- oracle coverage


def _bounds(intervals: Any) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Lower and upper bounds from an ``(n, 2)`` array or a sequence of objects with lo and hi."""
    if isinstance(intervals, np.ndarray):
        if intervals.ndim != 2 or intervals.shape[1] != 2:
            raise ValueError(f"intervals array must have shape (n, 2), got {intervals.shape}")
        return intervals[:, 0].astype(np.float64), intervals[:, 1].astype(np.float64)
    seq = list(intervals)
    lo = np.array([x.lo for x in seq], dtype=np.float64)
    hi = np.array([x.hi for x in seq], dtype=np.float64)
    return lo, hi


def _masked(
    intervals: Any, true_rul: Any, mask: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    lo, hi = _bounds(intervals)
    truth = np.asarray(true_rul, dtype=np.float64)
    if truth.shape != lo.shape:
        raise ValueError(f"{len(lo)} intervals but {truth.size} true RUL values")
    if np.isnan(truth).any():
        raise ValueError("true_rul holds NaN: the truth is missing for some checkpoint")
    if mask is not None:
        keep = np.asarray(mask, dtype=bool)
        if keep.shape != lo.shape:
            raise ValueError("mask must have one entry per interval")
        lo, hi, truth = lo[keep], hi[keep], truth[keep]
    return lo, hi, truth


def oracle_coverage(intervals: Any, true_rul: Any, mask: Any = None) -> float:
    """``C_orc``: the share of checkpoints with ``lo <= L <= hi``, ``L`` the realised RUL.

    ``intervals`` is an ``(n, 2)`` array or a sequence of objects with ``lo`` and ``hi``; ``mask``
    (optional) keeps a subset. Returns NaN when nothing is left, so a caller must decide what an
    empty population means.
    """
    lo, hi, truth = _masked(intervals, true_rul, mask)
    if truth.size == 0:
        return math.nan
    return float(np.mean((lo <= truth) & (truth <= hi)))


def oracle_lower_coverage(intervals: Any, true_rul: Any, mask: Any = None) -> float:
    """``C_orc^lo``: the share of checkpoints with ``L >= lo`` (the side a bound rule acts on)."""
    lo, _, truth = _masked(intervals, true_rul, mask)
    if truth.size == 0:
        return math.nan
    return float(np.mean(truth >= lo))


def _truths(checkpoints: Sequence[Checkpoint]) -> NDArray[np.float64]:
    out = np.empty(len(checkpoints))
    for i, cp in enumerate(checkpoints):
        if cp.true_rul is None:
            raise ValueError(f"checkpoint {i} ({cp.asset_id}, day {cp.day}) has no sim-only truth")
        out[i] = cp.true_rul
    return out


def checkpoint_coverage(
    checkpoints: Sequence[Checkpoint], population: str = "all", *, lower: bool = False
) -> float:
    """Oracle coverage (or its lower-bound version) of the checkpoints of one population."""
    cps = select(checkpoints, population)
    fn = oracle_lower_coverage if lower else oracle_coverage
    return fn(cps, _truths(cps)) if cps else math.nan


# ----------------------------------------------------------------------------- learner coverage


def learner_coverage(labels: Sequence[Label], population: str = "all") -> dict[str, float]:
    """Counts and rates of the four statuses, ``c_lrn`` and ``c_lrn_lo``.

    Rates are shares of all labels in the population; ``c_lrn`` is ``hit / (hit + miss +
    lower_hit)`` and ``c_lrn_lo`` the share of informative labels (all but ``UNKNOWN``) whose lower
    bound is known to have held. Both are NaN when no label is informative.
    """
    sel = select(labels, population)
    n = len(sel)
    count = dict.fromkeys(LabelStatus, 0)
    lower_ok = 0
    for lab in sel:
        count[lab.status] += 1
        if lab.status is not LabelStatus.UNKNOWN and lab.checkpoint_day + lab.lo <= lab.end_day:
            lower_ok += 1
    hit, miss = count[LabelStatus.HIT], count[LabelStatus.MISS]
    lower_hit, unknown = count[LabelStatus.LOWER_HIT], count[LabelStatus.UNKNOWN]
    informative = hit + miss + lower_hit

    def share(k: int, d: int) -> float:
        return k / d if d else math.nan

    return {
        "n": float(n),
        "hit": float(hit),
        "miss": float(miss),
        "lower_hit": float(lower_hit),
        "unknown": float(unknown),
        "informative": float(informative),
        "hit_rate": share(hit, n),
        "miss_rate": share(miss, n),
        "lower_hit_rate": share(lower_hit, n),
        "unknown_rate": share(unknown, n),
        "c_lrn": share(hit, informative),
        "c_lrn_lo": share(lower_ok, informative),
    }


def manski_bounds(labels: Sequence[Label], population: str = "all") -> tuple[float, float]:
    """``[C_min, C_max]`` on the oracle coverage of the labelled checkpoints.

    ``C_min`` counts every label that is not a certain hit as a miss, ``C_max`` counts every label
    that is not a certain miss as a hit. ``LOWER_HIT`` is not a certain hit (its upper side is
    unknown) and ``UNKNOWN`` is neither, so both widen the bounds. With no labels the bounds are
    ``(0, 1)``: nothing is known.
    """
    sel = select(labels, population)
    if not sel:
        return 0.0, 1.0
    n = len(sel)
    hit = sum(lab.status is LabelStatus.HIT for lab in sel)
    miss = sum(lab.status is LabelStatus.MISS for lab in sel)
    return hit / n, (n - miss) / n


# ----------------------------------------------------------------------------- per-life any-miss


def any_miss_rate(keys: Sequence[Hashable], missed: Sequence[bool]) -> float:
    """Share of distinct ``keys`` (lives) that have at least one ``True`` in ``missed``."""
    k, n = _any_miss_counts(keys, missed)
    return k / n if n else math.nan


def _any_miss_counts(keys: Sequence[Hashable], missed: Sequence[bool]) -> tuple[int, int]:
    if len(keys) != len(missed):
        raise ValueError("keys and missed must have the same length")
    lives: dict[Hashable, bool] = {}
    for key, miss in zip(keys, missed, strict=True):
        lives[key] = lives.get(key, False) or bool(miss)
    return sum(lives.values()), len(lives)


def per_life_counts(checkpoints: Sequence[Checkpoint], population: str = "all") -> tuple[int, int]:
    """``(lives with at least one checkpoint outside its interval, lives)``, on realised RUL.

    A life is ``(asset_id, life_id)``, so pass the checkpoints of one episode; sum the counts over
    episodes to pool them.
    """
    cps = select(checkpoints, population)
    truth = _truths(cps)
    keys = [(cp.asset_id, cp.life_id) for cp in cps]
    missed = [not (cp.lo <= t <= cp.hi) for cp, t in zip(cps, truth, strict=True)]
    return _any_miss_counts(keys, missed)


def per_life_any_miss(checkpoints: Sequence[Checkpoint], population: str = "all") -> float:
    """``M_life``: the share of lives with at least one checkpoint whose realised RUL is outside
    its interval (A10: coverage reads ``true_rul``). NaN with no life."""
    k, n = per_life_counts(checkpoints, population)
    return k / n if n else math.nan


def per_life_any_miss_labels(labels: Sequence[Label], population: str = "all") -> float:
    """The learner-visible version: the share of lives with at least one ``MISS`` label."""
    sel = select(labels, population)
    return any_miss_rate(
        [(lab.asset_id, lab.life_id) for lab in sel],
        [lab.status is LabelStatus.MISS for lab in sel],
    )


# ----------------------------------------------------------------------------- the two gaps


@dataclass(frozen=True)
class CoverageRun:
    """The checkpoints and labels of a set of episodes (concatenate lists to pool episodes)."""

    checkpoints: Sequence[Checkpoint]
    labels: Sequence[Label] = ()


def gaps(
    open_loop: CoverageRun, closed_loop: CoverageRun, population: str = "post_blind"
) -> dict[str, float]:
    """``Gap_A = C_lrn - C_orc`` (closed loop) and ``Gap_B = C_orc(open) - C_orc(closed)``.

    Both are on the same population, with the lower-bound versions alongside (``gap_a_lo``,
    ``gap_b_lo``) and the Manski bounds and label counts of the closed loop. Raises ``ValueError``
    when the open-loop or closed-loop population is empty or the learner has no informative label.
    """
    o_cps = select(open_loop.checkpoints, population)
    c_cps = select(closed_loop.checkpoints, population)
    if not o_cps or not c_cps:
        raise ValueError(f"no checkpoints in population {population!r} for one of the two runs")
    c_orc_open = oracle_coverage(o_cps, _truths(o_cps))
    c_orc_closed = oracle_coverage(c_cps, _truths(c_cps))
    lo_open = oracle_lower_coverage(o_cps, _truths(o_cps))
    lo_closed = oracle_lower_coverage(c_cps, _truths(c_cps))
    lrn = learner_coverage(closed_loop.labels, population)
    if not lrn["informative"]:
        raise ValueError(f"the closed loop has no informative label in population {population!r}")
    m_lo, m_hi = manski_bounds(closed_loop.labels, population)
    return {
        "c_orc_open": c_orc_open,
        "c_orc_closed": c_orc_closed,
        "c_lrn_closed": lrn["c_lrn"],
        "gap_a": lrn["c_lrn"] - c_orc_closed,
        "gap_b": c_orc_open - c_orc_closed,
        "c_orc_lo_open": lo_open,
        "c_orc_lo_closed": lo_closed,
        "c_lrn_lo_closed": lrn["c_lrn_lo"],
        "gap_a_lo": lrn["c_lrn_lo"] - lo_closed,
        "gap_b_lo": lo_open - lo_closed,
        "manski_lo": m_lo,
        "manski_hi": m_hi,
        "n_open": float(len(o_cps)),
        "n_closed": float(len(c_cps)),
        "n_labels": lrn["n"],
        "unknown_rate": lrn["unknown_rate"],
    }


# ----------------------------------------------------------------------------- open-loop lives


@dataclass(frozen=True)
class OpenLoopLife:
    """The first life of one asset in the world where nobody acts, with the estimator's daily
    output on it (SYNTHETIC; built from ``plant.simulate_lives``).

    ``days`` are the checkpoint days (the component is still running: ``day < failure_day`` and
    ``day < horizon_days``). The life ends in an observed failure at ``failure_day`` when that is
    inside the horizon; otherwise it is still running at the end of the episode (censored at
    ``horizon_days``, a replacement-free right censoring).
    """

    seed: int
    asset_id: str
    horizon_days: int
    failure_day: float
    days: NDArray[np.int64]
    outputs: tuple[EstimatorOutput, ...]
    true_rul: NDArray[np.float64]  # realised RUL at each checkpoint day

    @property
    def censored(self) -> bool:
        return self.failure_day >= self.horizon_days

    @property
    def end_day(self) -> float:
        return float(self.horizon_days) if self.censored else self.failure_day


def open_loop_lives(
    params: PlantParams,
    seed: int,
    scenario: Scenario | None = None,
    estimator_factory: Callable[[], StreamingEstimator] | None = None,
) -> list[OpenLoopLife]:
    """Run a fresh estimator over the first lives of the four assets for plant seed ``seed``.

    Calibration seeds come from the roles in 0001 section 11 (``dev`` 1000-1999, ``calib``
    2000-2999); held-out seeds are derived elsewhere and are never touched here.
    """
    table = simulate_lives(params, seed, scenario if scenario is not None else Scenario())
    est = estimator_factory() if estimator_factory is not None else TheilSenEstimator(params)
    lives: list[OpenLoopLife] = []
    for aid in ASSET_IDS:
        fail = float(table.failure_day[aid])
        last = min(table.n_days, math.ceil(fail))  # days d < fail are checkpoints
        hi = table.hi[aid]
        rul = table.true_rul_days[aid]
        outs: list[EstimatorOutput] = []
        for d in range(last):
            h = hi[d]
            outs.append(est.update(aid, d, None if math.isnan(h) else float(h), 0))
        lives.append(
            OpenLoopLife(
                seed=seed,
                asset_id=aid,
                horizon_days=table.n_days,
                failure_day=fail,
                days=np.arange(last, dtype=np.int64),
                outputs=tuple(outs),
                true_rul=np.asarray(rul[:last], dtype=np.float64),
            )
        )
    return lives
