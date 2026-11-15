"""Checkpoints, intervals and label resolution (packet P4a). Everything here is SYNTHETIC.

Interface text: docs/adr/0002-interfaces.md, section 3.6; rules: docs/adr/0001-architecture.md,
section 6.2.

A *checkpoint* is a daily RUL interval ``[lo, hi]`` (days from the checkpoint) the perception layer
handed out for one component life. Its *label* resolves when the life ends, either by a failure at
``T`` (the realised RUL ``T - t`` is then known exactly) or by a replacement at ``tau`` (right
censoring: all that is known is ``T > tau``). The status rules are 0001 section 6.2, exactly:

* failure observed: ``HIT`` if ``t + lo <= T <= t + hi``, else ``MISS``;
* censored at ``tau >= t + hi``: ``MISS`` (``T > tau >= t + hi``, so the upper bound failed);
* censored at ``t + lo <= tau < t + hi``: ``LOWER_HIT`` (the lower bound held, upper unknown);
* censored at ``tau < t + lo``: ``UNKNOWN``.

Nothing else decides what a censored label is worth. A policy that replaces before the lower bound
therefore produces ``UNKNOWN`` labels, because that is what the three rules above give; no case is
special-cased here.

Sim-only fields (``true_rul``, ``health_bin``) are the truth the simulator holds: ``true_rul`` is
the realised RUL (docs/adr/0003 A10: coverage reads it), ``health_bin`` is the bin of the rate RUL
(contract C1 reads it). A learner never sees them; ``learner_view`` strips them.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "EST_ESTIMATED",
    "Checkpoint",
    "Label",
    "LabelStatus",
    "RulInterval",
    "label_status",
    "learner_view",
    "resolve_labels",
]

EST_ESTIMATED = "estimated"  # estimator status that carries a conformal interval of its own


class LabelStatus(StrEnum):
    HIT = "HIT"
    MISS = "MISS"
    LOWER_HIT = "LOWER_HIT"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RulInterval:
    """A RUL interval in days. ``lo`` may be negative or ``-inf`` and ``hi`` may be ``inf``.

    ``prior`` is true when it was built from the class prior (blind window, not degrading,
    insufficient trend) and not from a trend estimate.
    """

    lo: float
    m: float
    hi: float
    level: float
    prior: bool

    def __post_init__(self) -> None:
        if math.isnan(self.lo) or math.isnan(self.hi) or math.isnan(self.m):
            raise ValueError("RulInterval bound is NaN")
        if self.lo > self.hi:
            raise ValueError(f"empty interval: lo={self.lo} > hi={self.hi}")
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level!r}")


@dataclass(frozen=True)
class Checkpoint:
    """One daily interval handed out for one life, kept until the life ends.

    ``m`` and ``sigma`` are the estimator's point and scale, kept so that the conformal score
    ``(L - m) / sigma`` of the label can be formed. ``est_status`` is the estimator status
    (``estimated``, ``prior``, ``not_degrading`` or ``insufficient``): ``prior`` (the blind window)
    is the only one that is "blind", the others carry the class prior as well but are post-blind.
    ``true_rul`` and ``health_bin`` are sim-only.
    """

    life_id: int
    asset_id: str
    day: int
    lo: float
    m: float
    hi: float
    sigma: float
    level: float
    est_status: str = EST_ESTIMATED
    true_rul: float | None = None  # sim-only: realised RUL, failure with no further intervention
    health_bin: int | None = None  # sim-only: bin of the rate RUL (the abstraction's h)

    def __post_init__(self) -> None:
        if math.isnan(self.lo) or math.isnan(self.hi) or math.isnan(self.m):
            raise ValueError("Checkpoint bound is NaN")
        if self.lo > self.hi:
            raise ValueError(f"empty interval: lo={self.lo} > hi={self.hi}")
        if not self.sigma > 0.0:
            raise ValueError(f"sigma must be positive, got {self.sigma!r}")
        if not 0.0 < self.level < 1.0:
            raise ValueError(f"level must be in (0, 1), got {self.level!r}")

    @property
    def prior(self) -> bool:
        """Carries a class-prior interval (any estimator status but ``estimated``)."""
        return self.est_status != EST_ESTIMATED

    @property
    def blind(self) -> bool:
        """Inside the estimator's blind window (status ``prior``)."""
        return self.est_status == "prior"


@dataclass(frozen=True)
class Label:
    """The resolved outcome of one checkpoint. See the module docstring for the status rules.

    ``end_day`` is the day of the failure or the replacement and ``resolved_day`` the first whole
    day on which a daily process can see it (``ceil(end_day)``). ``audit`` marks a life drawn into
    the audit sample (V4). ``m``, ``sigma`` and ``blind`` (additive, optional) carry the
    checkpoint's point, scale and blind flag, so the score of an exactly observed life can be
    formed from the label alone. ``true_rul`` is sim-only and ``None`` in learner-visible copies.
    """

    life_id: int
    asset_id: str
    checkpoint_day: int
    lo: float
    hi: float
    level: float
    resolved_day: int
    status: LabelStatus
    end_day: float
    censored: bool
    audit: bool
    prior: bool
    true_rul: float | None = None
    m: float | None = None
    sigma: float | None = None
    blind: bool = False  # the checkpoint fell inside the estimator's blind window

    @property
    def observed_rul(self) -> float | None:
        """``T - t`` when the life ended in an observed failure, else ``None`` (only a bound)."""
        return None if self.censored else self.end_day - self.checkpoint_day


def label_status(lo: float, hi: float, day: int, end_day: float, censored: bool) -> LabelStatus:
    """Status of a checkpoint at ``day`` with interval ``[lo, hi]``; the four rules of 0001 6.2."""
    if math.isnan(end_day):
        raise ValueError("end_day is NaN")
    if not censored:
        return LabelStatus.HIT if day + lo <= end_day <= day + hi else LabelStatus.MISS
    if end_day >= day + hi:
        return LabelStatus.MISS
    if end_day >= day + lo:
        return LabelStatus.LOWER_HIT
    return LabelStatus.UNKNOWN


def resolve_labels(
    checkpoints: Sequence[Checkpoint],
    end_day: float,
    censored: bool,
    audit: bool,
    resolved_at: float | None = None,
) -> list[Label]:
    """Labels for every checkpoint of ONE life that ended at ``end_day``.

    ``censored`` is true when the life ended in a replacement (or at the end of the episode, with
    the part still running) and false when it ended in an observed failure. The output keeps the
    order of ``checkpoints``. Raises ``ValueError`` for checkpoints of more than one life or for a
    checkpoint dated after an observed failure (a failed component gets no later checkpoints).

    A checkpoint dated after ``end_day`` of a *censored* life is legitimate: the component was
    already out of service for its replacement, but the daily loop kept producing intervals for the
    asset. The rules apply unchanged: ``end_day >= day + lo`` is false for any ``lo >= 0``, so the
    label is ``UNKNOWN`` (the component never runs again, nothing can be learned), unless a bound is
    below zero, where the status is the one the bound implies for a non-negative RUL.

    ``end_day`` must be the last time the component was *running*: the censoring rules say
    ``T > end_day``, and a component taken out of service for its replacement does not age. When
    the learner only learns of the end later (the replacement is completed days after the stop),
    pass that time as ``resolved_at``; the labels then resolve on ``ceil(resolved_at)`` and their
    statuses still follow ``end_day``. It defaults to ``end_day``.
    """
    if not checkpoints:
        return []
    if math.isnan(end_day):
        raise ValueError("end_day is NaN")
    if resolved_at is not None and not resolved_at >= end_day:
        raise ValueError(f"resolved_at ({resolved_at}) cannot precede end_day ({end_day})")
    first = checkpoints[0]
    resolved = math.ceil(end_day if resolved_at is None else resolved_at)
    out: list[Label] = []
    for cp in checkpoints:
        if (cp.asset_id, cp.life_id) != (first.asset_id, first.life_id):
            raise ValueError(
                "resolve_labels takes the checkpoints of one life: got "
                f"{(first.asset_id, first.life_id)} and {(cp.asset_id, cp.life_id)}"
            )
        if cp.day > end_day and not censored:
            raise ValueError(
                f"checkpoint on day {cp.day} lies after the failure of its life ({end_day})"
            )
        out.append(
            Label(
                life_id=cp.life_id,
                asset_id=cp.asset_id,
                checkpoint_day=cp.day,
                lo=cp.lo,
                hi=cp.hi,
                level=cp.level,
                resolved_day=resolved,
                status=label_status(cp.lo, cp.hi, cp.day, end_day, censored),
                end_day=float(end_day),
                censored=bool(censored),
                audit=bool(audit),
                prior=cp.prior,
                true_rul=cp.true_rul,
                m=cp.m,
                sigma=cp.sigma,
                blind=cp.blind,
            )
        )
    return out


def learner_view(label: Label) -> Label:
    """The copy of a label a learner may see: the sim-only truth removed."""
    return dataclasses.replace(label, true_rul=None)
