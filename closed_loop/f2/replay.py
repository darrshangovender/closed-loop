"""Replay a scenario vector found at F1 on the F2 stack (packet P11c). Everything is SYNTHETIC.

``replay_theta`` is the multi-fidelity step of 0001 section 9: a ``theta`` that made an F1 episode
violate a property is decoded again, the F1 episode and an F2 episode are run on the same plant seed
(so the same exogenous streams: load, wear, refusals, gas, notes, holder absences, approver draws),
and the result says whether F2 shows a violation of the same property.

An unconfirmed replay gets exactly one reason code. The rule is mechanical and ordered, so the
histogram of codes in a report can be reproduced from the outcomes alone:

``replay_nondeterminism``
    only when ``verify_determinism`` is on: a second F2 run (or a second F1 run) of the same
    ``(arm, seed, theta)`` does not give the same violated properties;
``agent_blocked_earlier``
    F2 never put a repair job on the plant, and the violated F1 property needs work in progress
    (P1, P2, P3, P6): the agent's draft was an inspection, or it was rejected, escalated or never
    raised, so the situation the violation needs did not arise;
``estimator_mismatch``
    F1's earliest violation was labelled with a broken assumption of the perception contract
    (``C1`` or ``A_phys``), or the violated property is P4 or P5, which depend on when assets fail
    and on what the perception layer let the controller do about it: the F2 estimator is the
    difference;
``plant_logic``
    anything else: the same plant, a different command stream, no violation.

``confirmed`` means that at least one property violated at F1 is also violated at F2 (reported in
``common``). Nothing here decides whether a find is a defect: that is the attribution of 0003 A5.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from closed_loop.eval.arms import Arm
from closed_loop.eval.context import PreparedContext, current_context
from closed_loop.eval.episode import simulate_episode
from closed_loop.eval.metrics import Attribution
from closed_loop.f2.episode import F2Log, require_siblings, simulate_f2_episode
from closed_loop.plant.scenario import Scenario

__all__ = [
    "JOB_PROPERTIES",
    "PERCEPTION_PROPERTIES",
    "REASON_CODES",
    "REPLAY_SEED",
    "ReplayOutcome",
    "replay_theta",
]

REPLAY_SEED: Final[int] = 3000  # the first ``search`` seed: P9b's plant seed is the search seed
REASON_CODES: Final[tuple[str, ...]] = (
    "estimator_mismatch",
    "plant_logic",
    "replay_nondeterminism",
    "agent_blocked_earlier",
)
JOB_PROPERTIES: Final[frozenset[str]] = frozenset({"P1", "P2", "P3", "P6"})  # work in progress
PERCEPTION_PROPERTIES: Final[frozenset[str]] = frozenset({"P4", "P5"})
_PERCEPTION_LABELS: Final[frozenset[str]] = frozenset({"C1", "A_phys"})


@dataclass(frozen=True)
class ReplayOutcome:
    """One replay: what F1 violated, what F2 violated, and why an unconfirmed one was not."""

    arm: str
    seed: int
    regime: str
    confirmed: bool
    reason: str  # "confirmed" or one of REASON_CODES
    f1_props: tuple[str, ...]
    f2_props: tuple[str, ...]
    common: tuple[str, ...]
    f1_labels: tuple[str, ...]  # the A5 label of F1's earliest event of each violated property
    f1_planned_jobs: int
    f2_repairs: int  # repair jobs the agent chain put on the plant
    f2_inspections: int
    f2_drafts: int
    f2_runs_rejected: int
    f2_runs_escalated: int
    evidence: str
    f1_wall_s: float
    f2_wall_s: float
    f2_log: F2Log | None = None


def _violated(min_robustness: Mapping[str, float]) -> tuple[str, ...]:
    return tuple(p for p in sorted(min_robustness) if min_robustness[p] < 0.0)


def _first_labels(attributions: Sequence[Attribution], props: tuple[str, ...]) -> tuple[str, ...]:
    first: dict[str, str] = {}
    for a in attributions:
        first.setdefault(a.prop, a.label)
    return tuple(first[p] for p in props if p in first)


def classify_unconfirmed(
    *,
    f1_props: tuple[str, ...],
    f1_labels: tuple[str, ...],
    f2_repairs: int,
    nondeterministic: bool,
) -> tuple[str, str]:
    """The reason code of an unconfirmed replay and the sentence that backs it."""
    if nondeterministic:
        return "replay_nondeterminism", "a second run of the same episode gave a different result"
    needs_job = bool(set(f1_props) & JOB_PROPERTIES)
    if f2_repairs == 0 and needs_job:
        return (
            "agent_blocked_earlier",
            f"F1 violated {list(f1_props)}, which need work in progress, and the agent chain put "
            "no repair job on the plant at F2",
        )
    if set(f1_labels) & _PERCEPTION_LABELS or set(f1_props) & PERCEPTION_PROPERTIES:
        return (
            "estimator_mismatch",
            f"F1 violated {list(f1_props)}, labelled {list(f1_labels)}: a property or a label that "
            "depends on the perception contract, whose estimator differs at F2",
        )
    return (
        "plant_logic",
        f"F1 violated {list(f1_props)} with {f2_repairs} repair job(s) at F2 and no common "
        "violation: the plant saw a different command stream",
    )


def replay_theta(
    arm: Arm,
    theta: NDArray[np.float64],
    regime: str,
    seed: int = REPLAY_SEED,
    *,
    context: PreparedContext | None = None,
    verify_determinism: bool = False,
) -> ReplayOutcome:
    """Run ``arm`` on plant seed ``seed`` at F1 and at F2 under the scenario ``theta`` decodes to.

    ``seed`` is the plant seed of the F1 find (P9b's plant seed is its search seed). Raises
    ``F2Unavailable`` if a sibling is missing. ``verify_determinism`` runs the F2 episode a second
    time and, when it disagrees with the first, reports ``replay_nondeterminism`` (it doubles the
    cost, so a report switches it on for a few replays, not for all).
    """
    require_siblings()  # before the F1 episode, so a missing sibling costs nothing
    ctx = context if context is not None else current_context()
    scenario = Scenario.decode(theta, regime)
    f1 = simulate_episode(arm, seed, scenario, context=ctx).result
    f1_props = _violated(f1.min_robustness)
    if not f1_props:
        raise ValueError(
            f"theta does not violate any property at F1 on seed {seed}: there is nothing to replay"
        )
    f2 = simulate_f2_episode(arm, seed, scenario, context=ctx)
    f2_props = _violated(f2.result.min_robustness)
    common = tuple(p for p in f1_props if p in f2_props)
    log = f2.log
    labels = _first_labels(f1.attributions, f1_props)
    confirmed = bool(common)
    reason = "confirmed"
    evidence = f"F1 and F2 both violate {list(common)}"
    if not confirmed:
        nondeterministic = False
        if verify_determinism:
            again = simulate_f2_episode(arm, seed, scenario, context=ctx)
            nondeterministic = _violated(again.result.min_robustness) != f2_props
        reason, evidence = classify_unconfirmed(
            f1_props=f1_props,
            f1_labels=labels,
            f2_repairs=log.repairs_issued,
            nondeterministic=nondeterministic,
        )
    return ReplayOutcome(
        arm=arm.name,
        seed=seed,
        regime=regime,
        confirmed=confirmed,
        reason=reason,
        f1_props=f1_props,
        f2_props=f2_props,
        common=common,
        f1_labels=labels,
        f1_planned_jobs=f1.planned_jobs,
        f2_repairs=log.repairs_issued,
        f2_inspections=log.inspections_issued,
        f2_drafts=len(log.drafts),
        f2_runs_rejected=log.runs_rejected,
        f2_runs_escalated=log.runs_escalated,
        evidence=evidence,
        f1_wall_s=f1.wall_s,
        f2_wall_s=f2.result.wall_s,
        f2_log=log,
    )
