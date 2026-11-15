"""The planner against exact dynamic programming (the validation of 0001 section 8, packet P8b).

``run_validation`` takes the tiny instance of ``exactdp.validation_model``, 50 random beliefs and
a planner with ``n_particles`` particles, and compares them. Two numbers per belief are kept, and
the definitions were fixed before any planner result was seen (docs/adr/progress/P8b-planner.md,
the 02:24 line):

``policy value``
    the cost the planner's *receding-horizon policy* actually achieves from the belief: the planner
    chooses the first action, the belief is updated by Bayes' rule over each of the four symbols,
    the planner is asked again with one step fewer, and so on, summed exactly over the belief tree
    (``exactdp.policy_value``). It cannot be below the optimum.
``estimate``
    the planner's own expected cost of the action it chose at the root: what it believes.

Targets (0001 section 8, set before looking, not widened): the achieved value within 5% of the
exact value on every one of the 50 beliefs (the literal reading of "within 5% on 50 random
beliefs"; the share and the mean are reported as well), and agreement with the exact best first
action on at least 90% of the beliefs whose best and second-best actions differ by more than 2% of
the value. The same targets are reported for the estimate, for information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from closed_loop.planning.exactdp import (
    VALIDATION_HORIZON,
    exact_q_values,
    policy_value,
    validation_beliefs,
    validation_model,
)
from closed_loop.planning.planner import AssetAssessment, BeliefPlanner, particles_from_belief
from closed_loop.planning.worldmodel import ACTION_NAMES, TabularWorldModel

__all__ = ["BeliefResult", "ValidationResult", "run_validation"]

VALUE_TOLERANCE = 0.05
GAP_THRESHOLD = 0.02
AGREEMENT_TARGET = 0.90


@dataclass(frozen=True)
class BeliefResult:
    belief: tuple[float, ...]
    exact_value: float
    exact_action: str
    planner_action: str
    gap_share: float  # (second best - best) / best, exact
    estimate: float  # the planner's own value of its root action
    policy_value: float  # what its receding-horizon policy achieves

    @property
    def policy_error(self) -> float:
        return (self.policy_value - self.exact_value) / self.exact_value

    @property
    def estimate_error(self) -> float:
        return abs(self.estimate - self.exact_value) / self.exact_value

    @property
    def decisive(self) -> bool:
        """The exact best action beats the second best by more than 2% of the value."""
        return self.gap_share > GAP_THRESHOLD

    @property
    def agrees(self) -> bool:
        return self.planner_action == self.exact_action


@dataclass(frozen=True)
class ValidationResult:
    n_particles: int
    seed: int
    horizon: int
    beliefs: tuple[BeliefResult, ...]
    lookahead_readings: int = 0

    @property
    def n(self) -> int:
        return len(self.beliefs)

    @property
    def policy_errors(self) -> NDArray[np.float64]:
        return np.array([b.policy_error for b in self.beliefs])

    @property
    def estimate_errors(self) -> NDArray[np.float64]:
        return np.array([b.estimate_error for b in self.beliefs])

    @property
    def n_policy_within(self) -> int:
        return int(np.sum(self.policy_errors <= VALUE_TOLERANCE))

    @property
    def n_estimate_within(self) -> int:
        return int(np.sum(self.estimate_errors <= VALUE_TOLERANCE))

    @property
    def n_decisive(self) -> int:
        return sum(b.decisive for b in self.beliefs)

    @property
    def n_agree(self) -> int:
        return sum(b.agrees for b in self.beliefs if b.decisive)

    @property
    def agreement(self) -> float:
        return self.n_agree / self.n_decisive if self.n_decisive else float("nan")

    @property
    def value_target_met(self) -> bool:
        """The achieved value is within 5% of exact on every belief."""
        return self.n_policy_within == self.n

    @property
    def agreement_target_met(self) -> bool:
        return self.agreement >= AGREEMENT_TARGET

    def summary(self) -> dict[str, float | int]:
        pe, ee = self.policy_errors, self.estimate_errors
        return {
            "n_particles": self.n_particles,
            "lookahead_readings": self.lookahead_readings,
            "seed": self.seed,
            "beliefs": self.n,
            "policy_within_5pct": self.n_policy_within,
            "policy_error_mean": float(pe.mean()),
            "policy_error_max": float(np.max(pe)),
            "estimate_within_5pct": self.n_estimate_within,
            "estimate_error_mean": float(ee.mean()),
            "estimate_error_max": float(np.max(ee)),
            "decisive_beliefs": self.n_decisive,
            "agree": self.n_agree,
            "agreement": self.agreement,
        }


def run_validation(
    n_particles: int = 128,
    *,
    n_beliefs: int = 50,
    seed: int = 0,
    model: TabularWorldModel | None = None,
    horizon: int = VALIDATION_HORIZON,
    lookahead_readings: int | None = None,
) -> ValidationResult:
    """Compare a ``BeliefPlanner`` of ``n_particles`` with exact DP on the validation instance.

    ``lookahead_readings`` is the planner's depth of preposterior (its default if ``None``).
    """
    m = model if model is not None else validation_model()
    planners = {
        h: BeliefPlanner(
            m,
            n_particles=n_particles,
            horizon_days=h,
            delta_plan=1.0,  # the DP has no chance constraint
            seed=seed,
            full_evaluation=True,
            lookahead_readings=lookahead_readings,
        )
        for h in range(1, horizon + 1)
    }
    counter = [0]

    def assess(belief: NDArray[np.float64], left: int) -> AssetAssessment:
        counter[0] += 1  # every node of the tree gets its own draws, in a fixed order
        return planners[left].assess("A", particles_from_belief(belief), day=counter[0])

    out: list[BeliefResult] = []
    for belief in validation_beliefs(n_beliefs):
        q = exact_q_values(m, belief, horizon)
        order = np.argsort(q, kind="stable")
        best, second = float(q[order[0]]), float(q[order[1]])
        root = assess(belief, horizon)
        achieved = policy_value(m, belief, horizon, lambda b, left: assess(b, left).action)
        out.append(
            BeliefResult(
                tuple(float(x) for x in belief),
                best,
                ACTION_NAMES[int(order[0])],
                root.action_name,
                (second - best) / best,
                root.value,
                achieved,
            )
        )
    return ValidationResult(n_particles, seed, horizon, tuple(out), planners[horizon].lookahead)
