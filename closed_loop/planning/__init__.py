"""World models, belief tracking and the belief planner (packets P8a and P8b).

P8a ships ``worldmodel`` and ``belief``; P8b adds ``planner`` (the belief planner, free of any
plant import), ``exactdp`` (the exact value on a finite model) and ``proposer``
(``PlannerProposer``, which needs the plant). Importing this package pulls numpy only:
``PlannerProposer`` and ``default_models`` are loaded on first use, and ``fitting``, which reads
the simulator on development seeds, is imported on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from closed_loop.planning.belief import BeliefTracker, systematic_resample
from closed_loop.planning.exactdp import (
    VALIDATION_HORIZON,
    exact_q_values,
    exact_value,
    policy_value,
    validation_beliefs,
    validation_model,
)
from closed_loop.planning.planner import (
    AssetAssessment,
    BeliefPlanner,
    PlanConstraints,
    PlanCosts,
    PlanDecision,
    mean_life_days,
    particles_from_belief,
)
from closed_loop.planning.worldmodel import (
    ACTION_CONTINUE,
    ACTION_INSPECT,
    ACTION_NAMES,
    ACTION_REPLACE,
    N_ACTIONS,
    Particles,
    PlantWorldModel,
    RenewableWorldModel,
    TabularWorldModel,
    WorldModel,
    failure_days,
    weighted_quantile,
)

if TYPE_CHECKING:
    from closed_loop.planning.proposer import PlannerProposer, default_models

_LAZY = {"PlannerProposer", "default_models"}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from closed_loop.planning import proposer

        return getattr(proposer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VALIDATION_HORIZON",
    "AssetAssessment",
    "BeliefPlanner",
    "PlanConstraints",
    "PlanCosts",
    "PlanDecision",
    "PlannerProposer",
    "default_models",
    "exact_q_values",
    "exact_value",
    "mean_life_days",
    "particles_from_belief",
    "policy_value",
    "validation_beliefs",
    "validation_model",
    "ACTION_CONTINUE",
    "ACTION_INSPECT",
    "ACTION_NAMES",
    "ACTION_REPLACE",
    "N_ACTIONS",
    "BeliefTracker",
    "Particles",
    "PlantWorldModel",
    "RenewableWorldModel",
    "TabularWorldModel",
    "WorldModel",
    "failure_days",
    "systematic_resample",
    "weighted_quantile",
]
