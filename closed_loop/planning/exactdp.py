"""Exact finite-horizon value of a belief on a ``TabularWorldModel`` (packet P8b).

The validation of 0001 section 8: one asset on a handful of hidden levels, three actions, a
confusion matrix for the observation, and a short horizon, small enough that the optimal
belief-space policy can be found by expectimax over the whole belief tree. The planner is then
compared with it.

*Timing of costs* (the planner uses the same convention, ``planner.PlanCosts.from_tabular``). At
each of ``horizon`` steps the controller sees a belief ``b`` over the hidden level, chooses an
action ``a`` in ``ACTION_NAMES`` (continue, inspect, replace), pays ``cost[a]``, the level moves
by ``T[a]``, it pays ``cost["failure"]`` if the new level is the failure state (so the failure
cost is *per step spent failed*; replacing from the failure state resets the level and stops the
charge), a symbol is read from the new level through ``O`` (an inspection reads through the sharper
matrix of ``observation_matrix(0.25)``; a replacement and a continuation read through ``O``) and
the belief is updated by Bayes' rule. There is no value after the last step.

The tree is evaluated level by level with numpy: the beliefs of depth ``d`` are one array, their
``3 x K`` children the next, so a horizon-5 tree of at most ``12^5 = 248,832`` leaves costs a few
milliseconds. Everything is SYNTHETIC and exact up to floating point.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from closed_loop.core.rng import rng_for
from closed_loop.planning.worldmodel import ACTION_NAMES, N_ACTIONS, TabularWorldModel

__all__ = [
    "VALIDATION_HORIZON",
    "ChooseFirstAction",
    "exact_q_values",
    "exact_value",
    "policy_value",
    "validation_beliefs",
    "validation_model",
]

# ``choose(belief, remaining_steps)`` -> index of the first action, into ``ACTION_NAMES``.
ChooseFirstAction = Callable[[NDArray[np.float64], int], int]

_INSPECT_NOISE = 0.25  # the noise scale of an inspection reading (worldmodel.ACTION_INSPECT)


def _observation_matrices(model: TabularWorldModel) -> NDArray[np.float64]:
    """``(3, S, K)``: the matrix a reading is drawn from after each action."""
    plain = model.observation_matrix(1.0)
    sharp = model.observation_matrix(_INSPECT_NOISE)
    return np.stack([plain, sharp, plain])


def _step_costs(model: TabularWorldModel) -> NDArray[np.float64]:
    """``(3,)``: what each action costs by itself."""
    return np.array([float(model.cost[name]) for name in ACTION_NAMES])


def _check_belief(model: TabularWorldModel, belief: NDArray[np.float64]) -> NDArray[np.float64]:
    b = np.asarray(belief, dtype=np.float64)
    if b.shape != (model.n_states,):
        raise ValueError(f"belief must have shape ({model.n_states},), got {b.shape}")
    if np.any(b < 0.0) or abs(float(b.sum()) - 1.0) > 1e-9:
        raise ValueError("belief must be a probability vector")
    return b


def _expected_failure(model: TabularWorldModel, pred: NDArray[np.float64]) -> NDArray[np.float64]:
    """Failure charge of a predicted level distribution, for each row of ``pred``."""
    return float(model.cost["failure"]) * pred[..., model.failure_state]


def _children(
    model: TabularWorldModel,
    beliefs: NDArray[np.float64],
    obs_mats: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """For beliefs ``(B, S)``: predicted levels ``(B, 3, S)``, symbol probabilities ``(B, 3, K)``
    and the posterior beliefs ``(B, 3, K, S)`` (rows with probability zero are left as the
    prediction, they carry no weight)."""
    pred = np.einsum("bs,ast->bat", beliefs, model.T)  # (B, 3, S)
    joint = pred[:, :, :, None] * np.transpose(obs_mats, (0, 1, 2))[None]  # (B, 3, S, K)
    prob = joint.sum(axis=2)  # (B, 3, K)
    post = np.transpose(joint, (0, 1, 3, 2))  # (B, 3, K, S)
    safe = np.where(prob > 0.0, prob, 1.0)[..., None]
    post = np.where(prob[..., None] > 0.0, post / safe, pred[:, :, None, :])
    return pred, prob, post


def exact_q_values(
    model: TabularWorldModel, belief: NDArray[np.float64], horizon: int
) -> NDArray[np.float64]:
    """The optimal cost-to-go of each first action, ``(3,)`` in ``ACTION_NAMES`` order."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    b0 = _check_belief(model, belief)
    obs_mats = _observation_matrices(model)
    step = _step_costs(model)
    # Expand the tree level by level; level d holds the beliefs reachable in d steps.
    levels: list[NDArray[np.float64]] = [b0[None, :]]
    cache: list[tuple[NDArray[np.float64], NDArray[np.float64]]] = []
    for _ in range(horizon - 1):
        pred, prob, post = _children(model, levels[-1], obs_mats)
        cache.append((pred, prob))
        levels.append(post.reshape(-1, model.n_states))
    # Last decision: no continuation value.
    pred_last = np.einsum("bs,ast->bat", levels[-1], model.T)
    q = step[None, :] + _expected_failure(model, pred_last)  # (B, 3)
    v = q.min(axis=1)
    # Backward induction.
    for d in range(horizon - 2, -1, -1):
        pred, prob = cache[d]
        k = prob.shape[2]
        child_v = v.reshape(levels[d].shape[0], N_ACTIONS, k)
        q = step[None, :] + _expected_failure(model, pred) + np.einsum("bak,bak->ba", prob, child_v)
        v = q.min(axis=1)
    return np.asarray(q[0], dtype=np.float64)


def exact_value(
    model: TabularWorldModel, belief: NDArray[np.float64], horizon: int
) -> tuple[float, str]:
    """Optimal expected cost of ``belief`` over ``horizon`` steps and the best first action.

    The action is one of ``ACTION_NAMES``; a tie goes to the first in that order.
    """
    q = exact_q_values(model, belief, horizon)
    best = int(np.argmin(q))
    return float(q[best]), ACTION_NAMES[best]


def policy_value(
    model: TabularWorldModel,
    belief: NDArray[np.float64],
    horizon: int,
    choose: ChooseFirstAction,
) -> float:
    """Exact expected cost of a receding-horizon policy on the same tree.

    ``choose(belief, remaining_steps)`` returns the index of the first action to take (an index
    into ``ACTION_NAMES``). The belief is then updated by Bayes' rule over every symbol and the
    policy is asked again, so a horizon-5 tree costs ``1 + 4 + 16 + 64 + 256`` calls of ``choose``
    for a four-symbol model (branches of probability zero are skipped). The result is the value
    the policy actually achieves, not the value it believes it will achieve.
    """
    b0 = _check_belief(model, belief)
    obs_mats = _observation_matrices(model)
    step = _step_costs(model)
    fail = float(model.cost["failure"])

    def value(b: NDArray[np.float64], left: int) -> float:
        a = int(choose(b, left))
        if not 0 <= a < N_ACTIONS:
            raise ValueError(f"choose returned {a!r}, expected an index into {ACTION_NAMES}")
        pred = b @ model.T[a]
        total = float(step[a]) + fail * float(pred[model.failure_state])
        if left == 1:
            return total
        joint = pred[:, None] * obs_mats[a]  # (S, K)
        prob = joint.sum(axis=0)
        for y in range(prob.size):
            if prob[y] <= 0.0:
                continue
            total += float(prob[y]) * value(joint[:, y] / prob[y], left - 1)
        return total

    return value(b0, horizon)


# ---------------------------------------------------------------- the instance of 0001 section 8
VALIDATION_HORIZON = 5


def validation_model() -> TabularWorldModel:
    """The tiny instance the planner is validated on (0001 section 8; fixed before any result).

    One asset, five hidden levels 0-4 with level 4 the absorbing failure. Continuing or inspecting
    keeps the level with probability 0.8 and raises it by one otherwise; replacing returns to level
    0. Four symbols are read through a confusion matrix whose rows peak at their own level
    (a failed asset is evident); an inspection reads through the matrix raised to the power
    ``1 / 0.25`` and renormalised, the model's default. Costs: continue 0, inspect 0.1, replace 1,
    and 2 per step spent failed.

    The first draft of the failure charge was 4 per step. The exact solution of that instance
    (nothing but the DP was run, no planner) chose to replace at once for 45 of 50 beliefs, so
    agreement on the best action would have meant little; the charge and the belief distribution
    were changed to one whose exact best action is mixed (continue, inspect and replace each best
    for a sizeable share) before the planner was run on anything.
    """
    n = 5
    rise = np.zeros((n, n))
    for s in range(n - 1):
        rise[s, s], rise[s, s + 1] = 0.8, 0.2
    rise[n - 1, n - 1] = 1.0
    replace = np.zeros((n, n))
    replace[:, 0] = 1.0
    obs = np.array(
        [
            [0.70, 0.20, 0.08, 0.02],
            [0.25, 0.50, 0.20, 0.05],
            [0.10, 0.25, 0.45, 0.20],
            [0.02, 0.08, 0.20, 0.70],
            [0.01, 0.04, 0.15, 0.80],
        ]
    )
    cost = {"continue": 0.0, "inspect": 0.1, "replace": 1.0, "failure": 2.0}
    return TabularWorldModel(np.stack([rise, rise, replace]), obs, cost)


def validation_beliefs(n: int = 50, seed: int = 0) -> list[NDArray[np.float64]]:
    """``n`` random beliefs over the five levels, Dirichlet(2, 1.5, 1, 1, 0.5), keyed by ``seed``.

    The concentration leans toward the healthy levels, so that waiting is sometimes right.
    """
    alpha = np.array([2.0, 1.5, 1.0, 1.0, 0.5])
    return [rng_for(seed, "exactdp-validation", i).dirichlet(alpha) for i in range(n)]
