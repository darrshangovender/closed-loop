"""P8b: the exact value of a belief on a tabular model (expectimax over the belief tree).

The vectorised level-by-level evaluation in ``exactdp`` is checked against a plain recursive
reference written here with Python loops, against hand-worked one-step values, and against the
properties any correct finite-horizon value must have.
"""

from __future__ import annotations

import numpy as np
import pytest

from closed_loop.core.rng import rng_for
from closed_loop.planning.exactdp import (
    exact_q_values,
    exact_value,
    policy_value,
)
from closed_loop.planning.worldmodel import (
    ACTION_CONTINUE,
    ACTION_INSPECT,
    ACTION_NAMES,
    ACTION_REPLACE,
    TabularWorldModel,
)
from tests.planning._helpers import small_tabular


def reference_q(model: TabularWorldModel, belief: np.ndarray, horizon: int) -> list[float]:
    """Plain-loop expectimax: the first-action costs, from the definition in exactdp's docstring."""
    n, k = model.n_states, model.n_symbols
    fail = model.cost["failure"]

    def q_of(b: list[float], a: int, left: int) -> float:
        total = model.cost[ACTION_NAMES[a]]
        pred = [sum(b[s] * model.T[a, s, t] for s in range(n)) for t in range(n)]
        total += fail * pred[model.failure_state]
        if left == 1:
            return total
        obs = model.observation_matrix(0.25 if a == ACTION_INSPECT else 1.0)
        for y in range(k):
            joint = [pred[t] * obs[t, y] for t in range(n)]
            p_y = sum(joint)
            if p_y <= 0.0:
                continue
            post = [j / p_y for j in joint]
            total += p_y * min(q_of(post, a2, left - 1) for a2 in range(3))
        return total

    b = [float(x) for x in belief]
    return [q_of(b, a, horizon) for a in range(3)]


def a_belief(seed: int = 0) -> np.ndarray:
    rng = rng_for(7, "exactdp-test", seed)
    return rng.dirichlet(np.ones(4))


class TestOneStep:
    def test_the_first_actions_cost_by_hand(self) -> None:
        m = small_tabular()
        b = np.array([0.5, 0.3, 0.2, 0.0])
        q = exact_q_values(m, b, 1)
        # continue: nothing paid, failure state reached from level 2 with probability 0.2
        assert q[ACTION_CONTINUE] == pytest.approx(0.2 * 0.2 * 8.0)
        # inspect: same dynamics, pays 0.05
        assert q[ACTION_INSPECT] == pytest.approx(0.05 + 0.2 * 0.2 * 8.0)
        # replace: pays 1.0 and the level is 0 whatever it was
        assert q[ACTION_REPLACE] == pytest.approx(1.0)

    def test_the_value_and_the_name_are_the_best_action(self) -> None:
        m = small_tabular()
        v, name = exact_value(m, np.array([0.0, 0.0, 0.1, 0.9]), 1)
        # level 3 is the failure state: continuing from it pays 8 each step
        assert name == "replace" and v == pytest.approx(1.0)
        v, name = exact_value(m, np.array([1.0, 0.0, 0.0, 0.0]), 1)
        assert name == "continue" and v == pytest.approx(0.0)

    def test_replace_from_the_failure_state_stops_the_charge(self) -> None:
        m = small_tabular()
        q = exact_q_values(m, np.array([0.0, 0.0, 0.0, 1.0]), 1)
        assert q[ACTION_CONTINUE] == pytest.approx(8.0)
        assert q[ACTION_REPLACE] == pytest.approx(1.0)


class TestAgainstTheReference:
    @pytest.mark.parametrize("horizon", [1, 2, 3, 4])
    @pytest.mark.parametrize("seed", range(4))
    def test_matches_the_plain_recursion(self, horizon: int, seed: int) -> None:
        m = small_tabular()
        b = a_belief(seed)
        assert exact_q_values(m, b, horizon) == pytest.approx(reference_q(m, b, horizon))

    def test_matches_without_an_inspection_matrix(self) -> None:
        m = small_tabular(inspect_sharper=False)
        b = a_belief(3)
        assert exact_q_values(m, b, 3) == pytest.approx(reference_q(m, b, 3))

    def test_horizon_five_runs_and_matches_on_one_belief(self) -> None:
        m = small_tabular()
        b = a_belief(5)
        assert exact_q_values(m, b, 5) == pytest.approx(reference_q(m, b, 5))


class TestProperties:
    def test_value_never_falls_as_the_horizon_grows(self) -> None:
        m = small_tabular()
        b = a_belief(1)
        values = [exact_value(m, b, h)[0] for h in range(1, 6)]
        assert all(b2 >= a - 1e-12 for a, b2 in zip(values, values[1:], strict=False))

    def test_the_value_is_the_minimum_of_the_q_values(self) -> None:
        m = small_tabular()
        b = a_belief(2)
        q = exact_q_values(m, b, 4)
        assert exact_value(m, b, 4)[0] == pytest.approx(float(q.min()))

    def test_a_certain_failure_is_replaced_at_every_horizon(self) -> None:
        m = small_tabular()
        for h in range(1, 6):
            assert exact_value(m, np.array([0.0, 0.0, 0.0, 1.0]), h)[1] == "replace"

    def test_a_fresh_asset_is_left_alone_on_a_short_horizon(self) -> None:
        m = small_tabular()
        assert exact_value(m, np.array([1.0, 0.0, 0.0, 0.0]), 2)[1] == "continue"

    def test_bad_inputs_are_refused(self) -> None:
        m = small_tabular()
        with pytest.raises(ValueError):
            exact_value(m, np.array([0.5, 0.5]), 3)
        with pytest.raises(ValueError):
            exact_value(m, np.array([0.5, 0.2, 0.2, 0.2]), 3)
        with pytest.raises(ValueError):
            exact_value(m, np.array([-0.1, 0.6, 0.3, 0.2]), 3)
        with pytest.raises(ValueError):
            exact_value(m, np.array([1.0, 0.0, 0.0, 0.0]), 0)


class TestPolicyValue:
    def test_the_optimal_policy_achieves_the_exact_value(self) -> None:
        m = small_tabular()
        b = a_belief(4)

        def optimal(belief: np.ndarray, left: int) -> int:
            return int(np.argmin(exact_q_values(m, belief, left)))

        assert policy_value(m, b, 4, optimal) == pytest.approx(exact_value(m, b, 4)[0])

    @pytest.mark.parametrize("action", [ACTION_CONTINUE, ACTION_INSPECT, ACTION_REPLACE])
    def test_a_fixed_policy_costs_at_least_the_optimum(self, action: int) -> None:
        m = small_tabular()
        b = a_belief(6)
        v = policy_value(m, b, 4, lambda belief, left: action)
        assert v >= exact_value(m, b, 4)[0] - 1e-9

    def test_always_continue_by_hand_on_two_steps(self) -> None:
        m = small_tabular()
        b = np.array([0.0, 0.0, 1.0, 0.0])
        v = policy_value(m, b, 2, lambda belief, left: ACTION_CONTINUE)
        # level 2 reaches the failure state in one move w.p. 0.2; a failed asset stays failed
        p_failed_1 = 0.2
        p_failed_2 = p_failed_1 + (1.0 - p_failed_1) * 0.2
        assert v == pytest.approx(8.0 * p_failed_1 + 8.0 * p_failed_2)

    def test_a_bad_choice_is_refused(self) -> None:
        m = small_tabular()
        with pytest.raises(ValueError):
            policy_value(m, a_belief(0), 2, lambda belief, left: 7)
