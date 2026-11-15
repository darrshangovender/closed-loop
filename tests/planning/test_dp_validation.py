"""P8b: the planner against exact dynamic programming (the validation test of 0001 section 8).

One asset, five hidden levels (the last an absorbing failure), four symbols, three actions, horizon
5; 50 random beliefs; the exact value by expectimax over the whole belief tree. The planner runs
at its default particle count and its default settings, with no chance constraint (the DP has
none). The targets are those of 0001 section 8, set before anything was run, and are not widened:

* the planner's value within 5% of the exact value on the 50 beliefs. "Planner value" is the cost
  its receding-horizon policy actually achieves (summed exactly over the belief tree), and the
  literal reading is used: every one of the 50 beliefs;
* agreement with the exact best first action on at least 90% of the beliefs whose best and
  second-best actions differ by more than 2% of the value.

``closed_loop.planning.validation`` holds the definitions and the comparison. The run is
deterministic. How it came out, and at other particle counts and planner seeds, is in
``docs/adr/progress/P8b-planner.md`` and ``P8b-dp-*.json``.
"""

import numpy as np
import pytest

from closed_loop.planning.exactdp import exact_value, validation_beliefs, validation_model
from closed_loop.planning.validation import (
    AGREEMENT_TARGET,
    VALUE_TOLERANCE,
    ValidationResult,
    run_validation,
)


@pytest.fixture(scope="module")
def result() -> ValidationResult:
    return run_validation(128, seed=0)


class TestTheInstance:
    def test_the_exact_best_action_is_mixed_so_that_agreement_means_something(self) -> None:
        m = validation_model()
        best = [exact_value(m, b, 5)[1] for b in validation_beliefs(50)]
        for action in ("continue", "inspect", "replace"):
            assert best.count(action) >= 5, (action, best.count(action))

    def test_most_beliefs_have_a_decisive_best_action(self, result: ValidationResult) -> None:
        assert result.n == 50 and result.n_decisive >= 35


class TestTheRun:
    def test_no_policy_beats_the_optimum(self, result: ValidationResult) -> None:
        """The exact value is a lower bound on any policy's cost: a check on both sides."""
        assert result.policy_errors.min() >= -1e-9

    def test_it_is_deterministic(self, result: ValidationResult) -> None:
        again = run_validation(128, n_beliefs=6, seed=0)
        assert [b.policy_value for b in again.beliefs] == [
            b.policy_value for b in result.beliefs[:6]
        ]

    def test_the_mean_error_of_the_achieved_value_is_small(self, result: ValidationResult) -> None:
        # a regression guard, not a target: it came out near 1% when this was written
        assert float(np.mean(result.policy_errors)) < 0.03

    def test_the_planners_own_estimate_is_the_less_reliable_number(
        self, result: ValidationResult
    ) -> None:
        # the estimate is the planner's belief about its value: sampled, optimistic at this N.
        # It is reported next to the achieved value and not held to the target.
        assert result.n_estimate_within <= result.n_policy_within


class TestTargets:
    """The stated targets, at the default particle count and planner seed."""

    def test_agreement_with_the_exact_best_action_is_at_least_90_percent(
        self, result: ValidationResult
    ) -> None:
        assert result.agreement >= AGREEMENT_TARGET, (
            f"{result.n_agree} of {result.n_decisive} decisive beliefs"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "target missed at the default N = 128: 49 of 50 beliefs within 5%, the worst at 5.3% "
            "(planner seeds 0 to 4: 48 or 49 of 50, worst 5.2% to 7.5%); met at N = 256 on 3 of 5 "
            "seeds. Recorded in docs/adr/progress/P8b-planner.md; the target is not widened"
        ),
    )
    def test_the_achieved_value_is_within_5_percent_of_exact_on_every_belief(
        self, result: ValidationResult
    ) -> None:
        worst = float(np.max(result.policy_errors))
        assert result.value_target_met, (
            f"{result.n_policy_within} of {result.n} beliefs within {VALUE_TOLERANCE:.0%}; "
            f"worst {worst:.1%}"
        )
